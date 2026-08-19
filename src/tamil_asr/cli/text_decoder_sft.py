from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from tamil_asr.config import load_yaml
from tamil_asr.data.text_corpus import ParquetTextBlockStream, TextBlockCollator, discover_text_parquet_files
from tamil_asr.evaluation.text import (
    collect_validation_blocks_by_root,
    generate_validation_examples,
    parameter_dtype_summary,
)
from tamil_asr.model.qwen3_asr import load_model_for_lora
from tamil_asr.training.artifacts import record_metrics
from tamil_asr.training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from tamil_asr.training.diagnostics import cuda_memory_metrics, gpu_activity_metrics, gradient_diagnostics


LLM_GROUP = "llm"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tamil text-only Qwen decoder LoRA adaptation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="", help="Checkpoint path, or 'latest'")
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Required guard: this command performs optimization and must be explicitly authorized",
    )
    return parser.parse_args()


def _validate_config(config: dict[str, Any]) -> None:
    for key in ("seed", "model", "data", "training"):
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    model = config["model"]
    if model.get("dtype") != "bfloat16":
        raise ValueError("Tamil decoder SFT requires model.dtype=bfloat16")
    if model.get("audio_lora", {}).get("enabled", False):
        raise ValueError("Text decoder SFT must disable audio_lora; only the LLM LoRA is trainable")
    if "adapter_init" not in model or not str(model["adapter_init"]).strip():
        raise ValueError("model.adapter_init must point to a compatible existing adapter")
    data = config["data"]
    if not data.get("roots"):
        raise ValueError("data.roots must list at least one Parquet corpus root")
    if int(data.get("block_size", 0)) <= 0:
        raise ValueError("data.block_size must be positive")
    training = config["training"]
    for key in ("output_dir", "gradient_accumulation_steps", "learning_rate", "warmup_steps", "max_steps"):
        if key not in training:
            raise ValueError(f"training.{key} is required")
    if int(training["max_steps"]) <= 0:
        raise ValueError("training.max_steps must be positive")
    if int(training["gradient_accumulation_steps"]) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")


def _schedule(optimizer: Any, warmup_steps: int, total_steps: int) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    batch.pop("_metadata", None)
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _evaluate(model: Any, loader: Any, device: Any, max_batches: int) -> float:
    import torch

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch_index, packed in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_batch(dict(packed), device)
            tokens = int((batch["labels"] != -100).sum().item())
            output = model(**batch)
            weighted_loss += float(output.loss.float().item()) * tokens
            token_count += tokens
    model.train()
    if token_count == 0:
        raise ValueError("Validation stream produced no text blocks")
    return weighted_loss / token_count


def _record_text_validation_examples(output_dir: Path, step: int, rows: list[dict[str, Any]]) -> Path:
    directory = output_dir / "evaluations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"text-step-{step:06d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _decoder_lora_parameters(model: Any) -> list[Any]:
    try:
        from tamil_asr.model.qwen3_asr import lora_parameter_groups
    except ImportError:
        lora_parameter_groups = None
    if lora_parameter_groups is not None:
        groups = lora_parameter_groups(model)
        parameters = groups.get(LLM_GROUP, [])
    else:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "audio_tower" not in name
        ]
    if not parameters:
        raise RuntimeError("The existing LoRA loader produced no trainable decoder parameters")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any("audio_tower" in name for name in trainable_names):
        raise AssertionError("Text decoder SFT unexpectedly has trainable audio parameters")
    non_lora = [
        name
        for name in trainable_names
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if non_lora:
        raise AssertionError(f"Non-LoRA parameters are trainable: {non_lora[:3]}")
    outside_decoder = [name for name in trainable_names if "model.layers." not in name]
    if outside_decoder:
        raise AssertionError(f"Trainable LoRA parameter is outside the text decoder: {outside_decoder[:3]}")
    return parameters


def main() -> None:
    args = _parse_args()
    if not args.allow_training:
        raise SystemExit("Training guard active; rerun with --allow-training after approving the run")
    config = load_yaml(args.config)
    _validate_config(config)

    import numpy as np
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Tamil decoder SFT requires an Ampere-or-newer CUDA GPU")

    training = config["training"]
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("Text decoder SFT is currently validated for one GPU")

    model_cfg = config["model"]
    resume_value = args.resume.strip()
    output_dir = Path(training["output_dir"])
    resume_path = latest_checkpoint(output_dir) if resume_value == "latest" else Path(resume_value) if resume_value else None
    if resume_value and (resume_path is None or not resume_path.is_dir()):
        raise FileNotFoundError(f"No complete text-SFT checkpoint found: {resume_value}")
    adapter_path = resume_path / "adapter" if resume_path else Path(model_cfg["adapter_init"])
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter initialization path does not exist: {adapter_path}")

    loader_kwargs = {"adapter_path": adapter_path}
    if "trainable_groups" in inspect.signature(load_model_for_lora).parameters:
        loader_kwargs["trainable_groups"] = [LLM_GROUP]
    model, processor, model_report = load_model_for_lora(model_cfg, **loader_kwargs)
    model_report.update(parameter_dtype_summary(model))
    files = discover_text_parquet_files(config["data"]["roots"])
    data_cfg = config["data"]
    train_stream = ParquetTextBlockStream(
        files,
        processor.tokenizer,
        int(data_cfg["block_size"]),
        "train",
        float(data_cfg.get("validation_fraction", 0.01)),
        int(data_cfg.get("parquet_batch_size", 512)),
        int(data_cfg.get("max_text_chars", 0)),
    )
    validation_stream = ParquetTextBlockStream(
        files,
        processor.tokenizer,
        int(data_cfg["block_size"]),
        "validation",
        float(data_cfg.get("validation_fraction", 0.01)),
        int(data_cfg.get("parquet_batch_size", 512)),
        int(data_cfg.get("max_text_chars", 0)),
    )
    batch_size = int(data_cfg.get("batch_size", 1))
    collator = TextBlockCollator()
    train_loader = DataLoader(train_stream, batch_size=batch_size, collate_fn=collator, num_workers=0)
    validation_loader = DataLoader(validation_stream, batch_size=batch_size, collate_fn=collator, num_workers=0)

    decoder_parameters = _decoder_lora_parameters(model)
    optimizer = torch.optim.AdamW(
        [{"params": decoder_parameters, "lr": float(training["learning_rate"])}],
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )
    scheduler = _schedule(optimizer, int(training["warmup_steps"]), int(training["max_steps"]))
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    device = accelerator.device

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_identity.json").write_text(
            json.dumps(
                {
                    "recipe_name": "tamil_decoder_text_sft",
                    "initialization_adapter": str(adapter_path),
                    "parquet_files": [str(path) for path in files],
                    "active_trainable_groups": [LLM_GROUP],
                    "model": model_report,
                    "data": data_cfg,
                    "training": training,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    wandb_run = None
    wandb_cfg = config.get("wandb", {})
    if accelerator.is_main_process and wandb_cfg.get("enabled", False):
        import wandb

        authenticated = wandb.login(anonymous="never", relogin=False, verify=True)
        if not authenticated:
            raise RuntimeError(
                "Weights & Biases is enabled but no authenticated credentials were found on the server"
            )
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg.get("run_name"),
            config={"training": config, "model": model_report, "parquet_files": [str(path) for path in files]},
            save_code=False,
        )
        (output_dir / "wandb_run.json").write_text(
            json.dumps({"id": wandb_run.id, "url": wandb_run.get_url()}, indent=2) + "\n",
            encoding="utf-8",
        )

    gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
    train_iterator = iter(train_loader)
    global_step = 0
    microbatches_consumed = 0
    if resume_path:
        state = load_checkpoint(accelerator, optimizer, scheduler, resume_path)
        global_step = int(state["global_step"])
        microbatches_consumed = int(
            state.get("microbatches_consumed", global_step * gradient_accumulation_steps)
        )
        for _ in range(microbatches_consumed):
            try:
                next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                next(train_iterator)
    log_every = int(training.get("log_every_steps", 100))
    eval_every = int(training.get("eval_every_steps", 500))
    save_every = int(training.get("save_every_steps", 500))
    max_eval_batches = int(training.get("max_eval_batches", 32))
    window_loss = 0.0
    window_tokens = 0
    window_microbatches = 0
    window_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(training["max_steps"]):
        try:
            packed = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            packed = next(train_iterator)
        microbatches_consumed += 1
        batch = _move_batch(dict(packed), device)
        grad_metrics: dict[str, float] = {}
        pre_clip_grad_norm: float | None = None
        with accelerator.accumulate(model):
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite text loss at optimizer step {global_step}")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_metrics = gradient_diagnostics(model)
                if grad_metrics["grad/nonfinite_parameters"]:
                    raise FloatingPointError(f"Non-finite text gradients at optimizer step {global_step}")
                clipped_norm = accelerator.clip_grad_norm_(
                    model.parameters(), float(training.get("max_grad_norm", 1.0))
                )
                pre_clip_grad_norm = float(clipped_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        window_loss += float(loss.detach().float().item())
        window_tokens += int((batch["labels"] != -100).sum().item())
        window_microbatches += 1
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        if accelerator.is_main_process and (global_step == 1 or global_step % log_every == 0):
            elapsed = max(time.perf_counter() - window_started, 1e-6)
            train_log = {
                "train/loss": window_loss / max(1, window_microbatches),
                "train/perplexity": math.exp(min(20.0, window_loss / max(1, window_microbatches))),
                "train/learning_rate": float(scheduler.get_last_lr()[0]),
                "train/supervised_tokens": window_tokens,
                "train/tokens_per_second": window_tokens / elapsed,
                **grad_metrics,
                "grad/pre_clip_global_norm": pre_clip_grad_norm,
                **cuda_memory_metrics(),
                **gpu_activity_metrics(),
            }
            record_metrics(
                output_dir,
                global_step,
                "text_train",
                train_log,
            )
            if wandb_run is not None:
                wandb_run.log(train_log, step=global_step)
            window_loss = 0.0
            window_tokens = 0
            window_microbatches = 0
            window_started = time.perf_counter()

        if global_step % eval_every == 0 or global_step == int(training["max_steps"]):
            validation_loss = _evaluate(model, validation_loader, device, max_eval_batches)
            if accelerator.is_main_process:
                example_limit = int(wandb_cfg.get("log_examples", training.get("validation_examples", 0)))
                validation_blocks = collect_validation_blocks_by_root(
                    data_cfg["roots"],
                    processor.tokenizer,
                    block_size=int(data_cfg["block_size"]),
                    validation_fraction=float(data_cfg.get("validation_fraction", 0.01)),
                    parquet_batch_size=int(data_cfg.get("parquet_batch_size", 512)),
                    max_text_chars=int(data_cfg.get("max_text_chars", 0)),
                    batch_size=batch_size,
                    max_examples=example_limit,
                )
                validation_examples = generate_validation_examples(
                    accelerator.unwrap_model(model),
                    processor.tokenizer,
                    validation_blocks,
                    device,
                    context_tokens=int(training.get("validation_context_tokens", 128)),
                    max_new_tokens=int(training.get("validation_max_new_tokens", 64)),
                ) if example_limit > 0 else []
                record_metrics(
                    output_dir,
                    global_step,
                    "text_validation",
                    {
                        "validation/loss": validation_loss,
                        "validation/perplexity": math.exp(min(20.0, validation_loss)),
                    },
                )
                if validation_examples:
                    _record_text_validation_examples(output_dir, global_step, validation_examples)
                if wandb_run is not None:
                    validation_log = {
                        "validation/loss": validation_loss,
                        "validation/perplexity": math.exp(min(20.0, validation_loss)),
                    }
                    wandb_run.log(validation_log, step=global_step)
                    if validation_examples:
                        import wandb

                        table = wandb.Table(
                            columns=[
                                "example_index",
                                "source",
                                "prompt",
                                "reference_continuation",
                                "hypothesis_continuation",
                                "prompt_tokens",
                                "reference_tokens",
                                "generated_tokens",
                            ]
                        )
                        for row in validation_examples:
                            table.add_data(
                                row["example_index"],
                                row["source"],
                                row["prompt"],
                                row["reference_continuation"],
                                row["hypothesis_continuation"],
                                row["prompt_tokens"],
                                row["reference_tokens"],
                                row["generated_tokens"],
                            )
                        wandb_run.log(
                            {f"validation/text_examples_step_{global_step:06d}": table},
                            step=global_step,
                        )

        if save_every > 0 and global_step % save_every == 0:
            save_checkpoint(
                accelerator,
                model,
                processor,
                optimizer,
                scheduler,
                output_dir,
                global_step,
                0,
                {
                    "recipe_name": "tamil_decoder_text_sft",
                    "stage_name": "tamil_text_adaptation",
                    "active_trainable_groups": [LLM_GROUP],
                    "microbatches_consumed": microbatches_consumed,
                    "initialization_adapter": str(adapter_path),
                },
                int(training.get("keep_last_checkpoints", 2)),
            )

    if accelerator.is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        print(json.dumps({"status": "completed", "output_dir": str(output_dir), "step": global_step}))


if __name__ == "__main__":
    main()
