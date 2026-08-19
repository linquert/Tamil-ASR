from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tamil_asr.config import load_yaml, validate_training_config
from tamil_asr.data.collator import QwenASRCollator, validate_batch_contract
from tamil_asr.data.manifest import ManifestRecord
from tamil_asr.data.manifest_io import load_manifest, summarize_dataset
from tamil_asr.data.parquet_dataset import ParquetAudioDataset
from tamil_asr.data.sampler import DurationBatchSampler
from tamil_asr.evaluation.generate import generate_transcriptions
from tamil_asr.evaluation.metrics import compute_asr_metrics
from tamil_asr.model.qwen3_asr import LLM_GROUP, load_model_for_lora
from tamil_asr.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    read_checkpoint_state,
    save_checkpoint,
)
from tamil_asr.training.artifacts import file_sha256, record_evaluation_rows, record_metrics
from tamil_asr.training.batch import extract_model_batch
from tamil_asr.training.diagnostics import cuda_memory_metrics, gpu_activity_metrics, gradient_diagnostics
from tamil_asr.training.evaluation import evaluate_loss
from tamil_asr.training.loss_normalization import LOSS_NORMALIZATION, normalize_accumulated_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety-checked Qwen3-ASR Tamil LoRA training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="", help="Checkpoint path, or 'latest'")
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Required guard: this command performs optimization and must be explicitly authorized",
    )
    return parser.parse_args()


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1 or count > length:
        raise ValueError(f"count must be in [1, {length}], found {count}")
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _stratified_indices(records: list[ManifestRecord], total: int) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_source[record.source_split.lower()].append(index)
    if total < len(by_source):
        raise ValueError("evaluation.num_examples must cover every validation source")
    total = min(int(total), len(records))
    allocations = {
        source: max(1, min(len(indices), round(total * len(indices) / len(records))))
        for source, indices in by_source.items()
    }
    while sum(allocations.values()) > total:
        source = max((key for key in allocations if allocations[key] > 1), key=allocations.get)
        allocations[source] -= 1
    while sum(allocations.values()) < total:
        candidates = [key for key, indices in by_source.items() if allocations[key] < len(indices)]
        source = max(candidates, key=lambda key: len(by_source[key]) - allocations[key])
        allocations[source] += 1
    selected: list[int] = []
    for source in sorted(by_source):
        indices = by_source[source]
        selected.extend(indices[position] for position in _evenly_spaced_indices(len(indices), allocations[source]))
    return selected


def _load_fixed_evaluation_indices(records: list[ManifestRecord], path: str | Path) -> list[int]:
    ids: list[str] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid hard-validation JSON at {path}:{line_number}") from exc
        identifier = value.get("id") if isinstance(value, dict) else value
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Hard-validation entry at {path}:{line_number} has no non-empty id")
        ids.append(identifier)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Hard-validation file contains duplicate ids: {path}")
    by_id = {record.id: index for index, record in enumerate(records)}
    missing = [identifier for identifier in ids if identifier not in by_id]
    if missing:
        raise ValueError(f"Hard-validation ids are absent from validation manifest: {missing[:3]}")
    return [by_id[identifier] for identifier in ids]


def _learning_rate_schedule(optimizer: Any, warmup: int, total: int, name: str) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    if name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    def factor(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def main() -> None:
    args = parse_args()
    if not args.allow_training:
        raise SystemExit(
            "Training guard active. Re-run with --allow-training only after the user approves the optimization run."
        )
    config = load_yaml(args.config)
    validate_training_config(config)

    import numpy as np
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    training_cfg = config["training"]
    gradient_accumulation_steps = int(training_cfg["gradient_accumulation_steps"])
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline is validated for the CUDA GPU server, not CPU training")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError(f"BF16 requires Ampere or newer CUDA GPU, found capability={capability}")

    data_cfg = config["data"]
    require_audited_audio = bool(data_cfg.get("require_audited_audio", True))
    train_records = load_manifest(
        data_cfg["train_manifest"], "train", require_audited_audio=require_audited_audio
    )
    validation_records = load_manifest(
        data_cfg["validation_manifest"], "validation", require_audited_audio=require_audited_audio
    )
    train_ids = {record.id for record in train_records}
    if train_ids.intersection(record.id for record in validation_records):
        raise ValueError("Manifest record leakage between train and validation")
    train_text = {record.text_fingerprint for record in train_records}
    if train_text.intersection(record.text_fingerprint for record in validation_records):
        raise ValueError("Transcript leakage between train and validation")

    evaluation_cfg = config["evaluation"]
    hard_examples_file = evaluation_cfg.get("hard_examples_file")
    fixed_evaluation_indices: list[int] | None = None
    if bool(evaluation_cfg.get("enabled", True)) and hard_examples_file:
        hard_examples_path = Path(hard_examples_file)
        if not hard_examples_path.is_file():
            raise FileNotFoundError(f"Configured evaluation.hard_examples_file does not exist: {hard_examples_path}")
        fixed_evaluation_indices = _load_fixed_evaluation_indices(validation_records, hard_examples_path)
        expected_examples = int(evaluation_cfg.get("num_examples", len(fixed_evaluation_indices)))
        if len(fixed_evaluation_indices) != expected_examples:
            raise ValueError(
                "evaluation.hard_examples_file size must equal evaluation.num_examples: "
                f"{len(fixed_evaluation_indices)} != {expected_examples}"
            )

    output_dir = Path(training_cfg["output_dir"])
    resume = args.resume.strip()
    resume_path = None
    if resume:
        resume_path = latest_checkpoint(output_dir) if resume == "latest" else Path(resume)
        if resume_path is None or not resume_path.is_dir():
            raise FileNotFoundError("No complete checkpoint found to resume")
        resume_state = read_checkpoint_state(resume_path)
        if resume_state.get("loss_normalization") != LOSS_NORMALIZATION:
            raise ValueError(
                "Refusing to resume optimizer state from a checkpoint created with a different loss "
                "normalization. Restart from the intended adapter initialization instead."
            )
    model_cfg = config["model"]
    if bool(model_cfg.get("audio_lora", {}).get("enabled", False)):
        raise ValueError("The integrated Tamil decoder recipe must keep model.audio_lora.enabled=false")
    frozen_adapter_path = model_cfg.get("frozen_adapter_path")
    if frozen_adapter_path and not Path(frozen_adapter_path).is_dir() and resume_path is None:
        raise FileNotFoundError(
            f"Configured model.frozen_adapter_path does not exist: {frozen_adapter_path}"
        )
    adapter_init = Path(model_cfg["adapter_init"]) if model_cfg.get("adapter_init") else None
    if adapter_init is not None and not adapter_init.is_dir() and resume_path is None:
        raise FileNotFoundError(f"Configured model.adapter_init does not exist: {adapter_init}")
    initialization_adapter = resume_path / "adapter" if resume_path else adapter_init
    model, processor, model_report = load_model_for_lora(
        model_cfg,
        initialization_adapter,
        trainable_groups=[LLM_GROUP],
    )
    collator = QwenASRCollator(processor, data_cfg["prompt"], int(data_cfg["sample_rate"]))
    train_dataset = ParquetAudioDataset(train_records, int(data_cfg["sample_rate"]))
    validation_dataset = ParquetAudioDataset(validation_records, int(data_cfg["sample_rate"]))
    train_sampler = DurationBatchSampler(
        [record.duration for record in train_records],
        float(data_cfg["max_batch_audio_seconds"]),
        int(data_cfg["max_batch_size"]),
        seed=seed,
        shuffle=True,
    )
    validation_sampler = DurationBatchSampler(
        [record.duration for record in validation_records],
        float(data_cfg["max_batch_audio_seconds"]),
        int(data_cfg["max_batch_size"]),
        seed=seed,
        shuffle=False,
    )
    workers = int(data_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(training_cfg["learning_rate"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "betas": (0.9, 0.95),
    }
    if training_cfg.get("fused_adamw", False):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        **optimizer_kwargs,
    )
    scheduler = _learning_rate_schedule(
        optimizer,
        int(training_cfg["warmup_steps"]),
        int(training_cfg["max_steps"]),
        str(training_cfg.get("lr_schedule", "cosine")),
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("This first-run checkpoint/evaluation path is intentionally single-GPU only")

    manifest_identity = {
        "loss_normalization": LOSS_NORMALIZATION,
        "train_manifest_sha256": file_sha256(data_cfg["train_manifest"]),
        "validation_manifest_sha256": file_sha256(data_cfg["validation_manifest"]),
        "train": summarize_dataset(train_records),
        "validation": summarize_dataset(validation_records),
        "model": model_report,
        "require_audited_audio": require_audited_audio,
        "initialization_adapter": str(initialization_adapter) if initialization_adapter else None,
        "frozen_adapter_path": str(frozen_adapter_path) if frozen_adapter_path else None,
        "evaluation": {
            "max_new_tokens": int(evaluation_cfg["max_new_tokens"]),
            "hard_examples_file": str(hard_examples_file) if hard_examples_file else None,
            "hard_examples_sha256": file_sha256(hard_examples_file) if hard_examples_file else None,
        },
    }
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_identity.json").write_text(
            json.dumps(manifest_identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    wandb_run = None
    wandb_cfg = config["wandb"]
    if accelerator.is_main_process and wandb_cfg.get("enabled", False):
        import wandb

        authenticated = wandb.login(anonymous="never", relogin=False, verify=True)
        if not authenticated:
            raise RuntimeError(
                "W&B is enabled but authentication failed. Run `wandb login` on the GPU server before training."
            )
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg.get("run_name"),
            config={"training": config, "identity": manifest_identity},
            save_code=False,
        )
        (output_dir / "wandb_run.json").write_text(
            json.dumps({"id": wandb_run.id, "url": wandb_run.get_url()}, indent=2) + "\n",
            encoding="utf-8",
        )

    global_step = 0
    start_epoch = 0
    resume_batch_in_epoch = 0
    best_metric_value: float | None = None
    if resume_path:
        state = load_checkpoint(accelerator, optimizer, scheduler, resume_path)
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        resume_batch_in_epoch = int(state.get("next_batch_in_epoch", 0))
        saved_best = state.get("best_metric_value")
        best_metric_value = float(saved_best) if saved_best is not None else None

    max_steps = int(training_cfg["max_steps"])
    contract_steps = int(training_cfg.get("validate_batch_contract_steps", 1))
    log_every = int(training_cfg.get("log_every_steps", 5))
    save_every = int(training_cfg.get("save_every_steps", 25))
    eval_every = int(training_cfg.get("eval_every_steps", 25))
    evaluation_enabled = bool(config["evaluation"].get("enabled", True))
    profile_timing = bool(training_cfg.get("profile_timing", False))
    best_metric_name = str(training_cfg.get("save_best_metric", "")).strip()
    best_metric_mode = str(training_cfg.get("save_best_mode", "min"))
    save_final_regular = bool(training_cfg.get("save_final_checkpoint", True))
    if best_metric_mode not in {"min", "max"}:
        raise ValueError("training.save_best_mode must be 'min' or 'max'")
    if best_metric_name and not evaluation_enabled:
        raise ValueError("training.save_best_metric requires evaluation.enabled=true")
    optimizer.zero_grad(set_to_none=True)
    epoch = start_epoch
    first_gradient_checked = global_step > 0
    accumulation_supervised_tokens = 0
    window_audio_seconds = 0.0
    window_supervised_tokens = 0
    window_token_loss_sum = 0.0
    window_microbatch_loss_sum = 0.0
    window_microbatches = 0
    window_data_wait_seconds = 0.0
    window_compute_seconds = 0.0
    window_started = time.perf_counter()
    previous_batch_finished = window_started
    while global_step < max_steps:
        train_sampler.set_epoch(epoch)
        for batch_in_epoch, packed in enumerate(train_loader):
            batch_ready = time.perf_counter()
            data_wait_seconds = batch_ready - previous_batch_finished
            if epoch == start_epoch and batch_in_epoch < resume_batch_in_epoch:
                previous_batch_finished = batch_ready
                continue
            batch, metadata = extract_model_batch(dict(packed))
            if global_step < contract_steps:
                contract = validate_batch_contract(accelerator.unwrap_model(model), batch)
                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log({f"contract/{key}": value for key, value in contract.items()}, step=global_step)
            microbatch_audio_seconds = sum(float(row["duration"]) for row in metadata)
            microbatch_supervised_tokens = int((batch["labels"] != -100).sum().item())
            if microbatch_supervised_tokens <= 0:
                raise ValueError("Training microbatch contains no supervised tokens")
            accumulation_supervised_tokens += microbatch_supervised_tokens
            if profile_timing:
                torch.cuda.synchronize()
            compute_started = time.perf_counter()
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at optimizer step {global_step}")
                accelerator.backward(loss * microbatch_supervised_tokens)
                grad_metrics: dict[str, float] = {}
                gradient_normalization_scale: float | None = None
                if accelerator.sync_gradients:
                    gradient_normalization_scale = normalize_accumulated_gradients(
                        model,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        supervised_tokens=accumulation_supervised_tokens,
                    )
                    grad_metrics = gradient_diagnostics(model)
                    if grad_metrics["grad/nonfinite_parameters"]:
                        raise FloatingPointError(f"Non-finite gradients at optimizer step {global_step}")
                    if not first_gradient_checked and grad_metrics["grad/nonzero_parameters"] <= 0:
                        raise RuntimeError("First backward pass produced no non-zero LoRA gradients")
                    first_gradient_checked = True
                    accelerator.clip_grad_norm_(model.parameters(), float(training_cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if profile_timing:
                torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_started
            previous_batch_finished = time.perf_counter()

            window_audio_seconds += microbatch_audio_seconds
            window_supervised_tokens += microbatch_supervised_tokens
            microbatch_loss_value = float(loss.detach().float().item())
            window_token_loss_sum += microbatch_loss_value * microbatch_supervised_tokens
            window_microbatch_loss_sum += microbatch_loss_value
            window_microbatches += 1
            window_data_wait_seconds += data_wait_seconds
            window_compute_seconds += compute_seconds

            if not accelerator.sync_gradients:
                continue
            if gradient_normalization_scale is None:
                raise AssertionError("Synchronized optimizer step did not normalize gradients")
            optimizer_supervised_tokens = accumulation_supervised_tokens
            accumulation_supervised_tokens = 0
            global_step += 1
            elapsed = time.perf_counter() - window_started
            if window_supervised_tokens <= 0:
                raise RuntimeError("Completed optimizer step has no supervised tokens")
            if accelerator.is_main_process and (global_step == 1 or global_step % log_every == 0):
                log = {
                    "train/loss": window_token_loss_sum / window_supervised_tokens,
                    "train/microbatch_mean_loss": window_microbatch_loss_sum / max(window_microbatches, 1),
                    "train/learning_rate": float(scheduler.get_last_lr()[0]),
                    "train/audio_seconds_per_second": window_audio_seconds / max(elapsed, 1e-6),
                    "train/supervised_tokens": window_supervised_tokens,
                    "train/optimizer_supervised_tokens": optimizer_supervised_tokens,
                    "train/gradient_normalization_scale": gradient_normalization_scale,
                    "train/window_audio_seconds": window_audio_seconds,
                    "train/microbatches": window_microbatches,
                    "performance/data_wait_seconds": window_data_wait_seconds,
                    "performance/compute_seconds": window_compute_seconds,
                    "performance/data_wait_fraction": window_data_wait_seconds / max(elapsed, 1e-6),
                    "performance/compute_fraction": window_compute_seconds / max(elapsed, 1e-6),
                    **grad_metrics,
                    **cuda_memory_metrics(),
                    **gpu_activity_metrics(),
                }
                record_metrics(output_dir, global_step, "train", log)
                if training_cfg.get("console_metrics", False):
                    print(json.dumps({"step": global_step, **log}, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(log, step=global_step)
            window_audio_seconds = 0.0
            window_supervised_tokens = 0
            window_token_loss_sum = 0.0
            window_microbatch_loss_sum = 0.0
            window_microbatches = 0
            window_data_wait_seconds = 0.0
            window_compute_seconds = 0.0

            new_best = False
            if evaluation_enabled and (global_step % eval_every == 0 or global_step == max_steps):
                validation_loss = evaluate_loss(
                    model,
                    validation_loader,
                    int(evaluation_cfg.get("validation_loss_batches", 32)),
                )
                unwrapped = accelerator.unwrap_model(model)
                if fixed_evaluation_indices is not None:
                    indices = fixed_evaluation_indices
                else:
                    indices = _stratified_indices(
                        validation_records,
                        min(int(evaluation_cfg["num_examples"]), len(validation_dataset)),
                    )
                rows = generate_transcriptions(
                    unwrapped,
                    processor,
                    validation_dataset,
                    indices,
                    data_cfg["prompt"],
                    int(config["evaluation"]["max_new_tokens"]),
                ) if accelerator.is_main_process else []
                if accelerator.is_main_process:
                    metrics = compute_asr_metrics(rows)
                    eval_log = {
                        "validation/loss": validation_loss,
                        "validation/wer": metrics["overall"]["wer"],
                        "validation/cer": metrics["overall"]["cer"],
                        "validation/max_new_tokens": int(evaluation_cfg["max_new_tokens"]),
                    }
                    cap_hits = sum(bool(row.get("hit_max_new_tokens", False)) for row in rows)
                    generated_token_counts = [int(row["generated_tokens"]) for row in rows]
                    eval_log.update(
                        {
                            "validation/generated_tokens_mean": (
                                sum(generated_token_counts) / max(len(generated_token_counts), 1)
                            ),
                            "validation/generation_cap_hits": cap_hits,
                            "validation/generation_cap_hit_rate": cap_hits / max(len(rows), 1),
                        }
                    )
                    for dataset_name, dataset_metrics in metrics.items():
                        if dataset_name != "overall":
                            eval_log[f"validation_by_dataset/{dataset_name}_wer"] = dataset_metrics["wer"]
                            eval_log[f"validation_by_dataset/{dataset_name}_cer"] = dataset_metrics["cer"]
                    record_metrics(output_dir, global_step, "validation", eval_log)
                    record_evaluation_rows(output_dir, global_step, rows)
                    if training_cfg.get("console_metrics", False):
                        print(json.dumps({"step": global_step, **eval_log}, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(eval_log, step=global_step)
                        import wandb

                        example_limit = min(int(wandb_cfg.get("log_examples", 0)), len(rows))
                        table = wandb.Table(
                            columns=[
                                "id",
                                "dataset",
                                "domain",
                                "duration",
                                "reference",
                                "hypothesis",
                                "generated_tokens",
                                "hit_max_new_tokens",
                            ]
                        )
                        for row in rows[:example_limit]:
                            table.add_data(
                                row["id"],
                                row["dataset"],
                                row["domain"],
                                row["duration"],
                                row["reference"],
                                row["hypothesis"],
                                row["generated_tokens"],
                                row["hit_max_new_tokens"],
                            )
                        wandb_run.log({f"validation/examples_step_{global_step:06d}": table}, step=global_step)

                    if best_metric_name:
                        if best_metric_name not in eval_log:
                            raise KeyError(
                                f"training.save_best_metric={best_metric_name!r} was not produced by evaluation"
                            )
                        candidate = float(eval_log[best_metric_name])
                        if not math.isfinite(candidate):
                            raise FloatingPointError(f"Non-finite best-checkpoint metric at step {global_step}")
                        new_best = best_metric_value is None or (
                            candidate < best_metric_value
                            if best_metric_mode == "min"
                            else candidate > best_metric_value
                        )
                        if new_best:
                            best_metric_value = candidate

            checkpoint_metadata = {
                **manifest_identity,
                "next_batch_in_epoch": batch_in_epoch + 1,
                "best_metric_name": best_metric_name or None,
                "best_metric_value": best_metric_value,
            }
            if new_best:
                save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step,
                    epoch,
                    checkpoint_metadata,
                    int(training_cfg.get("keep_last_checkpoints", 3)),
                    name="best",
                )
            elif save_every > 0 and (
                global_step % save_every == 0 or (global_step == max_steps and save_final_regular)
            ):
                save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step,
                    epoch,
                    checkpoint_metadata,
                    int(training_cfg.get("keep_last_checkpoints", 3)),
                )
            window_started = time.perf_counter()
            if global_step >= max_steps:
                break
        epoch += 1

    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
