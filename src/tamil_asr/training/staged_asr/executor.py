

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tamil_asr.data.collator import QwenASRCollator, validate_batch_contract
from tamil_asr.data.manifest import ManifestRecord, filter_records_by_base_wer
from tamil_asr.data.manifest_io import load_manifest, summarize_dataset
from tamil_asr.data.parquet_dataset import ParquetAudioDataset
from tamil_asr.data.sampler import DurationBatchSampler
from tamil_asr.evaluation.generate import generate_transcriptions
from tamil_asr.evaluation.metrics import compute_asr_metrics
from tamil_asr.model.qwen3_asr import (
    AUDIO_GROUP,
    LLM_GROUP,
    _adapter_target_from_parameter_name,
    load_model_for_lora,
    lora_parameter_groups,
)
from tamil_asr.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_stage_state,
)
from tamil_asr.training.artifacts import file_sha256, record_evaluation_rows, record_metrics
from tamil_asr.training.batch import extract_model_batch
from tamil_asr.training.diagnostics import cuda_memory_metrics, gpu_activity_metrics
from tamil_asr.training.evaluation import evaluate_loss
from tamil_asr.training.loss_normalization import LOSS_NORMALIZATION, normalize_accumulated_gradients
from tamil_asr.training.staged_asr.recipe import Stage, StageRecipe
from tamil_asr.training.staged_asr.run import RuntimeOptions, StageRun
from tamil_asr.training.staged_asr.model import model_config_for_run
from tamil_asr.training.staged_asr.checkpoint import resolve_training_initialization


@dataclass
class CurriculumSegment:
    index: int
    max_base_wer: float | None
    records: list[ManifestRecord]
    optimizer_steps: int
    loader: Any | None = None
    sampler: DurationBatchSampler | None = None


@dataclass
class PreparedTrainingRun:
    recipe: StageRecipe
    run: StageRun
    accelerator: Any
    model: Any
    processor: Any
    optimizer: Any
    scheduler: Any
    validation_records: list[ManifestRecord]
    validation_loader: Any
    segments: list[CurriculumSegment]
    max_steps: int
    identity: dict[str, Any]
    initialization_checkpoint: str | None
    merged_adapter_paths: list[Path]
    resume_path: Path | None
    resume_state: dict[str, Any] | None
    same_stage_resume: bool


@dataclass
class TrainingProgress:
    global_step: int = 0
    resume_segment: int = 0
    resume_segment_epoch: int = 0
    resume_batch: int = 0
    best_metric_value: float | None = None
    restored_state: dict[str, Any] | None = None


def _build_loader(
    records: list[ManifestRecord],
    collator: QwenASRCollator,
    runtime: RuntimeOptions,
    seed: int,
    shuffle: bool,
) -> tuple[Any, DurationBatchSampler]:
    from torch.utils.data import DataLoader

    dataset = ParquetAudioDataset(records, runtime.sample_rate)
    sampler = DurationBatchSampler(
        [record.duration for record in records],
        runtime.max_batch_audio_seconds,
        runtime.max_batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    workers = runtime.num_workers
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=runtime.prefetch_factor if workers > 0 else None,
    )
    return loader, sampler


def _build_curriculum(
    records: list[ManifestRecord], recipe: StageRecipe, max_steps: int | None
) -> list[CurriculumSegment]:
    thresholds = list(recipe.curriculum_thresholds)
    fractions = list(recipe.curriculum_fractions)
    if not fractions:
        fractions = [1.0 / len(thresholds)] * len(thresholds)
    if max_steps is None:
        steps = [0] * len(thresholds)
    else:
        if max_steps < len(thresholds):
            raise ValueError("training.max_steps must be at least the number of curriculum segments")
        steps = [max(1, int(max_steps * fraction)) for fraction in fractions]
        if sum(steps) > max_steps:
            raise ValueError("Curriculum segment fractions produce more steps than training.max_steps")
        steps[-1] += max_steps - sum(steps)

    segments = []
    for index, (threshold, segment_steps) in enumerate(zip(thresholds, steps)):
        selected = filter_records_by_base_wer(records, threshold)
        segments.append(
            CurriculumSegment(
                index=index,
                max_base_wer=threshold,
                records=selected,
                optimizer_steps=segment_steps,
            )
        )
    return segments


def _group_gradient_metrics(model: Any, active_groups: list[str]) -> dict[str, float]:
    import torch

    norms = {"audio": 0.0, "llm": 0.0}
    counts = {"audio": 0, "llm": 0}
    for name, parameter in model.named_parameters():
        target = _adapter_target_from_parameter_name(name)
        if target is None:
            if parameter.grad is not None:
                raise AssertionError(f"Frozen base parameter received a gradient: {name}")
            continue
        if target.startswith("audio_tower."):
            group = AUDIO_GROUP
        elif target.startswith("model.layers."):
            group = LLM_GROUP
        else:
            raise AssertionError(f"Unsupported LoRA target in gradient check: {target}")
        if parameter.grad is not None:
            if group not in active_groups:
                raise AssertionError(f"Frozen {group} LoRA parameter received a gradient: {name}")
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite {group} gradient: {name}")
            norms[group] += float(gradient.float().norm().item()) ** 2
            counts[group] += 1
    for group in active_groups:
        if counts[group] == 0 or norms[group] <= 0.0:
            raise RuntimeError(f"Active {group} LoRA group produced no non-zero gradient")
    return {
        f"grad/{group}_lora_norm": math.sqrt(norms[group])
        for group in (AUDIO_GROUP, LLM_GROUP)
    } | {
        f"grad/{group}_lora_parameters": float(counts[group])
        for group in (AUDIO_GROUP, LLM_GROUP)
    }


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


def _optimizer(model: Any, recipe: StageRecipe) -> Any:
    import torch

    active_groups = list(recipe.trainable_groups)
    groups = lora_parameter_groups(model)
    if any(not groups[group] for group in active_groups):
        raise AssertionError(f"A2S optimizer received no parameters for active groups: {active_groups}")
    optimizer_kwargs: dict[str, Any] = {
        "weight_decay": recipe.weight_decay,
        "betas": (0.9, 0.95),
    }
    return torch.optim.AdamW(
        [
            {
                "name": group,
                "params": groups[group],
                "lr": recipe.learning_rate_by_group[group],
            }
            for group in active_groups
        ],
        **optimizer_kwargs,
    )


def _stage_identity(
    run: StageRun,
    recipe: StageRecipe,
    segments: list[CurriculumSegment],
    model_report: dict[str, Any],
) -> dict[str, Any]:
    active_groups = list(recipe.trainable_groups)
    stage = recipe.stage
    return {
        "loss_normalization": LOSS_NORMALIZATION,
        "recipe_name": "megaasr_a2s_sft",
        "stage": stage.value,
        "phase_name": stage.phase_name,
        "active_trainable_groups": active_groups,
        "learning_rates": {
            group: recipe.learning_rate_by_group[group] for group in active_groups
        },
        "train_manifest_sha256": file_sha256(run.data.train_manifest),
        "validation_manifest_sha256": file_sha256(run.data.validation_manifest),
        "curriculum": [
            {
                "index": segment.index,
                "max_base_wer": segment.max_base_wer,
                "examples": len(segment.records),
                "hours": sum(record.duration for record in segment.records) / 3600.0,
                "optimizer_steps": segment.optimizer_steps,
            }
            for segment in segments
        ],
        "validation": {},
        "model": model_report,
    }


def _build_stage_segments(
    records: list[ManifestRecord], recipe: StageRecipe, max_steps: int | None
) -> list[CurriculumSegment]:
    stage = recipe.stage
    if stage is Stage.ENCODER:
        return _build_curriculum(records, recipe, max_steps)
    if stage in {Stage.DECODER, Stage.JOINT}:
        return [
            CurriculumSegment(
                index=0,
                max_base_wer=None,
                records=records,
                optimizer_steps=max_steps or 0,
            )
        ]
    raise ValueError(f"Unsupported A2S-SFT stage: {stage.value}")


def _checkpoint_stage(checkpoint_state: dict[str, Any]) -> Stage | None:
    value = checkpoint_state.get("stage", checkpoint_state.get("stage_name"))
    if not isinstance(value, str):
        return None
    try:
        return Stage.parse(value)
    except ValueError:
        return None


def _can_restore_training_state(
    checkpoint_state: dict[str, Any],
    stage: Stage | str,
    active_groups: list[str],
    train_manifest_sha256: str,
) -> bool:
    expected_stage = Stage.parse(stage)
    source_identity = checkpoint_state.get("manifest_identity", {})
    return (
        _checkpoint_stage(checkpoint_state) is expected_stage
        and sorted(checkpoint_state.get("active_trainable_groups", [])) == sorted(active_groups)
        and source_identity.get("train_manifest_sha256") == train_manifest_sha256
        and source_identity.get("loss_normalization") == LOSS_NORMALIZATION
    )


def prepare_training_run(recipe: StageRecipe, run: StageRun) -> PreparedTrainingRun:
    stage = recipe.stage
    active_groups = list(recipe.trainable_groups)

    import numpy as np
    import torch
    from accelerate import Accelerator

    seed = run.runtime.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    runtime = run.runtime
    gradient_accumulation_steps = runtime.gradient_accumulation
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("A2S-SFT training requires the CUDA GPU server")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("BF16 A2S-SFT training requires an Ampere-or-newer CUDA GPU")

    train_records = load_manifest(
        run.data.train_manifest,
        "train",
        require_audited_audio=True,
        require_base_wer=stage is Stage.ENCODER,
    )
    validation_records = load_manifest(
        run.data.validation_manifest,
        "validation",
        require_audited_audio=True,
    )
    train_ids = {record.id for record in train_records}
    if train_ids.intersection(record.id for record in validation_records):
        raise ValueError("Manifest record leakage between train and validation")
    train_text = {record.text_fingerprint for record in train_records}
    if train_text.intersection(record.text_fingerprint for record in validation_records):
        raise ValueError("Transcript leakage between train and validation")

    max_steps = runtime.max_steps
    segments = _build_stage_segments(train_records, recipe, max_steps)
    output_dir = runtime.output_dir
    initialization = resolve_training_initialization(recipe, run)
    resume_path = initialization.checkpoint
    resume_state = initialization.state
    same_stage_resume = initialization.same_stage_resume
    if same_stage_resume and not _can_restore_training_state(
        resume_state or {},
        stage,
        active_groups,
        file_sha256(run.data.train_manifest),
    ):
        raise ValueError("Resume checkpoint does not match this stage's data and objective contract")
    merge_adapter_paths = list(initialization.merged_adapters)
    model, processor, model_report = load_model_for_lora(
        model_config_for_run(run, recipe),
        initialization.current_adapter,
        trainable_groups=active_groups,
        adapter_groups=active_groups,
        merge_adapter_paths=merge_adapter_paths,
    )
    collator = QwenASRCollator(processor, run.data.prompt, runtime.sample_rate)
    validation_loader, _ = _build_loader(
        validation_records, collator, runtime, seed, shuffle=False
    )
    for segment in segments:
        segment.loader, segment.sampler = _build_loader(
            segment.records, collator, runtime, seed + segment.index, shuffle=True
        )

    if max_steps is None and stage is Stage.ENCODER:
        assert segments[-1].sampler is not None
        updates_per_epoch = math.ceil(
            len(segments[-1].sampler) / gradient_accumulation_steps
        )
        max_steps = recipe.epochs * updates_per_epoch
        if max_steps < len(segments):
            raise ValueError("The configured epoch budget produces fewer steps than curriculum segments")
        fractions = [
            float(value)
            for value in recipe.curriculum_fractions
        ]
        segment_steps = [max(1, int(max_steps * fraction)) for fraction in fractions]
        if sum(segment_steps) > max_steps:
            raise ValueError("Curriculum fractions exceed the derived two-epoch step budget")
        segment_steps[-1] += max_steps - sum(segment_steps)
        for segment, segment_steps_value in zip(segments, segment_steps):
            segment.optimizer_steps = segment_steps_value
    elif max_steps is None and stage in {Stage.DECODER, Stage.JOINT}:
        assert segments[0].sampler is not None
        updates_per_epoch = math.ceil(
            len(segments[0].sampler) / gradient_accumulation_steps
        )
        max_steps = recipe.epochs * updates_per_epoch
        segments[0].optimizer_steps = max_steps

    if max_steps is None:
        raise AssertionError(f"Could not derive an optimizer-step budget for {stage.value}")

    optimizer = _optimizer(model, recipe)
    warmup_steps = round(max_steps * recipe.warmup_ratio)
    scheduler = _learning_rate_schedule(
        optimizer,
        warmup_steps,
        max_steps,
        "cosine",
    )
    model, optimizer, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, validation_loader, scheduler
    )
    for segment in segments:
        segment.loader = accelerator.prepare(segment.loader)
    if accelerator.num_processes != 1:
        raise RuntimeError("A2S-SFT checkpoint/evaluation path is intentionally single-GPU only")

    identity = _stage_identity(run, recipe, segments, model_report)
    identity["validation"] = summarize_dataset(validation_records)
    initialization_checkpoint = str(resume_path) if resume_path else None
    identity["initialization_checkpoint"] = initialization_checkpoint
    identity["recipe"] = recipe.to_dict()
    identity["merged_adapter_lineage"] = [str(path) for path in merge_adapter_paths]
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return PreparedTrainingRun(
        recipe=recipe,
        run=run,
        accelerator=accelerator,
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_records=validation_records,
        validation_loader=validation_loader,
        segments=segments,
        max_steps=max_steps,
        identity=identity,
        initialization_checkpoint=initialization_checkpoint,
        merged_adapter_paths=merge_adapter_paths,
        resume_path=resume_path,
        resume_state=resume_state,
        same_stage_resume=same_stage_resume,
    )


def restore_training_progress(prepared: PreparedTrainingRun) -> TrainingProgress:
    progress = TrainingProgress()
    if prepared.resume_state:
        if prepared.same_stage_resume:
            validate_stage_state(
                prepared.resume_state,
                prepared.recipe.stage.value,
                list(prepared.recipe.trainable_groups),
                accepted_stage_names=(prepared.recipe.stage.phase_name,),
            )
            progress.restored_state = load_checkpoint(
                prepared.accelerator,
                prepared.optimizer,
                prepared.scheduler,
                prepared.resume_path,
            )
        if progress.restored_state is not None:
            state = progress.restored_state
            progress.global_step = int(state["global_step"])
            progress.resume_segment = int(state.get("curriculum_segment", 0))
            progress.resume_segment_epoch = int(state.get("segment_epoch", 0))
            progress.resume_batch = int(state.get("next_batch_in_segment", 0))
            saved_best = state.get("best_metric_value")
            progress.best_metric_value = float(saved_best) if saved_best is not None else None
    return progress


def evaluate_and_record(
    prepared: PreparedTrainingRun,
    global_step: int,
    best_wer: float | None,
) -> tuple[bool, float | None]:
    validation_loss = evaluate_loss(prepared.model, prepared.validation_loader)
    unwrapped = prepared.accelerator.unwrap_model(prepared.model)
    count = min(prepared.run.runtime.evaluation_examples, len(prepared.validation_records))
    rows = (
        generate_transcriptions(
            unwrapped,
            prepared.processor,
            ParquetAudioDataset(prepared.validation_records, prepared.run.runtime.sample_rate),
            list(range(count)),
            prepared.run.data.prompt,
            prepared.run.runtime.max_new_tokens,
        )
        if prepared.accelerator.is_main_process
        else []
    )
    candidate = best_wer if best_wer is not None else math.inf
    if prepared.accelerator.is_main_process:
        metrics = compute_asr_metrics(rows)
        eval_log = {
            "validation/loss": validation_loss,
            "validation/wer": metrics["overall"]["wer"],
            "validation/cer": metrics["overall"]["cer"],
        }
        record_metrics(prepared.run.runtime.output_dir, global_step, "validation", eval_log)
        record_evaluation_rows(prepared.run.runtime.output_dir, global_step, rows)
        candidate = float(eval_log["validation/wer"])
    new_best = candidate < (best_wer if best_wer is not None else math.inf)
    return new_best, candidate if new_best else best_wer


def checkpoint_metadata(
    prepared: PreparedTrainingRun,
    *,
    segment_index: int,
    segment_step: int,
    segment_epoch: int,
    next_batch: int,
    best_wer: float | None,
) -> dict[str, Any]:
    recipe = prepared.recipe
    segment = prepared.segments[segment_index]
    return {
        "recipe_name": "megaasr_a2s_sft",
        "stage": recipe.stage.value,
        "phase_name": recipe.stage.phase_name,
        "active_trainable_groups": list(recipe.trainable_groups),
        "learning_rates": recipe.learning_rate_by_group,
        "curriculum_segment": segment_index,
        "curriculum_max_base_wer": segment.max_base_wer,
        "segment_step": segment_step,
        "segment_epoch": segment_epoch,
        "next_batch_in_segment": next_batch,
        "manifest_identity": prepared.identity,
        "best_metric_name": "validation/wer",
        "best_metric_value": best_wer,
        "initialization_checkpoint": prepared.initialization_checkpoint,
        "merged_adapter_lineage": [str(path) for path in prepared.merged_adapter_paths],
    }


def execute(
    *, recipe: StageRecipe, run: StageRun, allow_training: bool
) -> None:
    run.validate(requires_previous_stage=recipe.requires_previous_stage is not None)
    if not allow_training:
        raise SystemExit(
            "Training guard active. Re-run with --allow-training only after the user approves the optimization run."
        )
    prepared = prepare_training_run(recipe, run)
    progress = restore_training_progress(prepared)
    run_training_loop(prepared, progress)


def run_training_loop(prepared: PreparedTrainingRun, progress: TrainingProgress) -> None:
    import torch

    recipe = prepared.recipe
    run = prepared.run
    stage = recipe.stage
    phase_name = stage.phase_name
    active_groups = list(recipe.trainable_groups)
    accelerator = prepared.accelerator
    model = prepared.model
    processor = prepared.processor
    optimizer = prepared.optimizer
    scheduler = prepared.scheduler
    segments = prepared.segments
    max_steps = prepared.max_steps
    identity = prepared.identity
    initialization_checkpoint = prepared.initialization_checkpoint
    merge_adapter_paths = prepared.merged_adapter_paths
    runtime = run.runtime
    output_dir = runtime.output_dir
    gradient_accumulation_steps = runtime.gradient_accumulation
    global_step = progress.global_step
    resume_segment = progress.resume_segment
    resume_segment_epoch = progress.resume_segment_epoch
    resume_batch = progress.resume_batch
    best_metric_value = progress.best_metric_value
    state = progress.restored_state

    eval_every = runtime.evaluate_every
    log_every = runtime.log_every
    save_every = runtime.save_every
    contract_steps = 1
    optimizer.zero_grad(set_to_none=True)
    window_started = time.perf_counter()
    accumulation_supervised_tokens = 0
    window_token_loss_sum = 0.0
    window_microbatch_loss_sum = 0.0
    window_microbatches = 0
    window_audio_seconds = 0.0
    window_tokens = 0

    for segment_index, segment in enumerate(segments):
        if segment_index < resume_segment:
            continue
        assert segment.loader is not None and segment.sampler is not None
        segment_step = 0
        if segment_index == resume_segment and state is not None:
            segment_step = int(state.get("segment_step", 0))
        segment_epoch = resume_segment_epoch if segment_index == resume_segment else 0
        batch_to_skip = resume_batch if segment_index == resume_segment else 0
        while segment_step < segment.optimizer_steps and global_step < max_steps:
            segment.sampler.set_epoch(segment_epoch)
            for batch_in_segment, packed in enumerate(segment.loader):
                if batch_in_segment < batch_to_skip:
                    continue
                batch, metadata = extract_model_batch(dict(packed))
                if global_step < contract_steps:
                    validate_batch_contract(accelerator.unwrap_model(model), batch)
                microbatch_tokens = int((batch["labels"] != -100).sum().item())
                if microbatch_tokens <= 0:
                    raise ValueError("Training microbatch contains no supervised tokens")
                accumulation_supervised_tokens += microbatch_tokens
                with accelerator.accumulate(model):
                    output = model(**batch)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite {stage.value} loss at optimizer step {global_step}"
                        )
                    accelerator.backward(loss * microbatch_tokens)
                    if accelerator.sync_gradients:
                        gradient_normalization_scale = normalize_accumulated_gradients(
                            model,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            supervised_tokens=accumulation_supervised_tokens,
                        )
                        grad_metrics = _group_gradient_metrics(model, active_groups)
                        accelerator.clip_grad_norm_(model.parameters(), recipe.max_gradient_norm)
                    else:
                        grad_metrics = {}
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                microbatch_loss_value = float(loss.detach().float().item())
                window_token_loss_sum += microbatch_loss_value * microbatch_tokens
                window_microbatch_loss_sum += microbatch_loss_value
                window_microbatches += 1
                window_audio_seconds += sum(float(row["duration"]) for row in metadata)
                window_tokens += microbatch_tokens
                if not accelerator.sync_gradients:
                    continue
                optimizer_supervised_tokens = accumulation_supervised_tokens
                accumulation_supervised_tokens = 0
                global_step += 1
                segment_step += 1
                if window_tokens <= 0:
                    raise RuntimeError("Completed optimizer step has no supervised tokens")
                if accelerator.is_main_process and (global_step == 1 or global_step % log_every == 0):
                    elapsed = max(time.perf_counter() - window_started, 1e-6)
                    log = {
                        "train/loss": window_token_loss_sum / window_tokens,
                        "train/microbatch_mean_loss": window_microbatch_loss_sum
                        / max(window_microbatches, 1),
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "train/audio_seconds_per_second": window_audio_seconds / elapsed,
                        "train/supervised_tokens": window_tokens,
                        "train/optimizer_supervised_tokens": optimizer_supervised_tokens,
                        "train/gradient_normalization_scale": gradient_normalization_scale,
                        "train/curriculum_segment": float(segment_index),
                        **grad_metrics,
                        **cuda_memory_metrics(),
                        **gpu_activity_metrics(),
                    }
                    record_metrics(output_dir, global_step, phase_name, log)
                    print(json.dumps({"step": global_step, **log}, sort_keys=True), flush=True)
                    window_started = time.perf_counter()
                    window_token_loss_sum = 0.0
                    window_microbatch_loss_sum = 0.0
                    window_microbatches = 0
                    window_audio_seconds = 0.0
                    window_tokens = 0

                new_best = False
                if global_step % eval_every == 0 or global_step == max_steps:
                    new_best, best_metric_value = evaluate_and_record(
                        prepared,
                        global_step,
                        best_metric_value,
                    )

                metadata = checkpoint_metadata(
                    prepared,
                    segment_index=segment_index,
                    segment_step=segment_step,
                    segment_epoch=segment_epoch,
                    next_batch=batch_in_segment + 1,
                    best_wer=best_metric_value,
                )
                if new_best:
                    save_checkpoint(
                        accelerator,
                        model,
                        processor,
                        optimizer,
                        scheduler,
                        output_dir,
                        global_step,
                        segment_epoch,
                        metadata,
                        3,
                        name="best",
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
                        segment_epoch,
                        metadata,
                        3,
                    )
                if global_step >= max_steps or segment_step >= segment.optimizer_steps:
                    break
            segment_epoch += 1
            batch_to_skip = 0
        resume_segment_epoch = 0
        resume_batch = 0

    if global_step > 0:
        save_checkpoint(
            accelerator,
            model,
            processor,
            optimizer,
            scheduler,
            output_dir,
            global_step,
            segments[-1].optimizer_steps,
            {
                "recipe_name": "megaasr_a2s_sft",
                "stage": stage.value,
                "phase_name": phase_name,
                "active_trainable_groups": active_groups,
                "learning_rates": {
                    group: recipe.learning_rate_by_group[group] for group in active_groups
                },
                "curriculum_segment": len(segments) - 1,
                "curriculum_max_base_wer": segments[-1].max_base_wer,
                "curriculum_complete": stage is Stage.ENCODER,
                "stage_complete": True,
                "initialization_checkpoint": initialization_checkpoint,
                "merged_adapter_lineage": [str(path) for path in merge_adapter_paths],
                "manifest_identity": identity,
                "best_metric_name": "validation/wer",
                "best_metric_value": best_metric_value,
            },
            3,
        )
