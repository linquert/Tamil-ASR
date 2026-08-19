from __future__ import annotations

import json

from tamil_asr.data.collator import QwenASRCollator, validate_batch_contract
from tamil_asr.data.manifest import filter_records_by_base_wer
from tamil_asr.data.manifest_io import load_manifest
from tamil_asr.data.parquet_dataset import ParquetAudioDataset
from tamil_asr.model.qwen3_asr import load_model_for_lora
from tamil_asr.training.diagnostics import cuda_memory_metrics
from tamil_asr.training.batch import extract_model_batch
from tamil_asr.training.staged_asr.checkpoint import resolve_training_initialization
from tamil_asr.training.staged_asr.executor import _group_gradient_metrics
from tamil_asr.training.staged_asr.model import model_config_for_run
from tamil_asr.training.staged_asr.recipe import StageRecipe
from tamil_asr.training.staged_asr.run import StageRun


def preflight_stage(recipe: StageRecipe, run: StageRun) -> dict[str, object]:
    run.validate(requires_previous_stage=recipe.requires_previous_stage is not None)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Staged ASR preflight must run on the target CUDA server")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("BF16 staged ASR preflight requires an Ampere-or-newer GPU")

    records = load_manifest(
        run.data.train_manifest,
        "train",
        require_audited_audio=True,
        require_base_wer=bool(recipe.curriculum_thresholds),
    )
    if recipe.curriculum_thresholds:
        records = filter_records_by_base_wer(records, recipe.curriculum_thresholds[0])
    if not records:
        raise ValueError("The first recipe segment contains no training records")

    initialization = resolve_training_initialization(recipe, run)
    model, processor, model_report = load_model_for_lora(
        model_config_for_run(run, recipe),
        initialization.current_adapter,
        trainable_groups=recipe.trainable_groups,
        adapter_groups=recipe.trainable_groups,
        merge_adapter_paths=initialization.merged_adapters,
    )
    model.to("cuda")
    collator = QwenASRCollator(processor, run.data.prompt, run.runtime.sample_rate)
    dataset = ParquetAudioDataset(records, run.runtime.sample_rate)
    packed = collator([dataset[0]])
    batch, _ = extract_model_batch(packed)
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    contract = validate_batch_contract(model, batch)

    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Staged ASR preflight produced a non-finite loss")
    output.loss.backward()
    gradients = _group_gradient_metrics(model, list(recipe.trainable_groups))
    result: dict[str, object] = {
        "status": "ok_no_optimizer_step",
        "stage": recipe.stage.value,
        "loss": float(output.loss.detach().float().item()),
        "contract": contract,
        "gradients": gradients,
        "model": model_report,
        "memory": cuda_memory_metrics(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result
