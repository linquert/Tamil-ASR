from __future__ import annotations

import argparse
import json
from pathlib import Path

from tamil_asr.config import load_yaml, validate_training_config
from tamil_asr.data.collator import QwenASRCollator, validate_batch_contract
from tamil_asr.data.manifest_io import load_manifest
from tamil_asr.data.parquet_dataset import ParquetAudioDataset
from tamil_asr.model.qwen3_asr import LLM_GROUP, load_model_for_lora
from tamil_asr.training.diagnostics import cuda_memory_metrics, gradient_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="One-batch Qwen3-ASR contract test; no optimizer step")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_training_config(config)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Preflight must run on the target CUDA server")
    data_cfg = config["data"]
    records = load_manifest(
        data_cfg["train_manifest"],
        "train",
        require_audited_audio=bool(data_cfg.get("require_audited_audio", True)),
    )
    dataset = ParquetAudioDataset(records, int(data_cfg["sample_rate"]))
    model_cfg = config["model"]
    if bool(model_cfg.get("audio_lora", {}).get("enabled", False)):
        raise ValueError("This preflight is for the decoder-only ASR trainer; audio_lora must be disabled")
    adapter_init = model_cfg.get("adapter_init")
    if adapter_init and not Path(adapter_init).is_dir():
        raise FileNotFoundError(f"Configured model.adapter_init does not exist: {adapter_init}")
    frozen_adapter_path = model_cfg.get("frozen_adapter_path")
    if frozen_adapter_path and not Path(frozen_adapter_path).is_dir():
        raise FileNotFoundError(
            f"Configured model.frozen_adapter_path does not exist: {frozen_adapter_path}"
        )
    model, processor, model_report = load_model_for_lora(
        model_cfg,
        adapter_init,
        trainable_groups=[LLM_GROUP],
    )
    model.cuda()
    collator = QwenASRCollator(processor, data_cfg["prompt"], int(data_cfg["sample_rate"]))
    batch = collator([dataset[0]])
    batch.pop("_metadata")
    batch.pop("_prefix_lengths")
    batch.pop("_full_lengths")
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    contract = validate_batch_contract(model, batch)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Preflight forward produced non-finite loss")
    output.loss.backward()
    gradients = gradient_diagnostics(model)
    if gradients["grad/nonfinite_parameters"] or gradients["grad/nonzero_parameters"] <= 0:
        raise RuntimeError(f"Invalid first-batch LoRA gradients: {gradients}")
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "audio_tower" in name
    ):
        raise AssertionError("Frozen audio tower unexpectedly received stored parameter gradients")
    if "asr_residual" in (model_report.get("trainable_adapter_names") or []):
        frozen_with_grad = [
            name
            for name, parameter in model.named_parameters()
            if ".frozen_base." in name and parameter.grad is not None
        ]
        residual_trainable = [
            name
            for name, parameter in model.named_parameters()
            if ".asr_residual." in name and parameter.requires_grad
        ]
        if frozen_with_grad:
            raise AssertionError(
                f"Frozen base adapter unexpectedly received gradients: {frozen_with_grad[:3]}"
            )
        if not residual_trainable:
            raise AssertionError("Residual adapter has no trainable parameters")
    print(
        json.dumps(
            {
                "status": "ok_no_optimizer_step",
                "loss": float(output.loss.detach().float().item()),
                "contract": contract,
                "model": model_report,
                "gradients": gradients,
                "memory": cuda_memory_metrics(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
