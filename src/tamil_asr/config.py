from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return config


def require_keys(mapping: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = keys.difference(mapping)
    if missing:
        raise ValueError(f"{context}: missing required configuration keys: {sorted(missing)}")


def validate_training_config(config: Mapping[str, Any]) -> None:
    require_keys(config, {"seed", "model", "data", "training", "evaluation", "wandb"}, "root")
    require_keys(config["model"], {"path", "dtype", "lora"}, "model")
    require_keys(
        config["data"],
        {
            "train_manifest",
            "validation_manifest",
            "sample_rate",
            "max_batch_audio_seconds",
            "max_batch_size",
            "prompt",
        },
        "data",
    )
    require_keys(
        config["training"],
        {
            "output_dir",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
            "max_grad_norm",
        },
        "training",
    )
    if "max_steps" not in config["training"]:
        raise ValueError("training.max_steps is required")
    if config["model"]["dtype"] != "bfloat16":
        raise ValueError("The validated A4000 baseline intentionally supports dtype=bfloat16 only")
    if "max_steps" in config["training"] and int(config["training"]["max_steps"]) <= 0:
        raise ValueError("training.max_steps must be positive")
    if float(config["data"]["max_batch_audio_seconds"]) < float(config["data"].get("max_duration", 0)):
        raise ValueError("max_batch_audio_seconds must be >= max_duration so every sample can form a batch")
    if int(config["data"].get("num_workers", 2)) < 0:
        raise ValueError("data.num_workers must be non-negative")
    if int(config["data"].get("prefetch_factor", 2)) <= 0:
        raise ValueError("data.prefetch_factor must be positive")
    evaluation = config["evaluation"]
    if int(evaluation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("evaluation.max_new_tokens must be positive")
    if int(evaluation.get("num_examples", 0)) <= 0:
        raise ValueError("evaluation.num_examples must be positive")
    if int(evaluation.get("generation_batch_size", 1)) <= 0:
        raise ValueError("evaluation.generation_batch_size must be positive")
    schedule = config["training"].get("lr_schedule", "cosine")
    if schedule not in {"cosine", "constant"}:
        raise ValueError("training.lr_schedule must be 'cosine' or 'constant'")
    best_mode = config["training"].get("save_best_mode", "min")
    if best_mode not in {"min", "max"}:
        raise ValueError("training.save_best_mode must be 'min' or 'max'")
