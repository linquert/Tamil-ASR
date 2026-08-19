from __future__ import annotations

from typing import Any, Mapping

from tamil_asr.training.staged_asr.recipe import StageRecipe
from tamil_asr.training.staged_asr.run import StageRun


def apply_recipe_to_model_config(
    base: Mapping[str, Any], recipe: StageRecipe
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "lora": {
                "rank": recipe.lora_rank,
                "alpha": recipe.lora_alpha,
                "dropout": recipe.lora_dropout,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            },
            "audio_lora": {
                "enabled": True,
                "layer_selection": "all",
                "target_modules": [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj", "fc1", "fc2",
                ],
                "projection_modules": ["conv_out", "proj1", "proj2"],
            },
        }
    )
    return config


def model_config_for_run(run: StageRun, recipe: StageRecipe) -> dict[str, Any]:
    return apply_recipe_to_model_config(
        {
            "path": str(run.model.base_model),
            "qwen_asr_repo": str(run.model.qwen_repository),
            "local_files_only": True,
        },
        recipe,
    )
