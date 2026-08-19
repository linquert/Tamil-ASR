from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tamil_asr.model.qwen3_asr import load_model_for_lora
from tamil_asr.training.checkpoint import latest_checkpoint, read_checkpoint_state
from tamil_asr.training.staged_asr.recipe import StageRecipe, recipe_for_stage
from tamil_asr.training.staged_asr.model import apply_recipe_to_model_config
from tamil_asr.training.staged_asr.run import StageRun


@dataclass(frozen=True)
class CheckpointLineage:
    checkpoint: Path
    recipe: StageRecipe
    merged_adapters: tuple[Path, ...]
    current_adapter: Path
    stage_complete: bool

    @classmethod
    def read(cls, checkpoint: str | Path) -> "CheckpointLineage":
        checkpoint_path = Path(checkpoint)
        state = read_checkpoint_state(checkpoint_path)
        stage = state.get("stage")
        if not isinstance(stage, str):
            raise ValueError(f"Staged ASR checkpoint is missing its typed stage identity: {checkpoint_path}")
        merged = tuple(Path(path) for path in state.get("merged_adapter_lineage", ()))
        current = checkpoint_path / "adapter"
        missing = [path for path in (*merged, current) if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Staged ASR checkpoint lineage is incomplete: {missing}")
        return cls(
            checkpoint=checkpoint_path,
            recipe=recipe_for_stage(stage),
            merged_adapters=merged,
            current_adapter=current,
            stage_complete=bool(state.get("stage_complete", False)),
        )

    @property
    def all_adapters(self) -> tuple[Path, ...]:
        return (*self.merged_adapters, self.current_adapter)


@dataclass(frozen=True)
class TrainingInitialization:
    checkpoint: Path | None
    merged_adapters: tuple[Path, ...]
    current_adapter: Path | None
    state: dict[str, Any] | None
    same_stage_resume: bool


def resolve_training_initialization(
    recipe: StageRecipe, run: StageRun
) -> TrainingInitialization:
    if run.resume is not None:
        checkpoint = (
            latest_checkpoint(run.runtime.output_dir)
            if run.resume == "latest"
            else Path(run.resume)
        )
        if checkpoint is None:
            raise FileNotFoundError(f"No complete checkpoint found in {run.runtime.output_dir}")
        lineage = CheckpointLineage.read(checkpoint)
        if lineage.recipe.stage is not recipe.stage:
            raise ValueError(
                f"Cannot resume {recipe.stage.value} from a {lineage.recipe.stage.value} checkpoint"
            )
        return TrainingInitialization(
            checkpoint=checkpoint,
            merged_adapters=lineage.merged_adapters,
            current_adapter=lineage.current_adapter,
            state=read_checkpoint_state(checkpoint),
            same_stage_resume=True,
        )

    if run.previous_stage is not None:
        lineage = CheckpointLineage.read(run.previous_stage)
        expected = recipe.requires_previous_stage
        if lineage.recipe.stage is not expected:
            raise ValueError(
                f"{recipe.stage.value} requires a completed {expected.value} checkpoint; "
                f"found {lineage.recipe.stage.value}"
            )
        if not lineage.stage_complete:
            raise ValueError(
                f"{recipe.stage.value} requires a completed {expected.value} checkpoint, not an intermediate one"
            )
        return TrainingInitialization(
            checkpoint=lineage.checkpoint,
            merged_adapters=lineage.all_adapters,
            current_adapter=None,
            state=read_checkpoint_state(lineage.checkpoint),
            same_stage_resume=False,
        )

    return TrainingInitialization(None, (), None, None, False)


def is_staged_asr_checkpoint(checkpoint: str | Path) -> bool:
    path = Path(checkpoint) / "state.json"
    if not path.is_file():
        return False
    try:
        state = read_checkpoint_state(Path(checkpoint))
    except (FileNotFoundError, ValueError):
        return False
    return isinstance(state.get("stage"), str)


def load_checkpoint_for_inference(
    model_config: Mapping[str, Any],
    checkpoint: str | Path,
) -> tuple[Any, Any, dict[str, Any]]:
    lineage = CheckpointLineage.read(checkpoint)
    model, processor, report = load_model_for_lora(
        apply_recipe_to_model_config(model_config, lineage.recipe),
        lineage.current_adapter,
        trainable_groups=lineage.recipe.trainable_groups,
        adapter_groups=lineage.recipe.trainable_groups,
        merge_adapter_paths=lineage.merged_adapters,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    report["checkpoint"] = str(lineage.checkpoint)
    report["checkpoint_stage"] = lineage.recipe.stage.value
    report["stage_complete"] = lineage.stage_complete
    report["effective_adapter_lineage"] = [str(path) for path in lineage.all_adapters]
    return model, processor, report


def load_asr_checkpoint(
    model_config: Mapping[str, Any], checkpoint: str | Path
) -> tuple[Any, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    if is_staged_asr_checkpoint(checkpoint_path):
        return load_checkpoint_for_inference(model_config, checkpoint_path)
    return load_model_for_lora(model_config, checkpoint_path / "adapter")
