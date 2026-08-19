from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["DataPaths", "ModelPaths", "RuntimeOptions", "StageRun"]


@dataclass(frozen=True)
class ModelPaths:
    base_model: Path
    qwen_repository: Path


@dataclass(frozen=True)
class DataPaths:
    train_manifest: Path
    validation_manifest: Path
    prompt: str


@dataclass(frozen=True)
class RuntimeOptions:
    output_dir: Path
    seed: int = 42
    sample_rate: int = 16_000
    max_batch_audio_seconds: float = 20.0
    max_batch_size: int = 8
    gradient_accumulation: int = 8
    num_workers: int = 2
    prefetch_factor: int = 2
    log_every: int = 10
    evaluate_every: int = 200
    save_every: int = 200
    evaluation_examples: int = 32
    max_new_tokens: int = 192
    max_steps: int | None = None


@dataclass(frozen=True)
class StageRun:
    model: ModelPaths
    data: DataPaths
    runtime: RuntimeOptions
    previous_stage: Path | None = None
    resume: Path | str | None = None

    def validate(self, *, requires_previous_stage: bool) -> None:
        if self.previous_stage is not None and self.resume is not None:
            raise ValueError("Choose either previous-stage initialization or same-stage resume")
        if requires_previous_stage and self.previous_stage is None and self.resume is None:
            raise ValueError("This stage requires the preceding stage checkpoint or a same-stage resume")
        if not requires_previous_stage and self.previous_stage is not None:
            raise ValueError("Encoder training starts from the base model")
        if not self.model.base_model.is_dir():
            raise FileNotFoundError(f"Base model directory does not exist: {self.model.base_model}")
        if not self.model.qwen_repository.is_dir():
            raise FileNotFoundError(
                f"Qwen repository directory does not exist: {self.model.qwen_repository}"
            )
        if not self.data.train_manifest.is_file():
            raise FileNotFoundError(f"Training manifest does not exist: {self.data.train_manifest}")
        if not self.data.validation_manifest.is_file():
            raise FileNotFoundError(
                f"Validation manifest does not exist: {self.data.validation_manifest}"
            )
        if self.previous_stage is not None and not self.previous_stage.is_dir():
            raise FileNotFoundError(f"Previous-stage checkpoint does not exist: {self.previous_stage}")
        if isinstance(self.resume, Path) and not self.resume.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume}")
        if self.runtime.gradient_accumulation <= 0:
            raise ValueError("Gradient accumulation must be positive")
        if self.runtime.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.runtime.max_batch_size <= 0:
            raise ValueError("Maximum batch size must be positive")
        if self.runtime.max_batch_audio_seconds <= 0:
            raise ValueError("Batch audio limit must be positive")
        if self.runtime.log_every <= 0 or self.runtime.evaluate_every <= 0:
            raise ValueError("Logging and evaluation intervals must be positive")
        if self.runtime.save_every < 0:
            raise ValueError("Checkpoint interval must be non-negative")
        if self.runtime.evaluation_examples <= 0 or self.runtime.max_new_tokens <= 0:
            raise ValueError("Evaluation example and generation limits must be positive")
        if self.runtime.max_steps is not None and self.runtime.max_steps <= 0:
            raise ValueError("Explicit maximum steps must be positive")
