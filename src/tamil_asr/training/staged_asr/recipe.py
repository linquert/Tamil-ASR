from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from tamil_asr.model.qwen3_asr import AUDIO_GROUP, LLM_GROUP


class Stage(str, Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    JOINT = "joint"

    @property
    def order(self) -> int:
        return _STAGE_SEQUENCE.index(self)

    @property
    def phase_name(self) -> str:
        return {
            Stage.ENCODER: "acoustic_curriculum",
            Stage.DECODER: "semantic_adaptation",
            Stage.JOINT: "joint_alignment",
        }[self]

    @property
    def previous(self) -> Stage | None:
        return None if self is Stage.ENCODER else _STAGE_SEQUENCE[self.order - 1]

    @classmethod
    def parse(cls, value: Stage | str) -> Stage:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            for stage in cls:
                if value == stage.phase_name:
                    return stage
            raise ValueError(f"Unknown staged ASR stage: {value!r}") from None


_STAGE_SEQUENCE = (Stage.ENCODER, Stage.DECODER, Stage.JOINT)


@dataclass(frozen=True)
class StageRecipe:

    stage: Stage
    trainable_groups: tuple[str, ...]
    learning_rates: tuple[tuple[str, float], ...]
    epochs: int
    warmup_ratio: float
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    curriculum_thresholds: tuple[float, ...] = ()
    curriculum_fractions: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("Stage epochs must be positive")
        unsupported_groups = set(self.trainable_groups).difference({AUDIO_GROUP, LLM_GROUP})
        if not self.trainable_groups or unsupported_groups:
            raise ValueError(f"Invalid trainable groups: {self.trainable_groups}")
        rates = self.learning_rate_by_group
        if set(rates) != set(self.trainable_groups) or any(rate <= 0 for rate in rates.values()):
            raise ValueError("Every trainable group requires exactly one positive learning rate")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("Warmup ratio must be in [0, 1)")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.weight_decay < 0.0 or self.max_gradient_norm <= 0.0:
            raise ValueError("Weight decay must be non-negative and gradient norm must be positive")
        if bool(self.curriculum_thresholds) != bool(self.curriculum_fractions):
            raise ValueError("Curriculum thresholds and fractions must be provided together")
        if self.curriculum_thresholds:
            if tuple(sorted(set(self.curriculum_thresholds))) != self.curriculum_thresholds:
                raise ValueError("Curriculum thresholds must be strictly increasing")
            if len(self.curriculum_thresholds) != len(self.curriculum_fractions):
                raise ValueError("Curriculum thresholds and fractions must have equal length")
            if any(not 0.0 < threshold <= 1.0 for threshold in self.curriculum_thresholds):
                raise ValueError("Curriculum thresholds must be in (0, 1]")
            if any(fraction <= 0.0 for fraction in self.curriculum_fractions):
                raise ValueError("Curriculum fractions must be positive")
            if abs(sum(self.curriculum_fractions) - 1.0) > 1e-9:
                raise ValueError("Curriculum fractions must sum to one")

    @property
    def learning_rate_by_group(self) -> dict[str, float]:
        return dict(self.learning_rates)

    @property
    def requires_previous_stage(self) -> Stage | None:
        return self.stage.previous

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["stage"] = self.stage.value
        value["learning_rates"] = self.learning_rate_by_group
        return value


ENCODER_RECIPE = StageRecipe(
    stage=Stage.ENCODER,
    trainable_groups=(AUDIO_GROUP,),
    learning_rates=((AUDIO_GROUP, 1e-6),),
    epochs=2,
    warmup_ratio=0.05,
    curriculum_thresholds=(0.30, 0.50, 0.70),
    curriculum_fractions=(0.25, 0.25, 0.50),
)

DECODER_RECIPE = StageRecipe(
    stage=Stage.DECODER,
    trainable_groups=(LLM_GROUP,),
    learning_rates=((LLM_GROUP, 1e-6),),
    epochs=1,
    warmup_ratio=0.05,
)

JOINT_RECIPE = StageRecipe(
    stage=Stage.JOINT,
    trainable_groups=(AUDIO_GROUP, LLM_GROUP),
    learning_rates=((AUDIO_GROUP, 5e-7), (LLM_GROUP, 1e-6)),
    epochs=1,
    warmup_ratio=0.03,
)

_RECIPES = {recipe.stage: recipe for recipe in (ENCODER_RECIPE, DECODER_RECIPE, JOINT_RECIPE)}


def recipe_for_stage(stage: Stage | str) -> StageRecipe:
    return _RECIPES[Stage.parse(stage)]


def validate_stage_transition(recipe: StageRecipe, source_stage: Stage | str | None) -> None:
    expected = recipe.requires_previous_stage
    if expected is None:
        if source_stage is not None:
            raise ValueError("Encoder stage must start from the configured base model")
        return
    actual = Stage.parse(source_stage) if source_stage is not None else None
    if actual is not expected:
        raise ValueError(
            f"{recipe.stage.value} stage requires a completed {expected.value} checkpoint; "
            f"found {source_stage!r}"
        )
