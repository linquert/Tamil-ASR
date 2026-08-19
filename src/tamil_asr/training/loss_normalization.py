from __future__ import annotations

from typing import Any




LOSS_NORMALIZATION = "supervised_token_mean_v1"


def normalize_accumulated_gradients(
    model: Any,
    *,
    gradient_accumulation_steps: int,
    supervised_tokens: int,
) -> float:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if supervised_tokens <= 0:
        raise ValueError("Accumulation window must contain supervised tokens")

    scale = float(gradient_accumulation_steps) / float(supervised_tokens)
    found_gradient = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter.grad.mul_(scale)
        found_gradient = True
    if not found_gradient:
        raise RuntimeError("No gradients found while normalizing the accumulation window")
    return scale
