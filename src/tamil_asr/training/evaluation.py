from __future__ import annotations

from typing import Any

from tamil_asr.training.batch import extract_model_batch


def evaluate_loss(model: Any, loader: Any, max_batches: int = 32) -> float:
    import torch

    model.eval()
    weighted_loss = 0.0
    tokens = 0
    with torch.inference_mode():
        for index, packed in enumerate(loader):
            if index >= max_batches:
                break
            batch, _ = extract_model_batch(dict(packed))
            supervised = int((batch["labels"] != -100).sum().item())
            output = model(**batch)
            weighted_loss += float(output.loss.float().item()) * supervised
            tokens += supervised
    model.train()
    if tokens == 0:
        raise ValueError("Validation loader produced no supervised tokens")
    return weighted_loss / tokens
