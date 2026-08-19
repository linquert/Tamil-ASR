from __future__ import annotations

from typing import Any


def extract_model_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = batch.pop("_metadata")
    batch.pop("_prefix_lengths", None)
    batch.pop("_full_lengths", None)
    return batch, metadata
