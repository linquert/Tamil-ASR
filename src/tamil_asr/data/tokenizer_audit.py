from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from tamil_asr.data.manifest import ManifestRecord


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def audit_tokenizer(
    records: Sequence[ManifestRecord],
    tokenizer: Any,
    *,
    worst_examples: int = 20,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Tokenizer audit requires at least one manifest record")
    if worst_examples <= 0:
        raise ValueError("worst_examples must be positive")

    unknown_id = getattr(tokenizer, "unk_token_id", None)
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    total_words = 0
    total_codepoints = 0
    total_tamil_codepoints = 0
    unknown_tokens = 0
    for record in records:
        token_ids = list(tokenizer(record.text, add_special_tokens=False)["input_ids"])
        words = max(len(record.text.split()), 1)
        codepoints = max(len(record.text), 1)
        tamil_codepoints = sum(0x0B80 <= ord(character) <= 0x0BFF for character in record.text)
        row = {
            "id": record.id,
            "dataset": record.dataset,
            "text": record.text,
            "tokens": len(token_ids),
            "words": words,
            "codepoints": codepoints,
            "tamil_codepoints": tamil_codepoints,
            "tokens_per_word": len(token_ids) / words,
            "tokens_per_codepoint": len(token_ids) / codepoints,
        }
        rows.append(row)
        total_tokens += len(token_ids)
        total_words += words
        total_codepoints += codepoints
        total_tamil_codepoints += tamil_codepoints
        if unknown_id is not None:
            unknown_tokens += sum(int(token_id) == int(unknown_id) for token_id in token_ids)

    tokens_per_word = [float(row["tokens_per_word"]) for row in rows]
    tokens_per_codepoint = [float(row["tokens_per_codepoint"]) for row in rows]
    worst = sorted(
        rows,
        key=lambda row: (float(row["tokens_per_word"]), int(row["tokens"]), str(row["id"])),
        reverse=True,
    )[:worst_examples]
    return {
        "examples": len(rows),
        "tokens": total_tokens,
        "words": total_words,
        "codepoints": total_codepoints,
        "tamil_codepoints": total_tamil_codepoints,
        "unknown_tokens": unknown_tokens,
        "corpus_tokens_per_word": total_tokens / max(total_words, 1),
        "corpus_tokens_per_codepoint": total_tokens / max(total_codepoints, 1),
        "tokens_per_word_p50": _percentile(tokens_per_word, 0.50),
        "tokens_per_word_p95": _percentile(tokens_per_word, 0.95),
        "tokens_per_codepoint_p95": _percentile(tokens_per_codepoint, 0.95),
        "worst_fragmented_examples": worst,
    }
