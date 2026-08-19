from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from tamil_asr.data.text import normalize_transcript
from tamil_asr.evaluation.slices import evaluation_slices


def _cer(reference: str, hypothesis: str) -> tuple[int, int]:
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError("jiwer is required for ASR metrics") from exc
    output = jiwer.process_characters(reference, hypothesis)
    errors = output.substitutions + output.deletions + output.insertions
    return int(errors), int(output.hits + output.substitutions + output.deletions)


def _wer(reference: str, hypothesis: str) -> tuple[int, int]:
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError("jiwer is required for ASR metrics") from exc
    output = jiwer.process_words(reference, hypothesis)
    errors = output.substitutions + output.deletions + output.insertions
    return int(errors), int(output.hits + output.substitutions + output.deletions)


def compute_asr_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty evaluation set")
    totals: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    output_flags: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        reference = normalize_transcript(row["reference"])
        hypothesis = normalize_transcript(row["hypothesis"])
        if not reference:
            raise ValueError(f"{row.get('id')}: empty normalized reference")
        word_errors, reference_words = _wer(reference, hypothesis)
        char_errors, reference_chars = _cer(reference, hypothesis)
        groups = [("overall", "overall"), *evaluation_slices(row).items()]
        hypothesis_length = len(hypothesis)
        reference_length = len(reference)
        for key in groups:
            group = totals[key]
            group[0] += word_errors
            group[1] += reference_words
            group[2] += char_errors
            group[3] += reference_chars
            group[4] += 1
            flags = output_flags[key]
            flags[0] += not hypothesis
            flags[1] += bool(row.get("hit_max_new_tokens", False))
            flags[2] += hypothesis_length > max(2 * reference_length, reference_length + 40)

    def summarize(key: tuple[str, str]) -> dict[str, Any]:
        values = totals[key]
        flags = output_flags[key]
        return {
            "wer": values[0] / max(values[1], 1),
            "cer": values[2] / max(values[3], 1),
            "word_errors": values[0],
            "reference_words": values[1],
            "char_errors": values[2],
            "reference_characters": values[3],
            "examples": values[4],
            "empty_hypotheses": flags[0],
            "hit_max_new_tokens": flags[1],
            "excessive_length_hypotheses": flags[2],
        }

    dimensions: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for key in totals:
        dimension, value = key
        if dimension != "overall":
            dimensions[dimension][value] = summarize(key)
    return {
        "overall": summarize(("overall", "overall")),
        "slices": {dimension: dict(values) for dimension, values in sorted(dimensions.items())},
    }
