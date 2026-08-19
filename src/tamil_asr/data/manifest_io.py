from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from tamil_asr.data.manifest import ManifestRecord, read_jsonl


def load_manifest(
    path: str | Path,
    expected_split: str,
    *,
    require_audited_audio: bool = True,
    require_base_wer: bool = False,
) -> list[ManifestRecord]:
    records = [ManifestRecord.from_dict(row) for row in read_jsonl(path)]
    if not records:
        raise ValueError(f"{path}: empty manifest")
    for record in records:
        if record.split != expected_split:
            raise ValueError(f"{record.id}: expected split={expected_split}, found {record.split}")
        if require_audited_audio and (
            not record.audio_sha1 or record.sample_rate <= 0 or not record.audio_format
        ):
            raise ValueError(f"{record.id}: unaudited audio metadata; run prepare with --audio-audit all")
        if not record.text or not record.text_column:
            raise ValueError(f"{record.id}: missing selected transcript or transcript provenance")
        if require_base_wer and (
            record.base_wer is None
            or not math.isfinite(float(record.base_wer))
            or not 0.0 <= float(record.base_wer) <= 1.0
        ):
            raise ValueError(f"{record.id}: acoustic curriculum requires base_wer in [0, 1]")
        if not Path(record.parquet_path).is_file():
            raise FileNotFoundError(f"{record.id}: source Parquet missing: {record.parquet_path}")
    return records


def summarize_dataset(records: list[ManifestRecord]) -> dict[str, Any]:
    counts = Counter(record.dataset for record in records)
    formats = Counter(record.audio_format for record in records)
    rates = Counter(str(record.sample_rate) for record in records)
    return {
        "examples": len(records),
        "hours": sum(record.duration for record in records) / 3600.0,
        "datasets": dict(counts),
        "audio_formats": dict(formats),
        "sample_rates": dict(rates),
    }
