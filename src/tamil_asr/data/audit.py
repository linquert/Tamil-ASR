from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Sequence

from .audio import audit_audio_bytes, extract_audio_bytes, validate_duration
from .manifest import ManifestRecord, audit_manifest_leakage
from .parquet_dataset import _RowGroupCache
from .text import tamil_fraction
from .text import stable_bucket


_LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.+-]*\b")
_ANNOTATION_RE = re.compile(r"\[[^\]\n]{1,80}\]|<[^<>\n]{1,80}>")


def text_profile(records: Sequence[ManifestRecord]) -> dict[str, Any]:
    datasets: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stats = datasets[record.dataset]
        stats["examples"] += 1
        stats["characters"] += len(record.text)
        stats["latin_token_examples"] += bool(_LATIN_TOKEN_RE.search(record.text))
        stats["annotation_examples"] += bool(_ANNOTATION_RE.search(record.text))
        fraction = tamil_fraction(record.text)
        if fraction >= 0.95:
            stats["tamil_fraction_ge_0.95"] += 1
        elif fraction >= 0.50:
            stats["tamil_fraction_0.50_0.95"] += 1
        else:
            stats["tamil_fraction_below_0.50"] += 1
        if len(examples[record.dataset]) < 8:
            examples[record.dataset].append(
                {
                    "id": record.id,
                    "text_column": record.text_column,
                    "duration": record.duration,
                    "text": record.text,
                    "latin_tokens": _LATIN_TOKEN_RE.findall(record.text),
                }
            )
    return {
        dataset: {"counts": dict(counts), "examples": examples[dataset]}
        for dataset, counts in datasets.items()
    }


def audit_audio_records(
    records: Sequence[ManifestRecord],
    *,
    minimum_rms: float = 1e-5,
) -> tuple[list[ManifestRecord], Counter[str], dict[str, Any]]:
    cache = _RowGroupCache(max_entries=2)
    rejected: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    signal: Counter[str] = Counter()
    total_decoded_seconds = 0.0
    accepted: list[ManifestRecord] = []

    ordered = sorted(records, key=lambda r: (r.parquet_path, r.row_group, r.row_in_group))
    for record in ordered:
        try:
            column = cache.get_column(record.parquet_path, record.row_group, record.audio_column)
            payload = extract_audio_bytes(column[record.row_in_group])
            audit = audit_audio_bytes(payload)
            validate_duration(record.duration, audit.duration)
            if audit.nonfinite_count:
                raise ValueError("non-finite decoded samples")
            if audit.rms < minimum_rms:
                raise ValueError("near-silent audio")
        except Exception as exc:
            reason = type(exc).__name__ + ":" + str(exc).split(":", 1)[0]
            rejected[f"audio_decode_or_validation:{reason}"] += 1
            continue

        record.audio_sha1 = audit.sha1
        record.audio_pcm_sha1 = audit.pcm_sha1
        record.audio_format = f"{audit.format}/{audit.subtype}".strip("/")
        record.sample_rate = audit.original_sample_rate
        record.channels = audit.channels
        record.decoded_duration = audit.duration
        accepted.append(record)
        formats[record.audio_format] += 1
        sample_rates[str(record.sample_rate)] += 1
        channels[str(record.channels)] += 1
        total_decoded_seconds += audit.duration
        signal["clipped_fraction_ge_0.001"] += audit.clipped_fraction >= 0.001
        signal["clipped_fraction_ge_0.01"] += audit.clipped_fraction >= 0.01
        signal["abs_dc_offset_ge_0.02"] += abs(audit.dc_offset) >= 0.02
        signal["rms_below_0.001"] += audit.rms < 0.001
        signal["rms_above_0.25"] += audit.rms > 0.25

    deduplicated, dedup_stats = remove_audio_duplicates(accepted)
    rejected.update(dedup_stats)
    audit_manifest_leakage(deduplicated)
    return deduplicated, rejected, {
        "formats": dict(formats),
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "signal_flags": dict(signal),
        "decoded_hours": round(total_decoded_seconds / 3600.0, 4),
    }


def remove_audio_duplicates(records: Sequence[ManifestRecord]) -> tuple[list[ManifestRecord], Counter[str]]:
    groups: defaultdict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        identity = record.audio_pcm_sha1 or record.audio_sha1
        if not identity:
            raise ValueError("Audio deduplication requires every record to have an audited hash")
        groups[identity].append(record)

    stats: Counter[str] = Counter()
    kept: list[ManifestRecord] = []
    split_rank = {"test": 0, "validation": 1, "train": 2}
    for group in groups.values():
        fingerprints = {record.text_fingerprint for record in group}
        if len(fingerprints) > 1:
            stats["conflicting_transcripts_same_audio_removed"] += len(group)
            continue
        group = sorted(group, key=lambda record: (split_rank[record.split], record.id))
        kept.append(group[0])
        stats["duplicate_audio_removed"] += len(group) - 1
    return kept, stats


def select_smoke_subset(
    records: Sequence[ManifestRecord],
    split: str,
    target_hours: float | None = None,
    max_examples: int | None = None,
    seed: int = 42,
) -> list[ManifestRecord]:
    by_dataset: defaultdict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.split == split:
            by_dataset[record.dataset].append(record)
    for values in by_dataset.values():
        values.sort(key=lambda record: stable_bucket(f"smoke:{seed}:{record.id}", modulo=2**31 - 1))

    selected: list[ManifestRecord] = []
    seconds = 0.0
    names = sorted(by_dataset)
    cursor = 0
    while names:
        name = names[cursor % len(names)]
        values = by_dataset[name]
        if not values:
            names.remove(name)
            continue
        record = values.pop(0)
        selected.append(record)
        seconds += record.duration
        if max_examples is not None and len(selected) >= max_examples:
            break
        if target_hours is not None and seconds >= target_hours * 3600:
            break
        cursor += 1
    return selected


def records_as_dicts(records: Sequence[ManifestRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
