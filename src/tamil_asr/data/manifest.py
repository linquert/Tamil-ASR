from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .text import normalize_transcript, stable_bucket, tamil_fraction, transcript_fingerprint


_LATIN_RE = re.compile(r"[A-Za-z]")
_ANNOTATION_RE = re.compile(r"\[[^\]\n]{1,80}\]|<[^<>\n]{1,80}>")


@dataclass(slots=True)
class ManifestRecord:
    id: str
    dataset: str
    split: str
    source_split: str
    parquet_path: str
    row_group: int
    row_in_group: int
    audio_column: str
    text_column: str
    text: str
    duration: float
    text_fingerprint: str
    speaker_id: str = ""
    domain: str = ""
    alignment_score: float | None = None
    audio_sha1: str = ""
    audio_pcm_sha1: str = ""
    audio_format: str = ""
    sample_rate: int = 0
    channels: int = 0
    decoded_duration: float = 0.0
    text_variants: dict[str, str] = field(default_factory=dict)
    base_prediction: str = ""
    base_wer: float | None = None
    scenario: str = ""
    severity: float | None = None

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "ManifestRecord":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: item[key] for key in allowed if key in item})


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def filter_records_by_base_wer(
    records: Sequence[ManifestRecord], max_base_wer: float
) -> list[ManifestRecord]:
    threshold = float(max_base_wer)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("max_base_wer must be finite and in (0, 1]")
    missing = [record.id for record in records if record.base_wer is None]
    if missing:
        raise ValueError(
            "Acoustic WER curriculum requires base_wer for every record; "
            f"missing metadata for {len(missing)} records (first={missing[0]})"
        )
    invalid = [
        record.id
        for record in records
        if not math.isfinite(float(record.base_wer)) or not 0.0 <= float(record.base_wer) <= 1.0
    ]
    if invalid:
        raise ValueError(
            f"base_wer must be finite and in [0, 1]; invalid metadata for {len(invalid)} records "
            f"(first={invalid[0]})"
        )
    selected = [record for record in records if float(record.base_wer) < threshold]
    if not selected:
        raise ValueError(f"WER curriculum threshold {threshold:.2f} selected zero records")
    return selected


def write_jsonl(path: str | os.PathLike[str], records: Iterable[ManifestRecord]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    count = 0
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return count


def _first_existing(candidates: Sequence[str], names: set[str], kind: str, dataset: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"{dataset}: no {kind} column found; candidates={list(candidates)}, schema={sorted(names)}")


def _optional_existing(candidates: Sequence[str], names: set[str]) -> list[str]:
    return [candidate for candidate in candidates if candidate in names]


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_value(batch: Mapping[str, list[Any]], columns: Sequence[str], index: int) -> Any:
    for column in columns:
        value = batch[column][index]
        if value is not None and str(value).strip():
            return value
    return None


def discover_parquet_files(dataset_cfg: Mapping[str, Any]) -> list[tuple[str, Path]]:
    root = Path(dataset_cfg["root"])
    found: list[tuple[str, Path]] = []
    for source_split, patterns in dataset_cfg.get("globs", {}).items():
        for pattern in patterns:
            found.extend((source_split, path) for path in sorted(root.glob(pattern)))
    unique = {(source_split, path.resolve()): None for source_split, path in found}
    return sorted(unique, key=lambda item: (item[0], str(item[1])))


def inspect_dataset_schema(
    name: str,
    dataset_cfg: Mapping[str, Any],
    sample_rows: int = 5_000,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inspect Parquet datasets") from exc

    files = discover_parquet_files(dataset_cfg)
    if not files:
        raise FileNotFoundError(f"{name}: no Parquet files matched under {dataset_cfg['root']}")
    first = pq.ParquetFile(files[0][1])
    schema = first.schema_arrow
    names = set(schema.names)
    audio_column = dataset_cfg.get("audio_column") or _first_existing(
        dataset_cfg["audio_columns"], names, "audio", name
    )
    text_column = dataset_cfg.get("text_column") or _first_existing(
        dataset_cfg["text_columns"], names, "text", name
    )
    audit_columns = [
        column
        for column in dataset_cfg.get("audit_text_columns", dataset_cfg.get("text_columns", []))
        if column in names and column != text_column
    ]
    audio_schema_type = str(schema.field(audio_column).type)
    if not (
        audio_schema_type.startswith("struct<")
        or audio_schema_type in {"binary", "large_binary"}
    ):
        raise ValueError(
            f"{name}: audio column {audio_column!r} has path-only type {audio_schema_type}; "
            "this pipeline requires embedded audio bytes"
        )
    columns = [text_column, *audit_columns]
    equality: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    selected_text_profile: Counter[str] = Counter()
    selected_text_examples: list[dict[str, Any]] = []
    inspected = 0

    for _, path in files:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=[column for column in columns if column in parquet.schema_arrow.names])
            batch = table.to_pydict()
            for index in range(table.num_rows):
                selected = batch[text_column][index]
                if selected is None:
                    nulls[text_column] += 1
                normalized_selected = normalize_transcript(selected)
                fraction = tamil_fraction(normalized_selected)
                if fraction >= 0.95:
                    selected_text_profile["tamil_fraction_ge_0.95"] += 1
                elif fraction >= 0.50:
                    selected_text_profile["tamil_fraction_0.50_0.95"] += 1
                elif fraction > 0:
                    selected_text_profile["tamil_fraction_0_0.50"] += 1
                else:
                    selected_text_profile["no_tamil_characters"] += 1
                selected_text_profile["contains_latin"] += bool(_LATIN_RE.search(normalized_selected))
                selected_text_profile["contains_annotation"] += bool(_ANNOTATION_RE.search(normalized_selected))
                selected_text_profile["characters"] += len(normalized_selected)
                for column in audit_columns:
                    value = batch[column][index]
                    if value is None:
                        nulls[column] += 1
                    elif normalize_transcript(value) == normalize_transcript(selected):
                        equality[f"{text_column}=={column}"] += 1
                    else:
                        equality[f"{text_column}!={column}"] += 1
                if len(selected_text_examples) < 8:
                    selected_text_examples.append(
                        {
                            "selected": normalized_selected,
                            "alternatives": {
                                column: normalize_transcript(batch[column][index])
                                for column in audit_columns
                                if batch[column][index] is not None
                            },
                        }
                    )
                inspected += 1
                if inspected >= sample_rows:
                    break
            if inspected >= sample_rows:
                break
        if inspected >= sample_rows:
            break

    return {
        "files": len(files),
        "first_file": str(files[0][1]),
        "schema": {field.name: str(field.type) for field in schema},
        "selected_audio_column": audio_column,
        "selected_text_column": text_column,
        "audited_alternative_text_columns": audit_columns,
        "sample_rows": inspected,
        "text_nulls": dict(nulls),
        "text_equality": dict(equality),
        "selected_text_profile": dict(selected_text_profile),
        "selected_text_examples": selected_text_examples,
        "audio_schema_type": audio_schema_type,
    }


def scan_dataset(
    name: str,
    dataset_cfg: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[list[ManifestRecord], Counter[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to scan Parquet datasets") from exc

    files = discover_parquet_files(dataset_cfg)
    if not files:
        raise FileNotFoundError(f"{name}: no Parquet files matched under {dataset_cfg['root']}")

    min_duration = float(dataset_cfg.get("min_duration", defaults["min_duration"]))
    max_duration = float(dataset_cfg.get("max_duration", defaults["max_duration"]))
    min_chars = int(dataset_cfg.get("min_text_chars", defaults["min_text_chars"]))
    min_cps = float(dataset_cfg.get("min_chars_per_second", defaults["min_chars_per_second"]))
    max_cps = float(dataset_cfg.get("max_chars_per_second", defaults["max_chars_per_second"]))
    min_tamil = float(dataset_cfg.get("min_tamil_fraction", defaults["min_tamil_fraction"]))
    min_alignment = dataset_cfg.get("min_alignment_score")

    records: list[ManifestRecord] = []
    rejected: Counter[str] = Counter()

    for source_split, path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        configured_audio = dataset_cfg.get("audio_column")
        if configured_audio:
            if configured_audio not in names:
                raise ValueError(f"{name}: required audio_column={configured_audio!r} missing from schema")
            audio_column = configured_audio
        else:
            audio_column = _first_existing(dataset_cfg["audio_columns"], names, "audio", name)
        configured_text = dataset_cfg.get("text_column")
        if configured_text:
            if configured_text not in names:
                raise ValueError(f"{name}: required text_column={configured_text!r} missing from schema")
            text_columns = [configured_text]
        else:
            text_columns = _optional_existing(dataset_cfg["text_columns"], names)
            if not text_columns:
                _first_existing(dataset_cfg["text_columns"], names, "text", name)
        audit_text_columns = _optional_existing(dataset_cfg.get("audit_text_columns", []), names)
        duration_columns = _optional_existing(dataset_cfg.get("duration_columns", []), names)
        if not duration_columns:
            raise ValueError(f"{name}: duration metadata is required for leakage-safe batching and filtering")
        speaker_columns = _optional_existing(dataset_cfg.get("speaker_columns", []), names)
        domain_columns = _optional_existing(dataset_cfg.get("domain_columns", []), names)
        alignment_columns = _optional_existing(dataset_cfg.get("alignment_columns", []), names)

        columns = list(
            dict.fromkeys(
                text_columns
                + audit_text_columns
                + duration_columns
                + speaker_columns
                + domain_columns
                + alignment_columns
            )
        )
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            batch = table.to_pydict()
            for row_in_group in range(table.num_rows):
                text = normalize_transcript(_first_value(batch, text_columns, row_in_group))
                duration = _to_float(_first_value(batch, duration_columns, row_in_group))
                alignment = _to_float(_first_value(batch, alignment_columns, row_in_group)) if alignment_columns else None
                variants = {
                    column: normalize_transcript(batch[column][row_in_group])
                    for column in audit_text_columns
                    if batch[column][row_in_group] is not None
                    and normalize_transcript(batch[column][row_in_group])
                }

                if len(text) < min_chars:
                    rejected["empty_or_short_text"] += 1
                    continue
                if duration is None:
                    rejected["invalid_duration"] += 1
                    continue
                if not min_duration <= duration <= max_duration:
                    rejected["duration_out_of_range"] += 1
                    continue
                chars_per_second = len(text) / duration
                if not min_cps <= chars_per_second <= max_cps:
                    rejected["implausible_text_audio_ratio"] += 1
                    continue
                if tamil_fraction(text) < min_tamil:
                    rejected["low_tamil_fraction"] += 1
                    continue
                if min_alignment is not None and (alignment is None or alignment < float(min_alignment)):
                    rejected["low_or_missing_alignment"] += 1
                    continue

                speaker = normalize_transcript(_first_value(batch, speaker_columns, row_in_group)) if speaker_columns else ""
                domain = normalize_transcript(_first_value(batch, domain_columns, row_in_group)) if domain_columns else ""
                relative = str(path.relative_to(Path(dataset_cfg["root"])))
                record_id = f"{name}:{relative}:rg{row_group}:r{row_in_group}"
                records.append(
                    ManifestRecord(
                        id=record_id,
                        dataset=name,
                        split="",
                        source_split=source_split,
                        parquet_path=str(path),
                        row_group=row_group,
                        row_in_group=row_in_group,
                        audio_column=audio_column,
                        text_column=text_columns[0],
                        text=text,
                        duration=duration,
                        text_fingerprint=transcript_fingerprint(text),
                        speaker_id=speaker,
                        domain=domain,
                        alignment_score=alignment,
                        text_variants=variants,
                    )
                )

    return records, rejected


def assign_splits(
    records: Sequence[ManifestRecord],
    dataset_configs: Mapping[str, Mapping[str, Any]],
    validation_fraction: float = 0.01,
    test_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[list[ManifestRecord], Counter[str]]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than one")

    validation_cutoff = int(validation_fraction * 10_000)
    test_cutoff = validation_cutoff + int(test_fraction * 10_000)
    stats: Counter[str] = Counter()

    for record in records:
        config = dataset_configs[record.dataset]
        official_only = bool(config.get("use_official_eval_only", False))
        source = record.source_split.lower()
        if source in {"validation", "valid", "dev"}:
            record.split = "validation"
        elif source == "test":
            record.split = "test"
        elif official_only:
            record.split = "train"
        else:
            group = f"speaker:{record.speaker_id}" if record.speaker_id else f"text:{record.text_fingerprint}"
            bucket = stable_bucket(f"{seed}:{record.dataset}:{group}")
            if bucket < validation_cutoff:
                record.split = "validation"
            elif bucket < test_cutoff:
                record.split = "test"
            else:
                record.split = "train"

    test_fingerprints = {r.text_fingerprint for r in records if r.split == "test"}
    test_speakers = {
        (r.dataset, r.speaker_id) for r in records if r.split == "test" and r.speaker_id
    }
    eval_pruned: list[ManifestRecord] = []
    for record in records:
        if record.split == "validation" and record.text_fingerprint in test_fingerprints:
            stats["validation_removed_test_text_overlap"] += 1
            continue
        if (
            record.split == "validation"
            and record.speaker_id
            and (record.dataset, record.speaker_id) in test_speakers
        ):
            stats["validation_removed_test_speaker_overlap"] += 1
            continue
        eval_pruned.append(record)
    records = eval_pruned

    eval_fingerprints = {r.text_fingerprint for r in records if r.split in {"validation", "test"}}
    eval_speakers = {
        (r.dataset, r.speaker_id)
        for r in records
        if r.split in {"validation", "test"} and r.speaker_id
    }

    clean: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    for record in records:
        if record.id in seen_ids:
            stats["duplicate_record_id"] += 1
            continue
        seen_ids.add(record.id)
        if record.split == "train" and record.text_fingerprint in eval_fingerprints:
            stats["train_removed_eval_text_overlap"] += 1
            continue
        if record.split == "train" and record.speaker_id and (record.dataset, record.speaker_id) in eval_speakers:
            stats["train_removed_eval_speaker_overlap"] += 1
            continue
        clean.append(record)

    audit_manifest_leakage(clean)
    return clean, stats


def audit_manifest_leakage(records: Sequence[ManifestRecord]) -> dict[str, Any]:
    ids: set[str] = set()
    fingerprints: defaultdict[str, set[str]] = defaultdict(set)
    speakers: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    hours: Counter[str] = Counter()

    for record in records:
        if record.id in ids:
            raise ValueError(f"Duplicate manifest id: {record.id}")
        ids.add(record.id)
        if record.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split {record.split!r} for {record.id}")
        fingerprints[record.text_fingerprint].add(record.split)
        if record.speaker_id:
            speakers[(record.dataset, record.speaker_id)].add(record.split)
        counts[record.split] += 1
        hours[record.split] += record.duration / 3600.0

    text_leaks = {key: value for key, value in fingerprints.items() if len(value) > 1}
    speaker_leaks = {key: value for key, value in speakers.items() if len(value) > 1}
    if text_leaks:
        raise ValueError(f"Transcript leakage across train/eval: {len(text_leaks)} fingerprints")
    if speaker_leaks:
        raise ValueError(f"Speaker leakage across train/eval: {len(speaker_leaks)} speakers")

    return {
        "counts": dict(counts),
        "hours": {key: round(value, 4) for key, value in hours.items()},
        "unique_text_fingerprints": len(fingerprints),
        "unique_scoped_speakers": len(speakers),
    }


def summarize_by_dataset(records: Sequence[ManifestRecord]) -> dict[str, Any]:
    summary: defaultdict[str, defaultdict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"examples": 0, "hours": 0.0})
    )
    for record in records:
        cell = summary[record.dataset][record.split]
        cell["examples"] += 1
        cell["hours"] += record.duration / 3600.0
    return {
        dataset: {
            split: {"examples": int(values["examples"]), "hours": round(values["hours"], 4)}
            for split, values in splits.items()
        }
        for dataset, splits in summary.items()
    }
