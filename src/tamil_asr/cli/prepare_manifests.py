from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from tamil_asr.data.audit import audit_audio_records, select_smoke_subset, text_profile
from tamil_asr.data.manifest import (
    assign_splits,
    audit_manifest_leakage,
    inspect_dataset_schema,
    scan_dataset,
    summarize_by_dataset,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe Tamil ASR manifests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-audit", choices=["none", "smoke", "all"], default="all")
    parser.add_argument("--smoke-train-hours", type=float, default=5.0)
    parser.add_argument("--smoke-validation-examples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("language") != "ta":
        raise ValueError("This preparation command only accepts an explicit Tamil configuration")

    all_records = []
    rejection_counts: Counter[str] = Counter()
    enabled_configs = {}
    schema_profiles = {}
    for name, dataset_config in config["datasets"].items():
        if not dataset_config.get("enabled", False):
            continue
        enabled_configs[name] = dataset_config
        schema_profiles[name] = inspect_dataset_schema(name, dataset_config)
        records, rejected = scan_dataset(name, dataset_config, config["defaults"])
        if not records:
            raise ValueError(f"{name}: schema scan and metadata filters accepted zero records")
        all_records.extend(records)
        rejection_counts.update({f"{name}:{key}": value for key, value in rejected.items()})

    records, split_stats = assign_splits(
        all_records,
        enabled_configs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    rejection_counts.update(split_stats)

    audio_profile = {"status": "not_run"}
    if args.audio_audit in {"smoke", "all"}:
        records_to_audit = records
        if args.audio_audit == "smoke":
            records_to_audit = [
                *select_smoke_subset(
                    records, "train", target_hours=args.smoke_train_hours, seed=args.seed
                ),
                *select_smoke_subset(
                    records,
                    "validation",
                    max_examples=args.smoke_validation_examples,
                    seed=args.seed,
                ),
            ]
        records, audio_rejections, audio_profile = audit_audio_records(records_to_audit)
        rejection_counts.update(audio_rejections)
        surviving_datasets = {record.dataset for record in records}
        missing_datasets = set(enabled_configs).difference(surviving_datasets)
        if missing_datasets:
            raise ValueError(
                "Full audio audit removed every row from enabled datasets: "
                f"{sorted(missing_datasets)}"
            )

    overall = audit_manifest_leakage(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", (record for record in records if record.split == split))

    smoke_train = select_smoke_subset(
        records, "train", target_hours=args.smoke_train_hours, seed=args.seed
    )
    smoke_validation = select_smoke_subset(
        records, "validation", max_examples=args.smoke_validation_examples, seed=args.seed
    )
    if not smoke_train:
        raise ValueError("No smoke-training records were selected")
    if not smoke_validation:
        raise ValueError("No smoke-validation records were selected")
    write_jsonl(output / "smoke_train.jsonl", smoke_train)
    write_jsonl(output / "smoke_validation.jsonl", smoke_validation)

    report = {
        "config": str(Path(args.config).resolve()),
        "audio_audit": args.audio_audit,
        "overall": overall,
        "by_dataset": summarize_by_dataset(records),
        "rejections": dict(rejection_counts),
        "audio_profile": audio_profile,
        "schema_profiles": schema_profiles,
        "text_profile": text_profile(records),
        "smoke": {
            "train_examples": len(smoke_train),
            "train_hours": round(sum(record.duration for record in smoke_train) / 3600.0, 4),
            "validation_examples": len(smoke_validation),
            "validation_hours": round(sum(record.duration for record in smoke_validation) / 3600.0, 4),
        },
    }
    temporary = output / "audit_report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "audit_report.json")
    print(json.dumps({"status": "ok", **report["overall"], "smoke": report["smoke"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
