from __future__ import annotations

import argparse
import json
from pathlib import Path

from tamil_asr.data.manifest import ManifestRecord, read_jsonl
from tamil_asr.data.tokenizer_audit import audit_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Qwen tokenization on Tamil ASR targets")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--worst-examples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    records = [ManifestRecord.from_dict(row) for row in read_jsonl(args.manifest)]
    if args.max_examples is not None:
        records = records[: args.max_examples]
    report = audit_tokenizer(records, tokenizer, worst_examples=args.worst_examples)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
