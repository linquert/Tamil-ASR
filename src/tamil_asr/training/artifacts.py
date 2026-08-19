from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_metrics(output_dir: Path, step: int, phase: str, values: dict[str, Any]) -> None:
    payload = {"step": step, "phase": phase, **values}
    with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def record_evaluation_rows(output_dir: Path, step: int, rows: list[dict[str, Any]]) -> Path:
    directory = output_dir / "evaluations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:06d}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
