from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def script_slice(text: str) -> str:
    has_tamil = any(0x0B80 <= ord(character) <= 0x0BFF for character in text)
    has_latin = any(
        "A" <= character <= "Z" or "a" <= character <= "z" for character in text
    )
    if has_tamil and has_latin:
        return "tamil_latin_code_switch"
    if has_tamil:
        return "tamil_script"
    if has_latin:
        return "latin_script"
    return "other_script"


def duration_slice(duration: float) -> str:
    if duration < 5.0:
        return "under_5s"
    if duration < 15.0:
        return "5_to_15s"
    if duration < 30.0:
        return "15_to_30s"
    return "30s_and_over"


def severity_slice(scenario: str, severity: float | None) -> str:
    if not scenario and severity is None:
        return "not_targeted"
    if severity is None:
        return "targeted_unspecified"
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Acoustic severity must be in [0, 1], found {severity}")
    if value <= 1.0 / 3.0:
        return "mild"
    if value <= 2.0 / 3.0:
        return "moderate"
    return "severe"


def evaluation_slices(row: Mapping[str, Any]) -> dict[str, str]:
    reference = str(row.get("reference", ""))
    scenario = str(row.get("scenario", "")).strip()
    raw_severity = row.get("severity")
    severity = None if raw_severity is None else float(raw_severity)
    values = {
        "dataset": str(row.get("dataset", "unknown")).strip() or "unknown",
        "domain": str(row.get("domain", "unknown")).strip() or "unknown",
        "source_split": str(row.get("source_split", "unknown")).strip() or "unknown",
        "script": script_slice(reference),
        "duration": duration_slice(float(row.get("duration", 0.0))),
        "scenario": scenario or "not_targeted",
        "severity": severity_slice(scenario, severity),
    }
    speaker = str(row.get("speaker_id", "")).strip()
    if speaker:
        values["speaker"] = speaker
    return values
