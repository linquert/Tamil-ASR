from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class AudioAudit:
    sha1: str
    pcm_sha1: str
    format: str
    subtype: str
    original_sample_rate: int
    channels: int
    frames: int
    duration: float
    peak: float
    rms: float
    dc_offset: float
    clipped_fraction: float
    nonfinite_count: int


def extract_audio_bytes(value: Any, *, allow_existing_path: bool = False) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        embedded = value.get("bytes")
        if isinstance(embedded, memoryview):
            embedded = embedded.tobytes()
        if isinstance(embedded, (bytes, bytearray)) and embedded:
            return bytes(embedded)
        path = value.get("path")
        if allow_existing_path and path and Path(path).is_file():
            return Path(path).read_bytes()
        raise ValueError("Audio struct has no embedded bytes; refusing an unverified or stale path")
    if isinstance(value, (str, os.PathLike)):
        if allow_existing_path and Path(value).is_file():
            return Path(value).read_bytes()
        raise ValueError("Path-only audio is disabled unless allow_existing_path=True and the path exists")
    raise TypeError(f"Unsupported audio cell type: {type(value).__name__}")


def decode_audio_bytes(payload: bytes, target_sample_rate: int | None = None):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("numpy and soundfile are required to decode embedded audio") from exc

    if not payload:
        raise ValueError("Empty audio payload")
    stream = io.BytesIO(payload)
    info = sf.info(stream)
    stream.seek(0)
    waveform, sample_rate = sf.read(stream, dtype="float32", always_2d=True)
    if waveform.size == 0:
        raise ValueError("Decoded audio contains zero frames")
    waveform = waveform.mean(axis=1, dtype=np.float32)

    if target_sample_rate and sample_rate != target_sample_rate:
        try:
            import soxr
        except ImportError as exc:
            raise RuntimeError("soxr is required for high-quality sample-rate conversion") from exc
        waveform = soxr.resample(waveform, sample_rate, target_sample_rate, quality="HQ").astype(np.float32)
        sample_rate = target_sample_rate
    return waveform, int(sample_rate), info


def audit_audio_bytes(payload: bytes) -> AudioAudit:
    import numpy as np

    waveform, sample_rate, info = decode_audio_bytes(payload)
    nonfinite = int((~np.isfinite(waveform)).sum())
    if nonfinite:
        peak = rms = dc = clipped = math.nan
    else:
        absolute = np.abs(waveform)
        peak = float(absolute.max())
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        dc = float(np.mean(waveform, dtype=np.float64))
        clipped = float(np.mean(absolute >= 0.999))
    return AudioAudit(
        sha1=hashlib.sha1(payload).hexdigest(),
        pcm_sha1=hashlib.sha1(
            f"{sample_rate}:".encode("ascii") + waveform.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        format=str(info.format or ""),
        subtype=str(info.subtype or ""),
        original_sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration=float(info.duration),
        peak=peak,
        rms=rms,
        dc_offset=dc,
        clipped_fraction=clipped,
        nonfinite_count=nonfinite,
    )


def validate_duration(metadata_duration: float, decoded_duration: float) -> None:
    tolerance = max(0.25, 0.02 * max(metadata_duration, decoded_duration))
    if abs(metadata_duration - decoded_duration) > tolerance:
        raise ValueError(
            f"Duration mismatch: metadata={metadata_duration:.3f}s decoded={decoded_duration:.3f}s "
            f"tolerance={tolerance:.3f}s"
        )
