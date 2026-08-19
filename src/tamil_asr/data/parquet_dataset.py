from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Sequence

from .audio import decode_audio_bytes, extract_audio_bytes, validate_duration
from .manifest import ManifestRecord


class _RowGroupCache:
    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, int, str], list[Any]] = OrderedDict()

    def get_column(self, path: str, row_group: int, column: str) -> list[Any]:
        key = (path, row_group, column)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for Parquet audio loading") from exc
        table = pq.ParquetFile(path).read_row_group(row_group, columns=[column])
        values = table[column].to_pylist()
        self._cache[key] = values
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return values


class ParquetAudioDataset:

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        sample_rate: int = 16_000,
        cache_row_groups: int = 4,
        verify_duration: bool = True,
    ):
        self.records = list(records)
        self.sample_rate = int(sample_rate)
        self.verify_duration = verify_duration
        self._cache = _RowGroupCache(cache_row_groups)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        column = self._cache.get_column(record.parquet_path, record.row_group, record.audio_column)
        if not 0 <= record.row_in_group < len(column):
            raise IndexError(f"{record.id}: row_in_group is outside row group")
        payload = extract_audio_bytes(column[record.row_in_group])
        waveform, sample_rate, info = decode_audio_bytes(payload, target_sample_rate=self.sample_rate)
        if self.verify_duration:
            validate_duration(record.duration, float(info.duration))
        if record.audio_sha1 and hashlib.sha1(payload).hexdigest() != record.audio_sha1:
            raise ValueError(f"{record.id}: embedded audio hash differs from audited manifest")
        return {
            "id": record.id,
            "dataset": record.dataset,
            "source_split": record.source_split,
            "domain": record.domain,
            "speaker_id": record.speaker_id,
            "duration": record.duration,
            "base_wer": record.base_wer,
            "scenario": record.scenario,
            "severity": record.severity,
            "audio": waveform,
            "sample_rate": sample_rate,
            "text": record.text,
        }
