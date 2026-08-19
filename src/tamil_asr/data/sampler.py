from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class DurationBatchSampler:

    def __init__(
        self,
        durations: Sequence[float],
        max_batch_audio_seconds: float,
        max_batch_size: int,
        bucket_size: int = 256,
        seed: int = 42,
        shuffle: bool = True,
    ):
        if not durations:
            raise ValueError("DurationBatchSampler requires at least one example")
        if max_batch_audio_seconds <= 0 or max_batch_size <= 0 or bucket_size <= 0:
            raise ValueError("Batch duration, batch size, and bucket size must be positive")
        self.durations = [float(value) for value in durations]
        if any(value <= 0 for value in self.durations):
            raise ValueError("All durations must be positive")
        self.max_batch_audio_seconds = float(max_batch_audio_seconds)
        self.max_batch_size = int(max_batch_size)
        self.bucket_size = int(bucket_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.durations)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            bucket.sort(key=self.durations.__getitem__)
            if self.shuffle and rng.random() < 0.5:
                bucket.reverse()

        batches: list[list[int]] = []
        for bucket in buckets:
            current: list[int] = []
            seconds = 0.0
            for index in bucket:
                duration = self.durations[index]
                if duration > self.max_batch_audio_seconds:
                    raise ValueError(
                        f"Example {index} duration {duration:.3f}s exceeds max_batch_audio_seconds="
                        f"{self.max_batch_audio_seconds:.3f}s"
                    )
                would_overflow = current and (
                    seconds + duration > self.max_batch_audio_seconds or len(current) >= self.max_batch_size
                )
                if would_overflow:
                    batches.append(current)
                    current = []
                    seconds = 0.0
                current.append(index)
                seconds += duration
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
