from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from torch.utils.data import IterableDataset


def discover_text_parquet_files(roots: Sequence[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix == ".parquet":
            files.add(root.resolve())
        elif root.is_dir():
            files.update(path.resolve() for path in root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {[str(root) for root in roots]}")
    return sorted(files)


def _validation_bucket(doc_id: str) -> int:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


class ParquetTextBlockStream(IterableDataset[list[int]]):

    def __init__(
        self,
        files: Sequence[str | Path],
        tokenizer: Any,
        block_size: int,
        split: str,
        validation_fraction: float = 0.01,
        parquet_batch_size: int = 512,
        max_text_chars: int | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if parquet_batch_size <= 0:
            raise ValueError("parquet_batch_size must be positive")
        self.files = [Path(path) for path in files]
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.split = split
        self.validation_cutoff = int(float(validation_fraction) * 10_000)
        self.parquet_batch_size = int(parquet_batch_size)
        self.max_text_chars = int(max_text_chars) if max_text_chars else None
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id for text-block packing")

    def _belongs_to_split(self, doc_id: str) -> bool:
        is_validation = _validation_bucket(doc_id) < self.validation_cutoff
        return is_validation if self.split == "validation" else not is_validation

    def _rows(self) -> Iterator[tuple[str, str]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for Parquet text streaming") from exc

        for path in self.files:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            if "text" not in names:
                raise ValueError(f"{path}: expected a text column, found {sorted(names)}")
            id_column = "doc_id" if "doc_id" in names else None
            for batch in parquet.iter_batches(
                batch_size=self.parquet_batch_size,
                columns=[column for column in (id_column, "text") if column is not None],
            ):
                columns = batch.to_pydict()
                texts = columns["text"]
                ids = columns.get("doc_id", [""] * len(texts))
                for row_index, (raw_id, raw_text) in enumerate(zip(ids, texts)):
                    text = str(raw_text or "").strip()
                    if not text:
                        continue
                    if self.max_text_chars is not None and len(text) > self.max_text_chars:
                        text = text[: self.max_text_chars]
                    doc_id = str(raw_id or f"{path}:{row_index}")
                    if self._belongs_to_split(doc_id):
                        yield doc_id, text

    def __iter__(self) -> Iterator[list[int]]:
        token_buffer: list[int] = []
        for _, text in self._rows():
            encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not encoded:
                continue
            token_buffer.extend(int(token) for token in encoded)
            token_buffer.append(int(self.eos_token_id))
            while len(token_buffer) >= self.block_size:
                block = token_buffer[: self.block_size]
                del token_buffer[: self.block_size]
                yield block


class TextBlockCollator:
    def __call__(self, blocks: Iterable[list[int]]) -> dict[str, Any]:
        import torch

        values = list(blocks)
        if not values:
            raise ValueError("Cannot collate an empty text block batch")
        input_ids = torch.tensor(values, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }
