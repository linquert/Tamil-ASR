

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable


def _set_use_cache(thinker: Any, enabled: bool) -> list[tuple[Any, bool]]:
    base = thinker.get_base_model() if hasattr(thinker, "get_base_model") else thinker
    configs = [getattr(base, "config", None), getattr(getattr(base, "config", None), "text_config", None)]
    previous: list[tuple[Any, bool]] = []
    seen: set[int] = set()
    for config in configs:
        if config is None or id(config) in seen or not hasattr(config, "use_cache"):
            continue
        seen.add(id(config))
        previous.append((config, bool(config.use_cache)))
        config.use_cache = enabled
    return previous


def generate_text_continuation(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    *,
    attention_mask: Any | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> tuple[str, int]:
    import torch

    thinker = getattr(model, "thinker", model)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    max_new_tokens = max(1, min(int(max_new_tokens), 512))
    temperature = max(0.0, min(float(temperature), 2.0))
    top_p = max(0.05, min(float(top_p), 1.0))
    do_sample = temperature > 0.0
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})

    was_training = bool(model.training)
    previous_cache = _set_use_cache(thinker, True)
    generation_config = getattr(thinker, "generation_config", None)
    previous_generation_temperature = getattr(generation_config, "temperature", None)
    if not do_sample and generation_config is not None and hasattr(generation_config, "temperature"):
        generation_config.temperature = None
    model.eval()
    try:
        with torch.inference_mode():
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                result = thinker.generate(**kwargs)
        sequences = getattr(result, "sequences", result)
        generated = sequences[0, input_ids.shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, int(generated.numel())
    finally:
        if generation_config is not None and hasattr(generation_config, "temperature"):
            generation_config.temperature = previous_generation_temperature
        for config, value in previous_cache:
            config.use_cache = value
        if was_training:
            model.train()


def parameter_dtype_summary(model: Any) -> dict[str, dict[str, int]]:
    all_parameters: Counter[str] = Counter()
    trainable_parameters: Counter[str] = Counter()
    for parameter in model.parameters():
        name = str(parameter.dtype).removeprefix("torch.")
        all_parameters[name] += int(parameter.numel())
        if parameter.requires_grad:
            trainable_parameters[name] += int(parameter.numel())
    return {
        "parameter_dtypes": dict(sorted(all_parameters.items())),
        "trainable_parameter_dtypes": dict(sorted(trainable_parameters.items())),
    }


def collect_validation_blocks(loader: Any, max_examples: int) -> list[Any]:
    import torch

    blocks: list[Any] = []
    if max_examples <= 0:
        return blocks
    for packed in loader:
        values = packed["input_ids"]
        for row in values:
            blocks.append(row.detach().cpu() if torch.is_tensor(row) else row)
            if len(blocks) >= max_examples:
                return blocks
    return blocks


def collect_validation_blocks_by_root(
    roots: Iterable[str | Path],
    tokenizer: Any,
    *,
    block_size: int,
    validation_fraction: float,
    parquet_batch_size: int,
    max_text_chars: int,
    batch_size: int,
    max_examples: int,
) -> list[tuple[Any, str]]:
    if max_examples <= 0:
        return []
    root_values = [Path(root) for root in roots]
    if not root_values:
        return []
    import math
    from torch.utils.data import DataLoader

    per_root = max(1, math.ceil(max_examples / len(root_values)))
    balanced: list[tuple[Any, str]] = []
    from tamil_asr.data.text_corpus import ParquetTextBlockStream, TextBlockCollator, discover_text_parquet_files

    for root in root_values:
        files = discover_text_parquet_files([root])
        stream = ParquetTextBlockStream(
            files,
            tokenizer,
            int(block_size),
            "validation",
            float(validation_fraction),
            int(parquet_batch_size),
            int(max_text_chars),
        )
        loader = DataLoader(
            stream,
            batch_size=int(batch_size),
            collate_fn=TextBlockCollator(),
            num_workers=0,
        )
        balanced.extend((block, root.name) for block in collect_validation_blocks(loader, per_root))
    return balanced[:max_examples]


def generate_validation_examples(
    model: Any,
    tokenizer: Any,
    blocks: Iterable[Any],
    device: Any,
    *,
    context_tokens: int = 128,
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for index, item in enumerate(blocks):
            source = None
            if isinstance(item, tuple) and len(item) == 2:
                item, source = item
            block = item
            input_ids = block if torch.is_tensor(block) else torch.tensor(block, dtype=torch.long)
            input_ids = input_ids.to(device)
            if input_ids.numel() < 2:
                continue
            prefix_length = min(max(1, int(context_tokens)), int(input_ids.numel()) - 1)
            reference_ids = input_ids[prefix_length : prefix_length + int(max_new_tokens)]
            prompt = tokenizer.decode(input_ids[:prefix_length], skip_special_tokens=True).strip()
            reference = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()
            hypothesis, generated_tokens = generate_text_continuation(
                model,
                tokenizer,
                input_ids[:prefix_length],
                attention_mask=torch.ones(prefix_length, dtype=torch.long, device=device),
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    "example_index": index,
                    "source": str(source) if source is not None else "unknown",
                    "prompt": prompt,
                    "reference_continuation": reference,
                    "hypothesis_continuation": hypothesis,
                    "prompt_tokens": prefix_length,
                    "reference_tokens": int(reference_ids.numel()),
                    "generated_tokens": generated_tokens,
                }
            )
    finally:
        if was_training:
            model.train()
    return rows
