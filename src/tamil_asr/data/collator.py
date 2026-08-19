from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any


def _build_prefix_messages(prompt: str, audio_array: Any) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


@dataclass
class QwenASRCollator:
    processor: Any
    prompt: str
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token

    def _prefix_text(self) -> str:
        messages = _build_prefix_messages(self.prompt, None)
        rendered = self.processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
        if not isinstance(rendered, list) or len(rendered) != 1 or not isinstance(rendered[0], str):
            raise TypeError("Qwen processor returned an unexpected chat-template representation")
        return rendered[0]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("numpy and torch are required for Qwen ASR collation") from exc

        prefix = self._prefix_text()
        audios: list[Any] = []
        targets: list[str] = []
        metadata: list[dict[str, Any]] = []
        for item in features:
            waveform = np.asarray(item["audio"], dtype=np.float32)
            if waveform.ndim != 1 or waveform.size == 0:
                raise ValueError(f"{item.get('id')}: expected non-empty mono waveform, got {waveform.shape}")
            if not np.isfinite(waveform).all():
                raise ValueError(f"{item.get('id')}: waveform contains non-finite samples")
            if int(item["sample_rate"]) != self.sample_rate:
                raise ValueError(f"{item.get('id')}: collator received an unexpected sample rate")
            target = str(item["text"]).strip()
            if not target:
                raise ValueError(f"{item.get('id')}: empty target reached the collator")
            if "<|" in target or "|>" in target:
                raise ValueError(f"{item.get('id')}: target contains a model control-token fragment")
            audios.append(waveform)
            targets.append(target)
            metadata.append(
                {
                    "id": item.get("id", ""),
                    "dataset": item.get("dataset", ""),
                    "source_split": item.get("source_split", ""),
                    "domain": item.get("domain", ""),
                    "speaker_id": item.get("speaker_id", ""),
                    "duration": float(item.get("duration", waveform.size / self.sample_rate)),
                    "base_wer": item.get("base_wer"),
                    "scenario": item.get("scenario", ""),
                    "severity": item.get("severity"),
                    "reference": target,
                }
            )

        eos_id = self.processor.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Qwen tokenizer must define an EOS token")
        prefix_texts = [prefix] * len(features)
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).to(torch.long)
        target_ids = [
            list(self.processor.tokenizer(target, add_special_tokens=False)["input_ids"]) + [int(eos_id)]
            for target in targets
        ]
        if any(not values for values in target_ids):
            raise ValueError("Tokenizer produced an empty transcript target")
        full_lengths = torch.tensor(
            [int(prefix_length) + len(target) for prefix_length, target in zip(prefix_lengths.tolist(), target_ids)],
            dtype=torch.long,
        )
        maximum_length = int(full_lengths.max().item())
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad token after collator initialization")
        input_ids = torch.full(
            (len(features), maximum_length), int(pad_id), dtype=prefix_inputs["input_ids"].dtype
        )
        attention_mask = torch.zeros((len(features), maximum_length), dtype=prefix_inputs["attention_mask"].dtype)
        labels = torch.full((len(features), maximum_length), -100, dtype=torch.long)
        for index, (prefix_length, target) in enumerate(zip(prefix_lengths.tolist(), target_ids)):
            prompt_ids = prefix_inputs["input_ids"][index, :prefix_length]
            target_tensor = torch.tensor(target, dtype=input_ids.dtype)
            input_ids[index, :prefix_length] = prompt_ids
            input_ids[index, prefix_length : prefix_length + len(target)] = target_tensor
            attention_mask[index, : prefix_length + len(target)] = 1
            labels[index, prefix_length : prefix_length + len(target)] = target_tensor

        supervised = (labels != -100).sum(dim=1)
        if torch.any(supervised <= 0):
            raise ValueError("At least one sample has zero supervised transcript tokens")
        if torch.any(labels[:, : int(prefix_lengths.min())] != -100):
            raise AssertionError("Prompt label leakage detected")
        for index, full_length in enumerate(full_lengths.tolist()):
            if eos_id is not None and int(labels[index, full_length - 1]) != int(eos_id):
                raise AssertionError(
                    f"{metadata[index]['id']}: EOS is not supervised; check pad/EOS masking"
                )

        prefix_inputs["input_ids"] = input_ids
        prefix_inputs["attention_mask"] = attention_mask
        prefix_inputs["labels"] = labels
        prefix_inputs["_metadata"] = metadata
        prefix_inputs["_prefix_lengths"] = prefix_lengths
        prefix_inputs["_full_lengths"] = full_lengths
        return prefix_inputs


def validate_batch_contract(model: Any, batch: dict[str, Any]) -> dict[str, int]:
    import torch

    required = {"input_ids", "attention_mask", "input_features", "feature_attention_mask", "labels"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"Qwen processor batch is missing required keys: {sorted(missing)}")

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    if input_ids.shape != labels.shape or input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"Text tensor mismatch: ids={tuple(input_ids.shape)} labels={tuple(labels.shape)} "
            f"mask={tuple(attention_mask.shape)}"
        )
    if torch.any(labels[attention_mask == 0] != -100):
        raise AssertionError("Padding tokens are supervised")

    thinker = getattr(model, "thinker", model)
    if hasattr(thinker, "get_base_model"):
        thinker = thinker.get_base_model()
    audio_token_id = getattr(getattr(thinker, "config", None), "audio_token_id", None)
    if audio_token_id is None:
        raise AttributeError("Cannot locate Qwen3-ASR audio_token_id")
    audio_positions = input_ids == int(audio_token_id)
    if not torch.any(audio_positions):
        raise ValueError("Processor produced no audio placeholder tokens")
    if torch.any(labels[audio_positions] != -100):
        raise AssertionError("Audio placeholder tokens are supervised")

    device_type = batch["input_features"].device.type
    autocast = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        features = thinker.get_audio_features(
            batch["input_features"],
            feature_attention_mask=batch["feature_attention_mask"],
        )
    hidden_size = int(features.shape[-1])
    placeholders = int(audio_positions.sum().item())
    feature_vectors = int(features.numel() // hidden_size)
    if placeholders != feature_vectors:
        raise ValueError(
            f"Audio placeholder/feature mismatch: placeholders={placeholders}, encoded_vectors={feature_vectors}, "
            f"feature_shape={tuple(features.shape)}"
        )
    return {
        "batch_size": int(input_ids.shape[0]),
        "text_positions": int(attention_mask.sum().item()),
        "audio_placeholders": placeholders,
        "supervised_tokens": int((labels != -100).sum().item()),
    }
