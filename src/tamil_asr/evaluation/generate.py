from __future__ import annotations

from typing import Any, Sequence

from tamil_asr.data.collator import _build_prefix_messages


def _move_inputs_to_model_device(inputs: Any, device: Any) -> dict[str, Any]:
    import torch

    return {
        key: (
            value.to(device=device, dtype=torch.bfloat16)
            if torch.is_tensor(value) and value.is_floating_point()
            else value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }


def _generation_metadata(
    generated: Any,
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    token_ids = [int(token) for token in generated.tolist()]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = {int(token) for token in eos_token_id} if isinstance(eos_token_id, (list, tuple, set)) else ({int(eos_token_id)} if eos_token_id is not None else set())
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    completion_end = len(token_ids)
    stopped_on_eos = False
    for position, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            completion_end = position + 1
            stopped_on_eos = True
            break
        if pad_token_id is not None and token_id == int(pad_token_id):
            completion_end = position
            break
    completion = generated[:completion_end]
    return {
        "text": tokenizer.decode(completion, skip_special_tokens=True).strip(),
        "generated_tokens": int(completion.numel()),
        "hit_max_new_tokens": (
            not stopped_on_eos and int(completion.numel()) >= int(max_new_tokens)
        ),
    }


def transcribe_audio_batch(
    model: Any,
    processor: Any,
    audios: Sequence[Any],
    prompt: str,
    max_new_tokens: int,
    *,
    return_metadata: bool = False,
) -> list[str] | list[dict[str, Any]]:
    import torch

    if not audios:
        return []
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    model.eval()
    device = next(model.parameters()).device
    messages = _build_prefix_messages(prompt, None)
    rendered = processor.apply_chat_template(
        [messages] * len(audios), add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=rendered,
        audio=list(audios),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = _move_inputs_to_model_device(inputs, device)
    prompt_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True
        )
    sequences = getattr(result, "sequences", result)
    if int(sequences.shape[0]) != len(audios):
        raise RuntimeError(
            "Model returned a different number of sequences than input audios: "
            f"{int(sequences.shape[0])} != {len(audios)}"
        )
    metadata = [
        _generation_metadata(
            sequences[row_index, prompt_width:],
            tokenizer=processor.tokenizer,
            max_new_tokens=max_new_tokens,
        )
        for row_index in range(len(audios))
    ]
    if return_metadata:
        return metadata
    return [str(row["text"]) for row in metadata]


def transcribe_audio(
    model: Any,
    processor: Any,
    audio: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    return_metadata: bool = False,
) -> str | dict[str, Any]:
    generated = transcribe_audio_batch(
        model,
        processor,
        [audio],
        prompt,
        max_new_tokens,
        return_metadata=return_metadata,
    )
    return generated[0]


def generate_transcriptions(
    model: Any,
    processor: Any,
    dataset: Any,
    indices: Sequence[int],
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for index in indices:
        item = dataset[index]
        generation = transcribe_audio(
            model,
            processor,
            item["audio"],
            prompt,
            max_new_tokens,
            return_metadata=True,
        )
        assert isinstance(generation, dict)
        hypothesis = str(generation["text"])
        rows.append(
            {
                "id": item["id"],
                "dataset": item["dataset"],
                "source_split": item.get("source_split", ""),
                "domain": item["domain"],
                "speaker_id": item.get("speaker_id", ""),
                "duration": item["duration"],
                "base_wer": item.get("base_wer"),
                "scenario": item.get("scenario", ""),
                "severity": item.get("severity"),
                "reference": item["text"],
                "hypothesis": hypothesis,
                "generated_tokens": int(generation["generated_tokens"]),
                "hit_max_new_tokens": bool(generation["hit_max_new_tokens"]),
            }
        )
    model.train()
    return rows
