from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIO_GROUP = "audio"
LLM_GROUP = "llm"
_GROUPS = {AUDIO_GROUP, LLM_GROUP}
_AUDIO_LAYER_RE = re.compile(r"^audio_tower\.layers\.(\d+)\.(.+)$")


def patch_outer_forward(model: Any) -> None:
    cls = model.__class__
    if getattr(cls, "_tamil_asr_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Qwen3-ASR model has no thinker.forward; incompatible qwen_asr checkout")

    def forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        input_features: Any = None,
        feature_attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._tamil_asr_forward_patched = True


def _load_qwen_wrapper(model_cfg: Mapping[str, Any]) -> Any:
    if model_cfg.get("dtype") != "bfloat16":
        raise ValueError("The validated Qwen-ASR recipes require model.dtype=bfloat16")
    repo = model_cfg.get("qwen_asr_repo")
    if repo:
        repo_path = str(Path(repo).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Install the local Qwen-ASR checkout (`pip install -e /path/to/QwenAsr`) before training"
        ) from exc
    return Qwen3ASRModel.from_pretrained(
        model_cfg["path"],
        dtype=torch.bfloat16,
        device_map=None,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )


def _matches_suffix(name: str, suffix: str) -> bool:
    suffix = suffix.lstrip(".")
    return name == suffix or name.endswith(f".{suffix}")


def _require_linear(module: Any, name: str) -> None:
    if module.__class__.__name__ != "Linear":
        raise TypeError(f"LoRA target {name} is {module.__class__.__name__}, expected Linear")


def _select_text_decoder_modules(thinker: Any, suffixes: Sequence[str]) -> list[str]:
    selected = []
    for name, module in thinker.named_modules():
        if ".layers." not in name or not name.startswith("model.layers."):
            continue
        if any(_matches_suffix(name, suffix) for suffix in suffixes):
            _require_linear(module, name)
            selected.append(name)
    if not selected:
        raise RuntimeError(
            "No text-decoder LoRA modules matched. Refusing to fall back to broad names that could modify the audio tower."
        )
    if any(name.startswith("audio_tower.") for name in selected):
        raise AssertionError("Audio-tower module selected for decoder-only LoRA")
    return selected


def _select_audio_modules(thinker: Any, audio_cfg: Mapping[str, Any]) -> list[str]:
    if not bool(audio_cfg.get("enabled", True)):
        return []

    layer_names = list(audio_cfg.get("target_modules", ()))
    projection_names = list(audio_cfg.get("projection_modules", ("conv_out", "proj1", "proj2")))
    selection = str(audio_cfg.get("layer_selection", "all"))
    layer_numbers = sorted(
        {
            int(match.group(1))
            for name in dict(thinker.named_modules())
            if (match := _AUDIO_LAYER_RE.match(name))
        }
    )
    if not layer_numbers:
        raise RuntimeError("No audio encoder layers were found under thinker.audio_tower.layers")
    if selection == "last_n":
        last_n = int(audio_cfg.get("last_n_layers", 0))
        if last_n <= 0:
            raise ValueError("model.audio_lora.last_n_layers must be positive for layer_selection=last_n")
        selected_layers = set(layer_numbers[-last_n:])
    elif selection == "all":
        selected_layers = set(layer_numbers)
    else:
        raise ValueError("model.audio_lora.layer_selection must be 'last_n' or 'all'")

    selected: list[str] = []
    for name, module in thinker.named_modules():
        match = _AUDIO_LAYER_RE.match(name)
        if match and int(match.group(1)) in selected_layers:
            if any(_matches_suffix(match.group(2), suffix) for suffix in layer_names):
                _require_linear(module, name)
                selected.append(name)
    for projection in projection_names:
        name = projection if projection.startswith("audio_tower.") else f"audio_tower.{projection}"
        module = dict(thinker.named_modules()).get(name)
        if module is None:
            raise RuntimeError(f"Configured audio projection module was not found: {name}")
        _require_linear(module, name)
        selected.append(name)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError("No audio LoRA modules matched the configured acoustic target contract")
    if any(not name.startswith("audio_tower.") for name in selected):
        raise AssertionError("Non-audio module selected for acoustic LoRA")
    return selected


def _adapter_target_from_parameter_name(name: str) -> str | None:
    relative = name.removeprefix("module.").removeprefix("thinker.")
    for marker in (".lora_A.", ".lora_B."):
        if marker in relative:
            target = relative.split(marker, maxsplit=1)[0]
            return target.removeprefix("base_model.model.")
    return None


def _adapter_name_from_parameter_name(name: str) -> str | None:
    relative = name.removeprefix("module.").removeprefix("thinker.")
    for marker in (".lora_A.", ".lora_B."):
        if marker in relative:
            suffix = relative.split(marker, maxsplit=1)[1]
            return suffix.split(".", maxsplit=1)[0]
    return None


def _group_for_target(target: str) -> str:
    if target.startswith("audio_tower."):
        return AUDIO_GROUP
    if target.startswith("model.layers."):
        return LLM_GROUP
    raise ValueError(f"LoRA target is outside the supported audio/LLM groups: {target}")


def _set_active_lora_groups(
    model: Any,
    target_groups: Mapping[str, str],
    active_groups: Sequence[str],
    trainable_adapter_names: Sequence[str] | None = None,
) -> None:
    active = set(active_groups)
    invalid = active.difference(_GROUPS)
    if invalid:
        raise ValueError(f"Unknown LoRA trainable groups: {sorted(invalid)}")
    if not active:
        raise ValueError("At least one LoRA trainable group is required")

    allowed_adapters = set(trainable_adapter_names) if trainable_adapter_names is not None else None

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        target = _adapter_target_from_parameter_name(name)
        adapter_name = _adapter_name_from_parameter_name(name)
        if (
            target is not None
            and target_groups.get(target) in active
            and (allowed_adapters is None or adapter_name in allowed_adapters)
        ):
            parameter.requires_grad = True

    missing = [
        group
        for group in active
        if not any(
            parameter.requires_grad
            and target_groups.get(target) == group
            and (allowed_adapters is None or _adapter_name_from_parameter_name(name) in allowed_adapters)
            for name, parameter in model.named_parameters()
            if (target := _adapter_target_from_parameter_name(name)) is not None
        )
    ]
    if missing:
        raise AssertionError(f"Active LoRA groups have no trainable parameters: {missing}")


def lora_parameter_groups(model: Any) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {AUDIO_GROUP: [], LLM_GROUP: []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = _adapter_target_from_parameter_name(name)
        if target is None:
            raise AssertionError(f"Non-LoRA parameter is trainable: {name}")
        groups[_group_for_target(target)].append(parameter)
    if not any(groups.values()):
        raise AssertionError("No trainable LoRA parameters found")
    return groups


def _validate_existing_lora_config(adapter_path: str | Path, lora_cfg: Mapping[str, Any]) -> None:
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Existing LoRA adapter is missing adapter_config.json: {adapter_config_path}"
        )
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if str(adapter_config.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError("Only LoRA adapters are supported for existing-adapter initialization")
    expected_contract = {
        "r": int(lora_cfg["rank"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
    }
    for key, expected in expected_contract.items():
        if key not in adapter_config:
            raise ValueError(f"Existing adapter config is missing required field: {key}")
        actual = float(adapter_config[key])
        if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"Existing adapter {key}={actual} does not match configured value {expected}"
            )


def _adapter_targets(model: Any) -> set[str]:
    return {
        target
        for name, parameter in model.named_parameters()
        if (target := _adapter_target_from_parameter_name(name)) is not None
    }


def load_model_for_lora(
    model_cfg: Mapping[str, Any],
    adapter_path: str | Path | None = None,
    trainable_groups: Sequence[str] | None = None,
    *,
    adapter_groups: Sequence[str] | None = None,
    merge_adapter_paths: Sequence[str | Path] = (),
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for decoder LoRA") from exc

    wrapper = _load_qwen_wrapper(model_cfg)
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)

    for parameter in model.parameters():
        parameter.requires_grad = False
    merged_adapter_paths = [Path(path) for path in merge_adapter_paths]
    for merged_path in merged_adapter_paths:
        if not merged_path.is_dir():
            raise FileNotFoundError(f"LoRA lineage adapter does not exist: {merged_path}")
        model.thinker = PeftModel.from_pretrained(
            model.thinker,
            str(merged_path),
            is_trainable=False,
        ).merge_and_unload()

    lora_cfg = model_cfg["lora"]
    requested_adapter_groups = set(adapter_groups or trainable_groups or (LLM_GROUP,))
    invalid_adapter_groups = requested_adapter_groups.difference(_GROUPS)
    if invalid_adapter_groups:
        raise ValueError(f"Unknown LoRA adapter groups: {sorted(invalid_adapter_groups)}")
    llm_targets = _select_text_decoder_modules(
        model.thinker, list(lora_cfg["target_modules"])
    ) if LLM_GROUP in requested_adapter_groups else []
    audio_targets = _select_audio_modules(
        model.thinker, model_cfg.get("audio_lora", {"enabled": False})
    ) if AUDIO_GROUP in requested_adapter_groups else []
    targets = [*audio_targets, *llm_targets]
    target_groups = {
        **{target: AUDIO_GROUP for target in audio_targets},
        **{target: LLM_GROUP for target in llm_targets},
    }
    active = list(trainable_groups) if trainable_groups is not None else [LLM_GROUP]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=targets,
        bias="none",
    )

    frozen_adapter_path = model_cfg.get("frozen_adapter_path")
    adapter_mode = "fresh_lora"
    trainable_adapter_names: list[str] | None = None
    if frozen_adapter_path:
        frozen_path = Path(frozen_adapter_path)
        if not frozen_path.is_dir():
            raise FileNotFoundError(f"Configured frozen_adapter_path does not exist: {frozen_path}")
        if audio_targets:
            raise ValueError(
                "frozen_adapter_path currently supports decoder-only residual LoRA; "
                "the frozen adapter must not include audio LoRA"
            )
        _validate_existing_lora_config(frozen_path, lora_cfg)
        model.thinker = PeftMixedModel.from_pretrained(
            model.thinker,
            str(frozen_path),
            adapter_name="frozen_base",
            is_trainable=False,
        )
        if adapter_path is None:
            model.thinker.add_adapter("asr_residual", peft_config)
        else:
            _validate_existing_lora_config(adapter_path, lora_cfg)
            model.thinker.load_adapter(
                str(adapter_path),
                adapter_name="asr_residual",
                is_trainable=True,
            )
        model.thinker.set_adapter(["frozen_base", "asr_residual"], inference_mode=False)
        trainable_adapter_names = ["asr_residual"]
        adapter_mode = (
            "frozen_adapter_plus_fresh_residual_lora"
            if adapter_path is None
            else "frozen_adapter_plus_residual_lora_resume"
        )
    else:
        if adapter_path is None:
            model.thinker = get_peft_model(model.thinker, peft_config)
        else:
            _validate_existing_lora_config(adapter_path, lora_cfg)
            model.thinker = PeftModel.from_pretrained(
                model.thinker, str(adapter_path), is_trainable=True
            )
            adapter_mode = "existing_trainable_lora"
    if hasattr(model.thinker, "get_base_model"):
        base_thinker = model.thinker.get_base_model()
    else:
        mixed_base = getattr(model.thinker, "base_model", None)
        base_thinker = getattr(mixed_base, "model", None)
        if base_thinker is None:
            raise RuntimeError("Could not unwrap the Qwen thinker from the PEFT model")
    text_model = getattr(base_thinker, "model", None)
    if model_cfg.get("gradient_checkpointing", True):
        for component in (text_model, getattr(base_thinker, "audio_tower", None)):
            if component is not None and hasattr(component, "gradient_checkpointing_enable"):
                component.gradient_checkpointing_enable()
    if hasattr(base_thinker, "enable_input_require_grads"):
        base_thinker.enable_input_require_grads()
    base_thinker.config.use_cache = False

    loaded_targets = _adapter_targets(model)
    if loaded_targets != set(targets):
        raise ValueError("Loaded adapter modules differ from the configured audio/LLM LoRA contract")
    _set_active_lora_groups(model, target_groups, active, trainable_adapter_names)
    groups = lora_parameter_groups(model)
    inactive_groups = _GROUPS.difference(active)
    unexpectedly_trainable = [group for group in inactive_groups if groups[group]]
    if unexpectedly_trainable:
        raise AssertionError(f"Inactive LoRA groups became trainable: {unexpectedly_trainable}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "lora_target_count": len(targets),
        "lora_targets": targets,
        "groups": {
            AUDIO_GROUP: {"targets": audio_targets, "parameters": sum(
                parameter.numel() for parameter in groups[AUDIO_GROUP]
            )},
            LLM_GROUP: {"targets": llm_targets, "parameters": sum(
                parameter.numel() for parameter in groups[LLM_GROUP]
            )},
        },
        "active_trainable_groups": sorted(active),
        "adapter_mode": adapter_mode,
        "trainable_adapter_names": trainable_adapter_names,
        "frozen_adapter_path": str(frozen_adapter_path) if frozen_adapter_path else None,
        "trainable_adapter_path": str(adapter_path) if adapter_path else None,
        "merged_adapter_lineage": [str(path) for path in merged_adapter_paths],
    }
    return model, processor, report


def save_adapter_and_processor(model: Any, processor: Any, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    thinker = getattr(model, "thinker", None)
    if thinker is None or not hasattr(thinker, "save_pretrained"):
        raise TypeError("Expected a PEFT-wrapped model.thinker")
    adapter_kwargs: dict[str, Any] = {}
    configured_adapters = getattr(thinker, "peft_config", {})
    if "asr_residual" in configured_adapters:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        adapter_dir = target / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        configured_adapters["asr_residual"].save_pretrained(adapter_dir)
        state = get_peft_model_state_dict(thinker, adapter_name="asr_residual")
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in state.items()},
            str(adapter_dir / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        thinker.save_pretrained(target / "adapter", safe_serialization=True, **adapter_kwargs)
    processor.save_pretrained(target / "processor")
