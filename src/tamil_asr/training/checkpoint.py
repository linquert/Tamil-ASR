from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

from tamil_asr.model.qwen3_asr import save_adapter_and_processor


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            match = _CHECKPOINT_RE.match(path.name)
            if path.is_dir() and match and (path / "state.json").exists():
                candidates.append((int(match.group(1)), path))
    latest = max(candidates, default=(0, None))[1]
    if latest is not None:
        return latest
    best = root / "best"
    return best if (best / "state.json").exists() else None


def validate_stage_state(
    state: dict[str, Any],
    expected_stage_name: str,
    expected_active_groups: list[str],
    *,
    accepted_stage_names: tuple[str, ...] = (),
) -> None:
    actual_stage = state.get("stage", state.get("stage_name"))
    actual_groups = sorted(state.get("active_trainable_groups", []))
    expected_groups = sorted(expected_active_groups)
    if actual_stage not in {expected_stage_name, *accepted_stage_names}:
        raise ValueError(
            f"Checkpoint stage mismatch: expected {expected_stage_name!r}, found {actual_stage!r}"
        )
    if actual_groups != expected_groups:
        raise ValueError(
            f"Checkpoint trainable groups mismatch: expected {expected_groups}, found {actual_groups}"
        )


def read_checkpoint_state(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    state_path = checkpoint / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint state is missing: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_checkpoint(
    accelerator: Any,
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: str | Path,
    step: int,
    epoch: int,
    metadata: dict[str, Any],
    keep_last: int,
    name: str | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = name or f"checkpoint-{step}"
    final = root / checkpoint_name
    temporary = root / f".{checkpoint_name}.tmp"
    if accelerator.is_main_process:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        import numpy as np
        import torch

        unwrapped = accelerator.unwrap_model(model)
        save_adapter_and_processor(unwrapped, processor, temporary)
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(training_state, temporary / "training_state.pt")
        state = {"global_step": step, "epoch": epoch, **metadata}
        state_path = temporary / "state.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        if name is None:
            checkpoints = sorted(
                (
                    (int(match.group(1)), path)
                    for path in root.iterdir()
                    if path.is_dir() and (match := _CHECKPOINT_RE.match(path.name))
                ),
                reverse=True,
            )
            for _, old in checkpoints[max(1, int(keep_last)) :]:
                shutil.rmtree(old)
    accelerator.wait_for_everyone()
    return final


def load_checkpoint(
    accelerator: Any,
    optimizer: Any,
    scheduler: Any,
    path: str | Path,
    *,
    restore_optimizer: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import torch

    checkpoint = Path(path)
    state = read_checkpoint_state(checkpoint)
    training_state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    if restore_optimizer:
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
    if restore_rng:
        random.setstate(training_state["python_rng"])
        np.random.set_state(training_state["numpy_rng"])
        torch.set_rng_state(training_state["torch_rng"])
        if torch.cuda.is_available() and training_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(training_state["cuda_rng"])
    return state
