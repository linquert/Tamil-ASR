from __future__ import annotations

import math
import subprocess
from collections import defaultdict
from typing import Any


def gradient_diagnostics(model: Any) -> dict[str, float]:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    nonfinite = 0
    nonzero = 0
    parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        parameters += 1
        gradient = parameter.grad.detach()
        if not gradient.isfinite().all():
            nonfinite += 1
            continue
        norm = float(gradient.float().norm().item())
        nonzero += norm > 0
        family = "attention" if "self_attn" in name else "mlp" if ".mlp." in name else "other"
        groups[family].append(norm)
    metrics = {
        "grad/parameters_with_grad": float(parameters),
        "grad/nonzero_parameters": float(nonzero),
        "grad/nonfinite_parameters": float(nonfinite),
    }
    for family, values in groups.items():
        metrics[f"grad/{family}_rms_norm"] = math.sqrt(sum(value * value for value in values) / len(values))
    return metrics


def cuda_memory_metrics() -> dict[str, float]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    scale = 1024.0**3
    return {
        "system/gpu_allocated_gib": torch.cuda.memory_allocated() / scale,
        "system/gpu_reserved_gib": torch.cuda.memory_reserved() / scale,
        "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / scale,
    }


def gpu_activity_metrics() -> dict[str, float]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        gpu, memory, power = (float(value) for value in values[:3])
        return {
            "system/gpu_utilization_percent": gpu,
            "system/gpu_memory_utilization_percent": memory,
            "system/gpu_power_watts": power,
        }
    except (FileNotFoundError, IndexError, OSError, ValueError, subprocess.SubprocessError):
        return {}
