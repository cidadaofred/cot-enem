"""Centralized device and precision selection."""

from dataclasses import dataclass

from cot_enem.configuration.schemas import DeviceName, PrecisionName
from cot_enem.runtime.environment import ExecutionEnvironment


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    device: str
    precision: str
    reason: str


class DeviceSelectionError(ValueError):
    """Requested hardware or precision is unavailable."""


def _cuda_supports_bf16() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_bf16_supported())
    except (ImportError, RuntimeError):
        return False


def select_device(
    requested_device: DeviceName,
    requested_precision: PrecisionName,
    environment: ExecutionEnvironment,
) -> DeviceSelection:
    if requested_device == "auto":
        if environment.cuda_available:
            device, reason = "cuda", "CUDA detected"
        elif environment.mps_available:
            device, reason = "mps", "Apple MPS detected"
        else:
            device, reason = "cpu", "no supported accelerator detected"
    else:
        device, reason = requested_device, "explicit CLI/config selection"
    if device == "cuda" and not environment.cuda_available:
        raise DeviceSelectionError("device='cuda' requested, but CUDA is not available")
    if device == "mps" and not environment.mps_available:
        raise DeviceSelectionError("device='mps' requested, but MPS is not available")

    if requested_precision == "auto":
        if device == "cuda":
            precision = "bf16" if _cuda_supports_bf16() else "fp16"
        elif device == "mps":
            precision = "fp16"
        else:
            precision = "fp32"
    else:
        precision = requested_precision
    if device == "cpu" and precision == "fp16":
        raise DeviceSelectionError("precision='fp16' is not supported by the CPU runtime")
    if precision == "bf16" and device == "cuda" and not _cuda_supports_bf16():
        raise DeviceSelectionError("precision='bf16' requested, but the CUDA device lacks support")
    return DeviceSelection(device=device, precision=precision, reason=reason)
