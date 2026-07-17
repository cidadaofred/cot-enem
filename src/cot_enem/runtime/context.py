"""Resolved execution context passed to infrastructure-aware orchestration."""

from dataclasses import dataclass, replace
from pathlib import Path

from cot_enem.configuration.loader import LoadedConfig
from cot_enem.runtime.device import DeviceSelection, select_device
from cot_enem.runtime.environment import ExecutionEnvironment, detect_environment


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    loaded_config: LoadedConfig
    environment: ExecutionEnvironment
    device: DeviceSelection
    output_directory: Path


def build_execution_context(loaded: LoadedConfig) -> ExecutionContext:
    detected = detect_environment()
    config = loaded.config
    if config.runtime.environment != "auto":
        detected = replace(detected, environment=config.runtime.environment)
    selected = select_device(config.runtime.device, config.runtime.precision, detected)
    output = Path(config.storage.persistent_path or config.storage.base_path)
    return ExecutionContext(
        loaded_config=loaded,
        environment=detected,
        device=selected,
        output_directory=output,
    )
