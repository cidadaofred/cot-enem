"""Portable runtime detection and execution context."""

from cot_enem.runtime.context import ExecutionContext, build_execution_context
from cot_enem.runtime.device import DeviceSelection, select_device
from cot_enem.runtime.environment import ExecutionEnvironment, detect_environment

__all__ = [
    "DeviceSelection",
    "ExecutionContext",
    "ExecutionEnvironment",
    "build_execution_context",
    "detect_environment",
    "select_device",
]
