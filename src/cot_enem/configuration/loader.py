"""Configuration precedence: defaults, profile, environment, CLI."""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from cot_enem.configuration.schemas import ApplicationConfig

ENVIRONMENT_OVERRIDES: dict[str, tuple[str, ...]] = {
    "COTENEM_ENVIRONMENT": ("runtime", "environment"),
    "COTENEM_DEVICE": ("runtime", "device"),
    "COTENEM_PRECISION": ("runtime", "precision"),
    "COTENEM_NUM_WORKERS": ("runtime", "num_workers"),
    "COTENEM_OUTPUT_DIR": ("storage", "base_path"),
    "COTENEM_PERSISTENT_DIR": ("storage", "persistent_path"),
    "COTENEM_CACHE_DIR": ("storage", "cache_path"),
    "COTENEM_MODEL": ("model", "name"),
    "COTENEM_JUDGE_MODEL": ("model", "judge_name"),
    "COTENEM_LOG_LEVEL": ("logging", "level"),
    "LLM_MODEL": ("model", "name"),
    "JUDGE_MODEL": ("model", "judge_name"),
}


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: ApplicationConfig
    files: tuple[Path, ...]


class ConfigurationError(ValueError):
    """Configuration failed validation with source context."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _coerce_environment_value(value: str) -> str | int | float | bool | None:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def load_application_config(
    *,
    default_path: str | Path = "configs/default.yaml",
    profile_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> LoadedConfig:
    """Resolve and validate configuration according to documented precedence."""

    default_file = Path(default_path)
    files = [default_file]
    resolved = _read_yaml(default_file)
    if profile_path:
        profile_file = Path(profile_path)
        resolved = _deep_merge(resolved, _read_yaml(profile_file))
        files.append(profile_file)
    environment = environ if environ is not None else os.environ
    for variable, field_path in ENVIRONMENT_OVERRIDES.items():
        if variable in environment and environment[variable] != "":
            _set_nested(
                resolved,
                field_path,
                _coerce_environment_value(environment[variable]),
            )
    for dotted_key, value in (cli_overrides or {}).items():
        if value is not None:
            _set_nested(resolved, tuple(dotted_key.split(".")), value)
    try:
        config = ApplicationConfig.model_validate(resolved)
    except ValidationError as exc:
        sources = ", ".join(str(path) for path in files)
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: "
            f"value={error.get('input')!r}, expected={error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"invalid configuration ({sources}): {details}") from exc
    return LoadedConfig(config=config, files=tuple(files))
