"""YAML-backed configuration with explicit prompt versions."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE entries without adding a runtime dependency.

    Existing process variables win by default, which lets CI, Colab and IDE run
    configurations override the local baseline safely.
    """

    source = Path(path)
    if not source.exists():
        return {}
    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid environment entry at {source}:{line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment key at {source}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    version: str
    system: str
    user: str


class PromptCatalog:
    def __init__(self, prompts: dict[str, PromptDefinition]) -> None:
        self._prompts = prompts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptCatalog":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream) or {}
        prompts = {
            name: PromptDefinition(
                version=str(value["version"]),
                system=str(value["system"]),
                user=str(value["user"]),
            )
            for name, value in data.get("prompts", {}).items()
        }
        return cls(prompts)

    def get(self, name: str) -> PromptDefinition:
        try:
            return self._prompts[name]
        except KeyError as exc:
            raise KeyError(f"prompt not configured: {name}") from exc
