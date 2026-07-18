"""Environment verification for local, Colab, and server launches."""

from dataclasses import asdict, dataclass
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile

from cot_enem.runtime.context import ExecutionContext
from cot_enem.runtime.environment import available_disk_gb


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    ok: bool
    detail: str


def verify_environment(context: ExecutionContext) -> list[VerificationCheck]:
    config = context.loaded_config.config
    checks = [
        VerificationCheck(
            "python",
            (3, 11) <= sys.version_info[:2] < (3, 14),
            f"Python {context.environment.python_version}; expected >=3.11,<3.14",
        )
    ]
    for package in ("pydantic", "PyYAML"):
        try:
            version = importlib.metadata.version(package)
            checks.append(VerificationCheck(f"dependency:{package}", True, version))
        except importlib.metadata.PackageNotFoundError:
            checks.append(VerificationCheck(f"dependency:{package}", False, "not installed"))
    output = context.output_directory
    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output, prefix=".verify-", delete=True):
            pass
        checks.append(VerificationCheck("output_writable", True, str(output.resolve())))
    except OSError as exc:
        checks.append(VerificationCheck("output_writable", False, str(exc)))
    free_disk = available_disk_gb(output)
    checks.append(
        VerificationCheck("output_disk_space", free_disk >= 0.1, f"{free_disk:.1f} GB free")
    )
    if config.model.provider == "huggingface" and config.storage.cache_path:
        cache = Path(config.storage.cache_path)
        try:
            cache.mkdir(parents=True, exist_ok=True)
            cache_free = available_disk_gb(cache)
            checks.append(
                VerificationCheck(
                    "model_cache_disk_space",
                    cache_free >= 45,
                    f"{cache_free:.1f} GB free; expected at least 45 GB for three models",
                )
            )
        except OSError as exc:
            checks.append(VerificationCheck("model_cache_disk_space", False, str(exc)))
    if config.model.provider == "openai_compatible":
        for variable in ("LLM_BASE_URL", "LLM_MODEL"):
            checks.append(
                VerificationCheck(
                    f"environment:{variable}",
                    bool(os.getenv(variable)),
                    "configured" if os.getenv(variable) else "missing",
                )
            )
    if config.model.provider == "huggingface":
        packages = ["transformers", "torch", "accelerate"]
        if config.model.quantization != "none":
            packages.append("bitsandbytes")
        for package in packages:
            try:
                version = importlib.metadata.version(package)
                checks.append(VerificationCheck(f"dependency:{package}", True, version))
            except importlib.metadata.PackageNotFoundError:
                checks.append(VerificationCheck(f"dependency:{package}", False, "not installed"))
    checks.append(
        VerificationCheck(
            "device",
            True,
            f"{context.device.device}/{context.device.precision}: {context.device.reason}",
        )
    )
    return checks


def environment_summary(context: ExecutionContext) -> dict[str, object]:
    return {
        "environment": asdict(context.environment),
        "device": asdict(context.device),
        "configuration_files": [str(path) for path in context.loaded_config.files],
        "output_directory": str(context.output_directory),
        "model": context.loaded_config.config.model.name,
    }
