from pathlib import Path

import pytest

from cot_enem.configuration.loader import ConfigurationError, load_application_config
from cot_enem.runtime.device import DeviceSelectionError, select_device
from cot_enem.runtime.environment import ExecutionEnvironment, detect_environment


def write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def environment(**updates) -> ExecutionEnvironment:
    values = {
        "platform": "linux",
        "environment": "linux",
        "is_wsl": False,
        "is_colab": False,
        "is_slurm": False,
        "cuda_available": False,
        "mps_available": False,
        "gpu_count": 0,
        "gpu_names": (),
        "total_memory_gb": 32.0,
        "available_memory_gb": 20.0,
        "temporary_directory": "/tmp",
        "persistent_directory": None,
        "hostname": "test",
        "python_version": "3.12.0",
        "slurm_job_id": None,
        "slurm_array_task_id": None,
    }
    values.update(updates)
    return ExecutionEnvironment(**values)


def test_configuration_precedence(tmp_path):
    default = write_yaml(
        tmp_path / "default.yaml",
        "runtime:\n  device: auto\nmodel:\n  name: default\nstorage:\n  base_path: out\n",
    )
    profile = write_yaml(
        tmp_path / "server.yaml",
        "runtime:\n  device: cpu\nmodel:\n  name: profile\n",
    )
    loaded = load_application_config(
        default_path=default,
        profile_path=profile,
        environ={"COTENEM_MODEL": "environment"},
        cli_overrides={"runtime.device": "cuda", "model.name": "cli"},
    )
    assert loaded.config.runtime.device == "cuda"
    assert loaded.config.model.name == "cli"
    assert loaded.config.storage.base_path == "out"


def test_configuration_error_names_source_and_field(tmp_path):
    source = write_yaml(tmp_path / "bad.yaml", "runtime:\n  device: quantum\n")
    with pytest.raises(ConfigurationError, match=r"bad.yaml.*runtime.device.*quantum"):
        load_application_config(default_path=source, environ={})


def test_device_auto_falls_back_to_cpu():
    selected = select_device("auto", "auto", environment())
    assert selected.device == "cpu"
    assert selected.precision == "fp32"


def test_explicit_unavailable_cuda_is_rejected():
    with pytest.raises(DeviceSelectionError, match="CUDA is not available"):
        select_device("cuda", "auto", environment())


def test_environment_detects_slurm(monkeypatch):
    monkeypatch.setattr("cot_enem.runtime.environment.platform.system", lambda: "Linux")
    monkeypatch.setattr("cot_enem.runtime.environment.platform.release", lambda: "generic")
    detected = detect_environment({"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "4"})
    assert detected.environment == "slurm"
    assert detected.slurm_job_id == "123"
    assert detected.slurm_array_task_id == "4"
