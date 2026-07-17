"""Environment inspection isolated from domain and pipeline code."""

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import tempfile


@dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    platform: str
    environment: str
    is_wsl: bool
    is_colab: bool
    is_slurm: bool
    cuda_available: bool
    mps_available: bool
    gpu_count: int
    gpu_names: tuple[str, ...]
    total_memory_gb: float | None
    available_memory_gb: float | None
    temporary_directory: str
    persistent_directory: str | None
    hostname: str
    python_version: str
    slurm_job_id: str | None
    slurm_array_task_id: str | None


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total_physical / 2**30, status.available_physical / 2**30
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        return total / 2**30, available / 2**30
    except (AttributeError, OSError, ValueError):
        return None, None


def _torch_capabilities() -> tuple[bool, bool, int, tuple[str, ...]]:
    try:
        import torch
    except ImportError:
        return False, False, 0, ()
    cuda = bool(torch.cuda.is_available())
    mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    count = int(torch.cuda.device_count()) if cuda else 0
    names = tuple(torch.cuda.get_device_name(index) for index in range(count))
    return cuda, mps, count, names


def detect_environment(environ: dict[str, str] | None = None) -> ExecutionEnvironment:
    environment_variables = environ if environ is not None else os.environ
    system = platform.system().lower()
    release = platform.release().lower()
    proc_version = ""
    if Path("/proc/version").exists():
        try:
            proc_version = Path("/proc/version").read_text(encoding="utf-8").lower()
        except OSError:
            pass
    is_wsl = system == "linux" and ("microsoft" in release or "microsoft" in proc_version)
    is_colab = bool(
        environment_variables.get("COLAB_RELEASE_TAG")
        or environment_variables.get("COLAB_BACKEND_VERSION")
    )
    is_slurm = "SLURM_JOB_ID" in environment_variables
    if is_colab:
        detected = "colab"
    elif is_slurm:
        detected = "slurm"
    elif is_wsl:
        detected = "wsl"
    elif system == "windows":
        detected = "windows"
    elif system == "linux":
        detected = "linux"
    else:
        detected = system
    cuda, mps, gpu_count, gpu_names = _torch_capabilities()
    total_memory, available_memory = _memory_gb()
    persistent = (
        environment_variables.get("COTENEM_PERSISTENT_DIR")
        or environment_variables.get("SLURM_SUBMIT_DIR")
    )
    return ExecutionEnvironment(
        platform=system,
        environment=detected,
        is_wsl=is_wsl,
        is_colab=is_colab,
        is_slurm=is_slurm,
        cuda_available=cuda,
        mps_available=mps,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        total_memory_gb=total_memory,
        available_memory_gb=available_memory,
        temporary_directory=tempfile.gettempdir(),
        persistent_directory=persistent,
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        slurm_job_id=environment_variables.get("SLURM_JOB_ID"),
        slurm_array_task_id=environment_variables.get("SLURM_ARRAY_TASK_ID"),
    )


def available_disk_gb(path: str | Path) -> float:
    target = Path(path)
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 2**30
