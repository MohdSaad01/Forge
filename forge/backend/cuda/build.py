"""Compiles `kernels.cu` into a loadable shared library via `nvcc`.

This is the only place Forge invokes a compiler. It is called lazily, only
when a CUDA device is actually requested (see `forge/backend/cuda/backend.py`
and `forge/backend/__init__.py`) -- a CPU-only environment never needs
`nvcc` on PATH and never pays this cost.

On Windows, `nvcc` delegates host-side compilation to MSVC's `cl.exe`, which
is not normally on PATH outside a "Developer Command Prompt". `_find_msvc_bin`
locates it via `vswhere.exe` (installed alongside every modern Visual Studio)
so the build works from an ordinary shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ...exceptions import CUDAError

_HERE = Path(__file__).parent
_SOURCE = _HERE / "kernels.cu"
_ARCH = "sm_50"  # Compute Capability 5.0 -- the verified development GPU (940MX).
_LIBRARY_PATH = _HERE / f"_forge_cuda_kernels_{_ARCH}.dll"


def _find_msvc_bin() -> "Path | None":
    """Locate MSVC's `Hostx64/x64` compiler directory via `vswhere`, if present."""
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    try:
        output = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not output:
        return None

    msvc_root = Path(output) / "VC" / "Tools" / "MSVC"
    if not msvc_root.is_dir():
        return None
    versions = sorted((p for p in msvc_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not versions:
        return None
    bin_dir = versions[-1] / "bin" / "Hostx64" / "x64"
    return bin_dir if bin_dir.is_dir() else None


def _compile(env: "dict[str, str] | None") -> subprocess.CompletedProcess:
    cmd = [
        "nvcc",
        "-O3",
        f"-arch={_ARCH}",
        "-std=c++17",
        "--cudart", "static",
        "-shared",
        str(_SOURCE),
        "-o", str(_LIBRARY_PATH),
    ]
    return subprocess.run(cmd, cwd=str(_HERE), env=env, capture_output=True, text=True)


def is_nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


def ensure_kernel_library() -> Path:
    """Return the path to the compiled kernel library, building it if needed.

    Raises `CUDAError` (never a bare `subprocess`/`OSError`) if `nvcc` is
    missing or compilation fails for any reason -- this is Forge's "CUDA
    initialization failure" case.
    """
    if not is_nvcc_available():
        raise CUDAError(
            "nvcc (the CUDA compiler) was not found on PATH. The Forge CUDA backend "
            "compiles its kernels at first use and cannot proceed without it; install "
            "the CUDA Toolkit or add its bin directory to PATH."
        )

    if _LIBRARY_PATH.exists() and _LIBRARY_PATH.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return _LIBRARY_PATH

    result = _compile(env=None)
    if result.returncode != 0:
        # nvcc on Windows needs MSVC's cl.exe as its host compiler, which is
        # frequently not already on PATH outside a Developer Command Prompt.
        # Retry once with a located MSVC bin directory prepended.
        msvc_bin = _find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = _compile(env=env)

    if result.returncode != 0:
        raise CUDAError(
            "Failed to compile the Forge CUDA backend (nvcc exited with code "
            f"{result.returncode}):\n{result.stderr or result.stdout}"
        )
    if not _LIBRARY_PATH.exists():
        raise CUDAError(
            "nvcc reported success but the expected CUDA kernel library was not produced "
            f"at '{_LIBRARY_PATH}'."
        )
    return _LIBRARY_PATH


__all__ = ["ensure_kernel_library", "is_nvcc_available"]
