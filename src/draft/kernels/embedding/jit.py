"""
src/kernels/embedding/jit.py
JIT-compile and load the custom embedding CUDA extension.

First call compiles `kernel.cu` + `bindings.cpp` into a .so under
~/.cache/torch_extensions/.../qwen_embedding_kernel/. Subsequent calls reuse
the cached build, so the only cost is on first import.

If JIT picks up a stale build after editing the .cu, nuke the cache:
    rm -rf ~/.cache/torch_extensions/*/qwen_embedding_kernel
"""

import os
import shutil
from pathlib import Path


_NVHPC_MARKERS = ("nvidia-hpc-sdk", "/nvhpc/")


def _looks_like_nvhpc(value: str) -> bool:
    return any(marker in value for marker in _NVHPC_MARKERS)


def _sanitize_env_for_nvcc() -> None:
    """
    Scrub NVHPC paths from the build environment so torch.utils.cpp_extension
    picks the stock CUDA toolkit (cuda/12.2 module) + g++ (gnu12 module),
    not NVHPC's nvcc + nvc++.

    Why: the cluster's `course/cme213/nvhpc/24.1` module sets PATH plus
    CC=nvc, CXX=nvc++, CUDA_HOME=<nvhpc cuda>. NVHPC's nvcc rejects every
    host compiler except `nvc++`, and `nvc++` identifies as GCC 8.x which
    fails PyTorch 2.4's "GCC >= 9" version check. PATH-stripping alone is
    insufficient -- the explicit CC/CXX/CUDA_HOME env vars beat PATH-based
    discovery -- so we also unset them when they point into NVHPC.

    Idempotent: a noop when the course module isn't loaded.
    """
    # 1. Drop NVHPC entries from PATH so `which nvcc` / `which g++` resolve
    # to the stock toolchain.
    kept = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if not _looks_like_nvhpc(entry)
    ]
    os.environ["PATH"] = os.pathsep.join(kept)

    # 2. Unset compiler / CUDA env vars iff they point at NVHPC. After this,
    # torch falls back to PATH-based discovery, which now finds cuda/12.2's
    # nvcc and gnu12's g++.
    for var in ("CC", "CXX", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN"):
        val = os.environ.get(var)
        if val and _looks_like_nvhpc(val):
            del os.environ[var]

    # 3. Pin CUDA_HOME from the now-clean PATH so torch.utils.cpp_extension's
    # _find_cuda_home() doesn't fall back to /usr/local/cuda or anything else
    # unexpected. nvcc lives at $CUDA_HOME/bin/nvcc by convention.
    if "CUDA_HOME" not in os.environ:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            os.environ["CUDA_HOME"] = str(Path(nvcc_path).resolve().parent.parent)

    # 4. Pin target arch so we don't generate cubins we'll never use, and
    # silence torch's "TORCH_CUDA_ARCH_LIST not set" warning.
    # 7.5 = Quadro RTX 6000 (Turing).
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")


# CRITICAL: scrub the env BEFORE importing torch.utils.cpp_extension.
# That module caches CUDA_HOME at first import via _find_cuda_home(), and
# never re-reads it. If we sanitize later (e.g. inside load_embedding_ops),
# torch will already have locked in NVHPC's nvcc path.
_sanitize_env_for_nvcc()

from torch.utils.cpp_extension import load  # noqa: E402


def load_embedding_ops():
    """Compile (if needed) and return the embedding extension module."""
    # Re-run for safety: cheap, idempotent, catches anything that mutated
    # the env between import and call time.
    _sanitize_env_for_nvcc()

    here = Path(__file__).resolve().parent
    return load(
        name="qwen_embedding_kernel",
        sources=[str(here / "kernel.cu"), str(here / "bindings.cpp")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
        extra_cflags=["-O3"],
        verbose=True,
    )
