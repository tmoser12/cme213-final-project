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
from pathlib import Path

from torch.utils.cpp_extension import load


def _strip_nvhpc_from_path() -> None:
    """
    Remove any PATH entry containing 'nvidia-hpc-sdk' or '/nvhpc/' so the
    extension build picks the stock CUDA nvcc instead of NVHPC's.

    Why: NVHPC's nvcc only supports `nvc++` as its host compiler, which is
    ABI-incompatible with the PyTorch wheel (built with g++/libstdc++ ABI 0).
    The cluster's `course/cme213/nvhpc/24.1` module prepends NVHPC to PATH and
    triggers an unrecoverable "Unsupported NVHPC compiler" error from nvcc.
    Stripping those entries here is idempotent and harmless when the module
    isn't loaded.
    """
    kept = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if "nvidia-hpc-sdk" not in entry and "/nvhpc/" not in entry
    ]
    os.environ["PATH"] = os.pathsep.join(kept)


def load_embedding_ops():
    """Compile (if needed) and return the embedding extension module."""
    _strip_nvhpc_from_path()

    here = Path(__file__).resolve().parent
    return load(
        name="qwen_embedding_kernel",
        sources=[str(here / "kernel.cu"), str(here / "bindings.cpp")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
        extra_cflags=["-O3"],
        verbose=True,
    )
