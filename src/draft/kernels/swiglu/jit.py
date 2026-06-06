"""src/kernels/swiglu/jit.py
JIT-compile and load the custom SwiGLU CUDA extension.

First call compiles `kernel.cu` + `bindings.cpp` into a .so under
~/.cache/torch_extensions/.../draft_swiglu_ops/. Subsequent calls
reuse the cached build.

If JIT picks up a stale build after editing the .cu, nuke the cache:
    rm -rf ~/.cache/torch_extensions/*/draft_swiglu_ops
"""

import os
import shutil


def _sanitize_env_for_nvcc():
    # Strip NVHPC from PATH so torch.utils.cpp_extension picks stock cuda/12.2.
    paths = os.environ.get("PATH", "").split(":")
    paths = [p for p in paths if "nvidia-hpc-sdk" not in p and "/nvhpc/" not in p]
    os.environ["PATH"] = ":".join(paths)

    # Drop CC/CXX/CUDA_HOME if they point at NVHPC (NVHPC's nvc++ identifies as
    # GCC 8.x which fails PyTorch 2.4's "GCC >= 9" check).
    for var in ["CC", "CXX", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN"]:
        val = os.environ.get(var, "").lower()
        if "nvhpc" in val or "nvidia-hpc-sdk" in val or "nvc" in val:
            del os.environ[var]

    # Pin CUDA_HOME from the now-clean PATH.
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

    # Pin target arch: 7.5 = Quadro RTX 6000 (Turing).
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")


# CRITICAL: scrub the env BEFORE importing torch.utils.cpp_extension --
# that module caches CUDA_HOME at first import via _find_cuda_home().
_sanitize_env_for_nvcc()

from torch.utils.cpp_extension import load  # noqa: E402
from pathlib import Path  # noqa: E402


def load_swiglu_ops():
    """Compile (if needed) and return the SwiGLU extension module."""
    _sanitize_env_for_nvcc()
    d = Path(__file__).resolve().parent
    return load(
        name="draft_swiglu_ops",
        sources=[d / "kernel.cu", d / "bindings.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
        verbose=True,
    )
