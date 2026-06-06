"""AOT build for runtime/production_kernels/target CUDA extensions.

Build one kernel:
    BUILD_KERNEL=rmsnorm python setup.py build_ext --inplace

Build all target kernels:
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _sanitize_env_for_nvcc() -> None:
    """Strip NVHPC from env so PyTorch extensions compile with g++/nvcc."""
    paths = os.environ.get("PATH", "").split(":")
    paths = [p for p in paths if "nvidia-hpc-sdk" not in p and "/nvhpc/" not in p]
    os.environ["PATH"] = ":".join(paths)

    for var in ["CC", "CXX", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN"]:
        val = os.environ.get(var, "").lower()
        if "nvhpc" in val or "nvidia-hpc-sdk" in val or "nvc" in val:
            os.environ.pop(var, None)

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))


_sanitize_env_for_nvcc()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "runtime" / "production_kernels" / "target"

NVCC_FLAGS = ["-O3", "--use_fast_math", "-arch=sm_75"]
CXX_FLAGS = ["-O3"]

KERNELS: dict[str, dict[str, object]] = {
    "rmsnorm": {
        "module": "target_rmsnorm_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    "embedding": {
        "module": "target_embedding_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    "residual_ops": {
        "module": "target_residual_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    "swiglu": {
        "module": "target_swiglu_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    "attention": {
        "module": "target_attention_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
}

PACKAGE_PREFIX = "runtime.production_kernels.target"


def _extension_module(op: str, spec: dict[str, object]) -> str:
    """Dotted module path — build_ext --inplace places .so beside ops.py."""
    return f"{PACKAGE_PREFIX}.{op}.{spec['module']}"


def _make_extension(name: str, spec: dict[str, object]) -> CUDAExtension:
    src_dir = TARGET / name
    sources = spec["sources"]
    assert isinstance(sources, list)
    return CUDAExtension(
        name=_extension_module(name, spec),
        sources=[str(src_dir / s) for s in sources],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )


def _selected_extensions() -> list[CUDAExtension]:
    build_kernel = os.environ.get("BUILD_KERNEL", "all")
    if build_kernel == "all":
        return [_make_extension(name, spec) for name, spec in KERNELS.items()]
    if build_kernel not in KERNELS:
        known = ", ".join(sorted(KERNELS))
        raise SystemExit(
            f"Unknown BUILD_KERNEL={build_kernel!r}. Choose one of: {known}, all"
        )
    return [_make_extension(build_kernel, KERNELS[build_kernel])]


setup(
    name="cme213-runtime-kernels",
    version="0.1.0",
    ext_modules=_selected_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
