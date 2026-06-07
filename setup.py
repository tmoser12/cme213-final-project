"""AOT build for runtime/production_kernels/{target,draft} CUDA extensions.

Build one target kernel:
    BUILD_KERNEL=rmsnorm python setup.py build_ext --inplace

Build all kernels (both roles):
    python setup.py build_ext --inplace

Build a single role / op (used by scripts/build_kernels.sh):
    BUILD_ROLE=draft BUILD_KERNEL=attention python setup.py build_ext --inplace
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
PRODUCTION_KERNELS = ROOT / "runtime" / "production_kernels"

NVCC_FLAGS = ["-O3", "--use_fast_math", "-arch=sm_75"]
CXX_FLAGS = ["-O3"]

# Each op compiles kernel.cu + bindings.cpp into <role>_<op>_ops, beside its ops.py.
OPS = ["rmsnorm", "embedding", "residual_ops", "swiglu", "attention"]
ROLES = ["target", "draft"]
SOURCES = ["kernel.cu", "bindings.cpp"]


def _module_name(role: str, op: str) -> str:
    # Match the names each ops.py imports: "<role>_<op>_ops", except ops whose
    # name already ends in "_ops" (e.g. residual_ops -> "<role>_residual_ops").
    suffix = op if op.endswith("_ops") else f"{op}_ops"
    return f"{role}_{suffix}"


def _make_extension(role: str, op: str) -> CUDAExtension:
    src_dir = PRODUCTION_KERNELS / role / op
    return CUDAExtension(
        name=f"runtime.production_kernels.{role}.{op}.{_module_name(role, op)}",
        sources=[str(src_dir / s) for s in SOURCES],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )


def _selected_extensions() -> list[CUDAExtension]:
    # BUILD_ROLE in {target, draft, all}; BUILD_KERNEL in {<op>, all}. Both default
    # to "all". The draft tree may not exist on older checkouts — skip missing roles.
    build_role = os.environ.get("BUILD_ROLE", "all")
    build_kernel = os.environ.get("BUILD_KERNEL", "all")

    roles = ROLES if build_role == "all" else [build_role]
    if build_role != "all" and build_role not in ROLES:
        raise SystemExit(f"Unknown BUILD_ROLE={build_role!r}. Choose one of: {', '.join(ROLES)}, all")

    ops = OPS if build_kernel == "all" else [build_kernel]
    if build_kernel != "all" and build_kernel not in OPS:
        raise SystemExit(f"Unknown BUILD_KERNEL={build_kernel!r}. Choose one of: {', '.join(OPS)}, all")

    exts: list[CUDAExtension] = []
    for role in roles:
        if not (PRODUCTION_KERNELS / role).is_dir():
            continue
        exts.extend(_make_extension(role, op) for op in ops)
    return exts


setup(
    name="cme213-runtime-kernels",
    version="0.1.0",
    ext_modules=_selected_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
