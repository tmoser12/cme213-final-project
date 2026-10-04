"""AOT build for runtime/production_kernels/{target,draft} CUDA extensions.

Build everything (both roles, all ops):
    python setup.py build_ext --inplace

Restrict by role and/or op via env vars (default "all" for each):
    BUILD_ROLE=draft python setup.py build_ext --inplace            # all draft ops
    BUILD_ROLE=draft BUILD_KERNEL=rmsnorm python setup.py build_ext --inplace

Each op compiles to ``<role>_<op>_ops`` colocated with its ops.py, e.g.
``runtime/production_kernels/draft/rmsnorm/draft_rmsnorm_ops*.so``.
"""

from __future__ import annotations

import os
from pathlib import Path


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
KERNELS_ROOT = ROOT / "runtime" / "production_kernels"

# Let PyTorch select the visible GPU, or honor TORCH_CUDA_ARCH_LIST for builds
# without a GPU (e.g. 8.9 on an RTX 4090).
# A hard-coded -arch suppresses PyTorch's architecture selection entirely.
NVCC_FLAGS = ["-O3", "--use_fast_math"]
CXX_FLAGS = ["-O3"]

ROLES = ("target", "draft")

# Same op set + sources for both roles; the .cu under each role/ dir differs
# (target is tuned for 7B dims, draft for 0.5B). The module name encodes the
# role so the two .so files never collide: ``<role>_<op>_ops``.
KERNELS: dict[str, list[str]] = {
    "rmsnorm": ["kernel.cu", "bindings.cpp"],
    "embedding": ["kernel.cu", "bindings.cpp"],
    "residual_ops": ["kernel.cu", "bindings.cpp"],
    "swiglu": ["kernel.cu", "bindings.cpp"],
    "attention": ["kernel.cu", "bindings.cpp"],
}


def _module_basename(role: str, op: str) -> str:
    """``<role>_<op>_ops`` — but ``residual_ops`` already ends in ``_ops``.

    Must match what each ops.py imports (e.g. ``draft_residual_ops``, not
    ``draft_residual_ops_ops``).
    """
    suffix = op if op.endswith("_ops") else f"{op}_ops"
    return f"{role}_{suffix}"


def _extension_module(role: str, op: str) -> str:
    """Dotted module path — build_ext --inplace places .so beside ops.py."""
    return f"runtime.production_kernels.{role}.{op}.{_module_basename(role, op)}"


# WALKTHROUGH [EXTENSION] — see demo/WALKTHROUGH.md.
# One shared library combines a Python binding with CUDA implementation code.
# BuildExtension invokes the host compiler/nvcc; inference imports the result.
def _make_extension(role: str, op: str) -> CUDAExtension:
    src_dir = KERNELS_ROOT / role / op
    return CUDAExtension(
        name=_extension_module(role, op),
        sources=[str(src_dir / s) for s in KERNELS[op]],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )


def _selected_extensions() -> list[CUDAExtension]:
    build_role = os.environ.get("BUILD_ROLE", "all")
    build_kernel = os.environ.get("BUILD_KERNEL", "all")

    if build_role == "all":
        roles = list(ROLES)
    elif build_role in ROLES:
        roles = [build_role]
    else:
        raise SystemExit(
            f"Unknown BUILD_ROLE={build_role!r}. Choose one of: {', '.join(ROLES)}, all"
        )

    if build_kernel == "all":
        ops = list(KERNELS)
    elif build_kernel in KERNELS:
        ops = [build_kernel]
    else:
        known = ", ".join(sorted(KERNELS))
        raise SystemExit(
            f"Unknown BUILD_KERNEL={build_kernel!r}. Choose one of: {known}, all"
        )

    return [_make_extension(role, op) for role in roles for op in ops]


setup(
    name="cme213-runtime-kernels",
    version="0.1.0",
    ext_modules=_selected_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
