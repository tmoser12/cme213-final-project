"""
kernels/setup.py
Build the custom CUDA kernels as a PyTorch C++/CUDA extension.

Build (in-place) on a GPU node with the cme213 conda env active:

    cd kernels
    python setup.py build_ext --inplace

This produces `kernels/_C*.so`, which `kernels/__init__.py` imports.

Targets sm_75 only (Quadro RTX 6000). No fat binary -- nvcc 12.3 + torch
2.4.0 (CUDA 12.x runtime) is the only configuration this needs to support.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="kernels",
    ext_modules=[
        CUDAExtension(
            name="kernels._C",
            sources=["embedding.cu"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_75,code=sm_75",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
