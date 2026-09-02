import os
import shutil

def _sanitize_env_for_nvcc():
    # Strip NVHPC from PATH to prevent nvcc from forcing nvc++
    paths = os.environ.get("PATH", "").split(":")
    paths = [p for p in paths if "nvidia-hpc-sdk" not in p and "/nvhpc/" not in p]
    os.environ["PATH"] = ":".join(paths)
    
    # Strip NVHPC specific environment variables
    for var in ["CC", "CXX", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN"]:
        val = os.environ.get(var, "").lower()
        if "nvhpc" in val or "nvidia-hpc-sdk" in val or "nvc" in val:
            del os.environ[var]
            
    # Force CUDA_HOME to the standard cuda toolkit, not NVHPC
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

# MUST be called BEFORE importing torch.utils.cpp_extension
_sanitize_env_for_nvcc()

from torch.utils.cpp_extension import load
from pathlib import Path

def get_ops():
    d = Path(__file__).resolve().parent
    return load(
        name="custom_rmsnorm_ops",
        sources=[d / "kernel.cu", d / "bindings.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"] # Turing GPU optimized
    )
