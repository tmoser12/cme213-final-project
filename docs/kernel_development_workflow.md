# Step-by-Step Custom Kernel Development Workflow

> **Historical SOP.** Written while the first kernels were being built; module paths are shown for the
> `target` (7B) tree. The same steps apply to `kernel_dev/draft/kernels/` (0.5B). The final,
> AOT-compiled versions of every kernel live under `runtime/production_kernels/`.

This document outlines the standard operating procedure for developing, testing, and integrating a single custom CUDA kernel into the Qwen 7B model. 

By testing at the `nn.Module` level against the native Hugging Face implementation across multiple batch dimensions, you ensure your performance numbers accurately reflect the true, end-to-end speedup that the model will experience.

---

## Step 1: Write the CUDA Kernel & Bindings (`kernel.cu`, `bindings.cpp`)
Write your actual hardware-optimized code.

1. **`kernel.cu`**: Write your `__global__` CUDA kernel and the C++ host function that allocates outputs and launches the grid/blocks.
2. **`bindings.cpp`**: Expose the C++ host function to Python using PyBind11.

```cpp
// kernel_dev/target/kernels/swiglu/bindings.cpp
#include <torch/extension.h>

torch::Tensor launch_fused_swiglu(torch::Tensor x, torch::Tensor gate_w, torch::Tensor up_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_swiglu, "Fused SwiGLU forward pass");
}
```

---

## Step 2: Setup JIT Compilation (`jit.py`)
Create the script that compiles your `.cu` and `.cpp` files. 

**CRITICAL:** Always include the environment sanitization block before importing `torch` to prevent the cluster's module environment from forcing PyTorch to use an incompatible NVIDIA HPC compiler.

```python
# kernel_dev/target/kernels/swiglu/jit.py
import os
import shutil

def _sanitize_env_for_nvcc():
    paths = os.environ.get("PATH", "").split(":")
    paths = [p for p in paths if "nvidia-hpc-sdk" not in p and "/nvhpc/" not in p]
    os.environ["PATH"] = ":".join(paths)
    for var in ["CC", "CXX", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN"]:
        val = os.environ.get(var, "").lower()
        if "nvhpc" in val or "nvidia-hpc-sdk" in val or "nvc" in val:
            del os.environ[var]
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

_sanitize_env_for_nvcc()

from torch.utils.cpp_extension import load
from pathlib import Path

def get_ops():
    d = Path(__file__).resolve().parent
    return load(
        name="custom_swiglu_ops",
        sources=[d / "kernel.cu", d / "bindings.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"] # Turing GPU
    )
```

---

## Step 3: Model Wrapper (`wrapper.py`)
Create a `torch.nn.Module` that accepts the original Hugging Face module, steals its weights (to avoid reallocating GPU memory), and calls your custom JIT-compiled kernel.

```python
# kernel_dev/target/kernels/swiglu/wrapper.py
import torch.nn as nn
from kernel_dev.target.kernels.swiglu.jit import get_ops

custom_ops = get_ops()

class CustomQwenMLP(nn.Module):
    def __init__(self, original_mlp):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        
    def forward(self, x):
        fused_activation = custom_ops.forward(x, self.gate_proj.weight, self.up_proj.weight)
        return self.down_proj(fused_activation)

def patch_model(model):
    for layer in model.model.layers:
        layer.mlp = CustomQwenMLP(layer.mlp)
```

---

## Step 4: Correctness & Micro-Benchmarking (`benchmark.py`)
You must prove the kernel is mathematically correct, and then benchmark it against *two* baselines: PyTorch Eager and `torch.compile`. You should test across multiple batch size and sequence length combinations to understand the kernel's behavior.

```python
# kernel_dev/target/kernels/swiglu/benchmark.py
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from kernel_dev.target.kernels.swiglu.wrapper import CustomQwenMLP

def run_benchmark_for_config(b, s, hf_baseline, hf_compiled, custom_module):
    x = torch.randn(b, s, 3584, dtype=torch.float16, device="cuda")
    
    # 1. Correctness check before benchmarking
    with torch.no_grad():
        assert torch.allclose(hf_baseline(x), custom_module(x), atol=1e-3, rtol=1e-3)
        
    # 2. Warmup
    with torch.no_grad():
        for _ in range(20):
            hf_baseline(x); hf_compiled(x); custom_module(x)
            
    # 3. Timing Logic (using torch.cuda.Event)
    # ... record start/end for baseline, compiled, and custom ...
    # return hf_time, comp_time, custom_time

def main():
    # Instantiate the real Hugging Face module directly
    hf_baseline = Qwen2MLP(config).cuda().half()
    hf_compiled = torch.compile(hf_baseline)
    
    # Wrap it
    custom_module = CustomQwenMLP(hf_baseline).cuda()
    
    configs = [(1, 1), (1, 128), (2, 128), (8, 512), (16, 1024)]
    results = []
    
    for b, s in configs:
        hf_time, comp_time, custom_time = run_benchmark_for_config(b, s, hf_baseline, hf_compiled, custom_module)
        results.append(...)
        
    # Save a markdown report!
```
---

## Step 5: Execution Script (`run_benchmark.sh`)
Because PyTorch eagerly caches JIT-compiled C++ extensions, you must clear the cache every time you modify `kernel.cu` or `bindings.cpp`. To automate this, create a standard bash script in your kernel folder that clears the cache and submits the benchmark to SLURM.

```bash
# kernel_dev/target/kernels/swiglu/run_benchmark.sh
#!/bin/bash
cd "$(dirname "$0")/../../.." || exit
source setup.sh

KERNEL_DIR=$(basename "$(dirname "$0")")
echo "Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

srun --partition=gpu-turing --gres=gpu:1 python -m kernel_dev.target.kernels.${KERNEL_DIR}.benchmark
```

**Final Step:** Once the script proves your kernel is correct and fast, add `--patch-swiglu` to your `scripts/run_patched_model.py` to trigger the `patch_model()` function, and run the full end-to-end evaluation!
