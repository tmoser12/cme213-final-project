# Step-by-Step Custom Kernel Development Workflow

This document outlines the standard operating procedure for developing, testing, and integrating a single custom CUDA kernel into the Qwen 7B model. 

Following these steps strictly ensures you don't spend hours debugging a kernel inside a 7-Billion parameter model, but instead verify it in complete isolation first.

---

## Step 1: Establish the PyTorch Baseline (`baseline.py`)
Before writing any C++ or CUDA, you must establish the "ground truth" implementation. 

1. Open `src/models/modeling_qwen2.py` (the reference file you extracted).
2. Locate the specific `forward()` pass for the module you are replacing (e.g., `Qwen2RMSNorm` or `Qwen2MLP`).
3. Copy the exact mathematical operations into a standalone function in `baseline.py`.

```python
# src/kernels/swiglu/baseline.py
import torch
import torch.nn.functional as F

def pytorch_swiglu_baseline(x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """The unoptimized PyTorch eager implementation (Ground Truth)."""
    gate_proj = F.linear(x, gate_weight)
    up_proj = F.linear(x, up_weight)
    return F.silu(gate_proj) * up_proj
```

---

## Step 2: Write the CUDA Kernel & Bindings (`kernel.cu`, `bindings.cpp`)
Write your actual hardware-optimized code.

1. **`kernel.cu`**: Write your `__global__` CUDA kernel and the C++ host function that allocates outputs and launches the grid/blocks.
2. **`bindings.cpp`**: Expose the C++ host function to Python using PyBind11.

```cpp
// src/kernels/swiglu/bindings.cpp
#include <torch/extension.h>

torch::Tensor launch_fused_swiglu(torch::Tensor x, torch::Tensor gate_w, torch::Tensor up_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_swiglu, "Fused SwiGLU forward pass");
}
```

---

## Step 3: Setup JIT Compilation (`jit.py`)
Create the script that automatically compiles your `.cu` and `.cpp` files into a Python library.

```python
# src/kernels/swiglu/jit.py
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

## Step 4: Correctness Testing (`test.py`)
Never benchmark or integrate a kernel until it passes this test.

1. Generate random dummy tensors of the exact shape and `dtype` (FP16) the model uses.
2. Run the baseline function.
3. Run your custom kernel.
4. Compare the outputs using `torch.allclose`.

```python
# src/kernels/swiglu/test.py
import torch
from baseline import pytorch_swiglu_baseline
from jit import get_ops

def run_test():
    custom_ops = get_ops()
    
    # Dummy tensors (Batch=1, Seq=128, Hidden=3584)
    x = torch.randn(1, 128, 3584, dtype=torch.float16, device="cuda")
    gate_w = torch.randn(18944, 3584, dtype=torch.float16, device="cuda")
    up_w = torch.randn(18944, 3584, dtype=torch.float16, device="cuda")
    
    out_baseline = pytorch_swiglu_baseline(x, gate_w, up_w)
    out_custom = custom_ops.forward(x, gate_w, up_w)
    
    # FP16 requires slightly looser tolerances
    assert torch.allclose(out_baseline, out_custom, atol=1e-3, rtol=1e-3)
    print("✅ Correctness test passed!")
```

---

## Step 5: Micro-Benchmarking (`benchmark.py`)
Now, prove that your custom kernel is actually faster than PyTorch. 

You must measure against **two** baselines using `torch.cuda.Event` (since CUDA launches are asynchronous).

```python
# src/kernels/swiglu/benchmark.py
import torch
from baseline import pytorch_swiglu_baseline
from jit import get_ops

def benchmark():
    # ... setup tensors as in test.py ...
    
    # Warmup
    for _ in range(10):
        pytorch_swiglu_baseline(...)
        
    # 1. Benchmark Eager PyTorch (The Lower Bound)
    # 2. Benchmark Optimized PyTorch (e.g., torch.compile(pytorch_swiglu_baseline))
    # 3. Benchmark Custom Kernel (The Upper Bound)
    
    # Example timing loop:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(1000):
        custom_ops.forward(...)
    end.record()
    torch.cuda.synchronize()
    
    print(f"Custom Kernel Time: {start.elapsed_time(end) / 1000:.3f} ms per run")
```

---

## Step 6: Model Integration (`wrapper.py`)
Once the kernel is correct and fast, wrap it in a `torch.nn.Module` and write the patch function to inject it into the full Hugging Face model.

1. Inherit from `nn.Module`.
2. Accept the original `modeling_qwen2.py` module as an input to the `__init__` function.
3. Keep references to the original weights.
4. Call your custom kernel in the `forward` pass.

```python
# src/kernels/swiglu/wrapper.py
import torch.nn as nn
from .jit import get_ops

custom_ops = get_ops()

class CustomQwenMLP(nn.Module):
    def __init__(self, original_mlp):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        
    def forward(self, x):
        # Swap the eager PyTorch implementation with our custom CUDA kernel
        fused_activation = custom_ops.forward(x, self.gate_proj.weight, self.up_proj.weight)
        return self.down_proj(fused_activation)

def patch_model(model):
    for layer in model.model.layers:
        layer.mlp = CustomQwenMLP(layer.mlp)
```

**Final Step:** Add `--patch-swiglu` to your `scripts/run_patched_model.py` to trigger the `patch_model()` function, and run the full end-to-end evaluation!
