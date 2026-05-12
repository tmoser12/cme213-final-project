# Custom CUDA Kernel Development Strategy (Monkey Patching)

This document outlines the step-by-step development environment and workflow for replacing specific operations in Hugging Face models (like Qwen 7B) with custom CUDA kernels.

## 1. Recommended Directory Structure

To keep the project organized, isolate your CUDA C++ code from your Python inference logic:

```text
cme213-final-project/
├── src/
│   ├── kernels/                  # All custom CUDA code
│   │   ├── custom_swiglu.cu      # CUDA kernel implementations
│   │   ├── bindings.cpp          # Pybind11 Python bindings
│   │   └── jit_compile.py        # Helper to compile/load the extension
│   ├── models/
│   │   ├── wrappers.py           # PyTorch nn.Module wrappers for custom kernels
│   │   └── patcher.py            # Logic to traverse model and swap modules
│   └── inference/
├── scripts/
│   ├── test_custom_kernel.py     # Isolated test for just the kernel
│   └── verify_monkey_patch.py    # End-to-end model correctness verification
└── ...
```

## 2. Step-by-Step Environment Setup

### Step 2.1: Scaffold the CUDA Files
Create the base files in `src/kernels/`. 

**`src/kernels/custom_swiglu.cu`**: Contains your `__global__` CUDA functions and the C++ host launcher functions.
**`src/kernels/bindings.cpp`**: Uses `pybind11` to expose the C++ host launchers to Python.

```cpp
// Example bindings.cpp structure
#include <torch/extension.h>

// Declaration of the CUDA launcher
torch::Tensor launch_custom_swiglu(torch::Tensor x, torch::Tensor weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &launch_custom_swiglu, "Custom SwiGLU forward pass");
}
```

### Step 2.2: Set Up JIT Compilation
Instead of writing complex `setup.py` files, use PyTorch's Just-In-Time (JIT) compiler. This automatically compiles the `.cu` files the first time you run your script and caches the `.so` binary.

Create `src/kernels/jit_compile.py`:

```python
import os
from torch.utils.cpp_extension import load
from pathlib import Path

def load_custom_ops():
    kernel_dir = Path(__file__).resolve().parent
    
    # This will compile the code and return a Python module
    custom_ops = load(
        name="qwen_custom_kernels",
        sources=[
            kernel_dir / "custom_swiglu.cu",
            kernel_dir / "bindings.cpp"
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"], # Turing architecture
        verbose=True
    )
    return custom_ops
```

### Step 2.3: Implement the PyTorch Wrapper
Create `src/models/wrappers.py`. This module acts as the bridge between Hugging Face's architecture and your custom C++ extension.

```python
import torch
import torch.nn as nn
from src.kernels.jit_compile import load_custom_ops

# Load the compiled kernels
custom_ops = load_custom_ops()

class CustomQwenMLP(nn.Module):
    def __init__(self, original_mlp):
        super().__init__()
        # Transfer the weights from the HF model
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn # e.g., silu
        
    def forward(self, x):
        # Call your custom CUDA operation
        # This replaces: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        return custom_ops.swiglu_forward(
            x, 
            self.gate_proj.weight, 
            self.up_proj.weight, 
            self.down_proj.weight
        )
```

### Step 2.4: Implement the Patcher
Create `src/models/patcher.py`. This script handles the "Monkey Patching" by swapping out the specific attributes in the loaded Hugging Face model.

```python
from src.models.wrappers import CustomQwenMLP

def patch_qwen_model(model):
    """
    Replaces all MLP layers in a Qwen2.5 model with CustomQwenMLP.
    """
    for i, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        
        # Instantiate custom wrapper with original weights
        custom_mlp = CustomQwenMLP(original_mlp)
        
        # Replace the module
        layer.mlp = custom_mlp
        
    print(f"✅ Successfully patched {len(model.model.layers)} MLP layers.")
    return model
```

## 3. The Iterative Workflow

Once the environment is set up, your development loop should look like this:

1. **Edit CUDA Code**: Make changes to `src/kernels/custom_swiglu.cu`.
2. **Clear Cache (Optional)**: If PyTorch JIT fails to detect a change, delete the cache: `rm -rf ~/.cache/torch_extensions/py311_cu121/qwen_custom_kernels`
3. **Run Isolated Test**: Run `scripts/test_custom_kernel.py` to test the kernel with dummy random tensors. Validate it against standard PyTorch functions using `torch.allclose()`.
4. **Run End-to-End Test**: Run `scripts/verify_monkey_patch.py`. This script should:
   - Load the original model and save the logits for a test prompt.
   - Run `patch_qwen_model(model)`.
   - Run the prompt through the patched model.
   - Assert `torch.allclose(original_logits, patched_logits, atol=1e-2)`.
5. **Benchmark**: Once correctness is verified, run your `benchmarks/run_baseline.py` on the patched model to measure the speedup.
