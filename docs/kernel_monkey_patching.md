# Modular CUDA Kernel Development Strategy (Monkey Patching)

> **Historical design note.** This is the original plan for how custom kernels would be developed
> and spliced into the HuggingFace model. The layout it sketches (`test.py`, `baseline.py`,
> `scripts/run_patched_model.py`) evolved into `kernel_dev/<draft|target>/kernels/<op>/`, where each
> op's `benchmark.py` performs the correctness check against the HF module *and* the timing, and
> `run_benchmark.sh` drives it on a GPU node. The methodology (JIT-build one op, monkey-patch it into
> the HF `nn.Module`, compare against the reference) is exactly what was used.

Based on your requirements, the development environment should be highly modular. Every custom operation (like SwiGLU or Attention) will have its own self-contained directory containing its CUDA code, bindings, PyTorch wrapper, and isolated tests. A high-level script will then allow you to dynamically choose which of these modules to patch into the full model.

## 1. Modular Directory Structure

```text
cme213-final-project/
├── src/
│   ├── kernels/
│   │   ├── swiglu/                   # Self-contained SwiGLU component
│   │   │   ├── kernel.cu             # The core CUDA __global__ functions
│   │   │   ├── bindings.cpp          # PyBind11 interface
│   │   │   ├── jit.py                # JIT compilation logic for this specific kernel
│   │   │   ├── wrapper.py            # Custom nn.Module that wraps the compiled kernel
│   │   │   ├── baseline.py           # Pure PyTorch reference implementation
│   │   │   ├── test.py               # Isolated unit test (kernel vs. baseline)
│   │   │   └── benchmark.py          # Micro-benchmark (Eager vs. cuBLAS vs. Custom)
│   │   │
│   │   ├── attention/                # Self-contained Attention component
│   │   │   ├── kernel.cu
│   │   │   └── ...
│   │   └── ...
│   └── inference/
├── scripts/
│   └── run_patched_model.py          # Master script to patch the model and run E2E tests
└── ...
```

## 2. Anatomy of a Kernel Module (e.g., `swiglu/`)

Inside `kernel_dev/target/kernels/swiglu/`, you will have everything needed to build and test the SwiGLU operation independently of the rest of the project.

### 2.1 JIT Compilation (`swiglu/jit.py`)
This script only compiles the files inside its own directory.
```python
from torch.utils.cpp_extension import load
from pathlib import Path

def get_ops():
    d = Path(__file__).resolve().parent
    return load(
        name="custom_swiglu",
        sources=[d / "kernel.cu", d / "bindings.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
    )
```

### 2.2 PyTorch Wrapper (`swiglu/wrapper.py`)
This defines the drop-in replacement module. It loads the compiled ops via `jit.py`.
```python
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
        return custom_ops.swiglu_forward(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        )

def patch_model(model):
    """Entry point for the high-level script to apply this patch."""
    for layer in model.model.layers:
        layer.mlp = CustomQwenMLP(layer.mlp)
    print(f"✅ Patched {len(model.model.layers)} SwiGLU layers.")
```

### 2.3 Isolated Unit Test (`swiglu/test.py`)
This allows you to test the kernel in a vacuum before touching the 7B model. Ensure you use `torch.allclose()` to verify numerical stability against the naive PyTorch implementation.
```python
import torch
from .jit import get_ops
from .baseline import pytorch_swiglu_reference

def test():
    ops = get_ops()
    # 1. Generate random dummy tensors
    # 2. Run ops.swiglu_forward(...)
    # 3. Run pytorch_swiglu_reference(...)
    # 4. Assert torch.allclose(...)
    print("SwiGLU Kernel verified!")

if __name__ == "__main__":
    test()
```

### 2.4 Micro-Benchmarking (`swiglu/benchmark.py`)
Once correctness is verified, you should benchmark your kernel against **two** baselines to establish the full performance picture:

1. **The Lower Bound (PyTorch Eager):** Compare against the naive PyTorch implementation (e.g., standard math operations without `torch.compile`). This proves your kernel works and overcomes standard Python overhead.
2. **The Upper Bound (Optimized Backend):** Compare against the highly optimized backend (e.g., `cuBLAS` for GEMMs, or xFormers SDPA for Attention). This shows how close your hand-written kernel gets to industry-standard, hand-tuned CUDA code.

```python
import torch
import time
# setup tensors...

# 1. Time Eager PyTorch
# 2. Time Optimized Backend (cuBLAS/SDPA)
# 3. Time Custom Kernel
```


## 3. High-Level Master Script (`scripts/run_patched_model.py`)

This script loads the standard Hugging Face model, parses command-line arguments to determine which patches to apply, delegates the patching to the respective module wrappers, and finally runs a forward pass or benchmark.

```python
#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the patch functions from each component
from kernel_dev.target.kernels.swiglu.wrapper import patch_model as patch_swiglu
# from kernel_dev.target.kernels.attention.wrapper import patch_model as patch_attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--patch-swiglu", action="store_true", help="Replace MLP with custom SwiGLU")
    parser.add_argument("--patch-attention", action="store_true", help="Replace Attention with custom kernel")
    args = parser.parse_args()

    print(f"Loading baseline model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ---------------------------------------------------------
    # Apply requested patches dynamically
    # ---------------------------------------------------------
    if args.patch_swiglu:
        patch_swiglu(model)
        
    if args.patch_attention:
        # patch_attention(model)
        pass

    # ---------------------------------------------------------
    # End-to-End Verification / Benchmarking
    # ---------------------------------------------------------
    print("Running forward pass...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
    print(f"Output: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    main()
```

### Example Usage:
```bash
# Run isolated unit test for SwiGLU
python kernel_dev/target/kernels/swiglu/test.py

# Run the full model with NO patches (baseline)
python scripts/run_patched_model.py

# Run the full model with ONLY the SwiGLU patch
python scripts/run_patched_model.py --patch-swiglu

# Run the full model with ALL patches
python scripts/run_patched_model.py --patch-swiglu --patch-attention
```
