# Runtime Kernel System

How custom CUDA kernels in `runtime/` are compiled, exposed to Python, tested, and wired into the inference engine.

## Overview

The runtime follows a **Python host + prebuilt CUDA extensions** split:

- **Python** owns orchestration: config loading, weight I/O, buffer allocation, and the decoder loop (`executor.py`, planned).
- **C++/CUDA** owns compute: each op is a `kernel.cu` + `bindings.cpp` pair compiled ahead of time (AOT) into a shared library.
- **`ops.py`** is the only Python surface the inference engine imports — a thin wrapper over the prebuilt extension module.

Kernels are grouped by **model role** under `runtime/production_kernels/<role>/`. Today only `target/` (the main model) exists; a future `draft/` tree will hold speculative-decoding draft-model kernels with the same layout.

```
YAML config  →  RuntimeConfig  →  executor (Python)
                                      │
                                      ▼
                         production_kernels/target/<op>/ops.py
                                      │
                                      ▼
                         target_<op>_ops.so  (AOT pybind extension)
                                      │
                                      ▼
                              kernel.cu launchers
```

End-to-end, a single RMSNorm call looks like this:

```python
# What the executor (or a test) writes:
from runtime.production_kernels.target.rmsnorm import forward

out = forward(hidden, weights["model.layers.0.input_layernorm.weight"], eps=1e-6)
#   → ops.py calls target_rmsnorm_ops.forward(...)
#     → bindings.cpp releases GIL, calls rmsnorm_forward(...)
#       → kernel.cu launches rmsnorm_forward_kernel_vectorized<<<...>>>
```

## Directory layout

```
runtime/production_kernels/
└── target/                         # main-model kernel set
    ├── rmsnorm/
    │   ├── kernel.cu               # CUDA kernels + C++ launchers
    │   ├── bindings.cpp            # pybind11 exports → Python
    │   ├── ops.py                  # host ABI (executor imports this)
    │   ├── __init__.py             # re-exports ops.py symbols
    │   ├── jit.py                  # legacy JIT loader (benchmarks only)
    │   ├── wrapper.py              # HF monkey-patch helpers (benchmarks only)
    │   └── benchmark.py            # correctness + micro-benchmarks
    ├── embedding/
    ├── attention/                  # multi-sub-op module (see below)
    ├── swiglu/
    └── residual_ops/               # residual_add + lm_head

setup.py                            # registers all CUDAExtension targets
scripts/build_kernels.sh            # cluster-friendly build wrapper
```

**Inference path:** `executor.py` → `ops.py` → prebuilt `.so`.

**Not on the inference path:** `jit.py`, `wrapper.py`, `benchmark.py`, `run_benchmark.sh`. These remain for kernel development and SLURM micro-benchmarks.

## Build pipeline (AOT)

Extensions are compiled **once at build time**, not at import time. This avoids slow JIT recompiles on every SLURM cold start.

### 1. `setup.py` registers extensions

Root `setup.py` maps each op folder to a PyTorch `CUDAExtension`. The `KERNELS` dict is the source of truth:

```python
# setup.py (excerpt)
TARGET = ROOT / "runtime" / "production_kernels" / "target"
NVCC_FLAGS = ["-O3", "--use_fast_math", "-arch=sm_75"]

KERNELS = {
    "rmsnorm": {
        "module": "target_rmsnorm_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    "embedding": {
        "module": "target_embedding_ops",
        "sources": ["kernel.cu", "bindings.cpp"],
    },
    # ... residual_ops, swiglu, attention ...
}

def _make_extension(name, spec):
    src_dir = TARGET / name
    return CUDAExtension(
        name=str(spec["module"]),
        sources=[str(src_dir / s) for s in spec["sources"]],
        extra_compile_args={"cxx": ["-O3"], "nvcc": NVCC_FLAGS},
    )
```

| Op folder       | Extension module          | Sources                          |
|-----------------|---------------------------|----------------------------------|
| `rmsnorm`       | `target_rmsnorm_ops`      | `kernel.cu`, `bindings.cpp`      |
| `embedding`     | `target_embedding_ops`    | `kernel.cu`, `bindings.cpp`      |
| `residual_ops`  | `target_residual_ops`     | `kernel.cu`, `bindings.cpp`      |
| `swiglu`        | `target_swiglu_ops`       | `kernel.cu`, `bindings.cpp`      |
| `attention`     | `target_attention_ops`    | `kernel.cu`, `bindings.cpp`      |

### 2. `build_kernels.sh` runs the build on the cluster

```bash
#!/bin/bash
# scripts/build_kernels.sh (excerpt)
KERNEL="${1:-all}"
export BUILD_KERNEL="$KERNEL"
module load gnu12/12.3.0
export CC=gcc
export CXX=g++
conda run -n cme213 env BUILD_KERNEL="$BUILD_KERNEL" python setup.py build_ext --inplace
```

Build one kernel or all:

```bash
# from project root
bash scripts/build_kernels.sh              # all target kernels
bash scripts/build_kernels.sh rmsnorm      # single kernel

# equivalent direct invocation
BUILD_KERNEL=rmsnorm python setup.py build_ext --inplace
```

Successful builds drop `.so` files in the project root (e.g. `target_rmsnorm_ops.cpython-311-x86_64-linux-gnu.so`), importable as top-level Python modules:

```python
import target_rmsnorm_ops
print(target_rmsnorm_ops.__file__)
# → /path/to/project/target_rmsnorm_ops.cpython-311-x86_64-linux-gnu.so
#   (NOT ~/.cache/torch_extensions/...)
```

### 3. Rebuild when CUDA sources change

Rebuild after editing `kernel.cu` or `bindings.cpp`. Inference startup is a plain `import` — no compile step at runtime.

## Layer stack: CUDA → Python

Each op follows the same vertical stack. RMSNorm is the canonical example.

### Step 1: `kernel.cu` — device code + launchers

The `__global__` kernel does the math. The C++ host function validates inputs, allocates the output tensor, and launches:

```cpp
// runtime/production_kernels/target/rmsnorm/kernel.cu (excerpt)

__global__ void rmsnorm_forward_kernel_vectorized(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float eps
) {
    int row_idx = blockIdx.x;
    // ... warp/block reduction for RMS, vectorized fp16 normalize ...
}

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);

    auto output = torch::empty_like(input);
    int num_blocks = batch_size * seq_len;
    int num_threads = hidden_size / 8;  // 448 threads for 7B hidden=3584

    rmsnorm_forward_kernel_vectorized<<<num_blocks, num_threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        hidden_size,
        eps
    );
    return output;
}
```

Launchers accept PyTorch tensors already on GPU, assume FP16 on Turing, and return a new tensor.

### Step 2: `bindings.cpp` — pybind11 module

Exposes the launcher to Python. `TORCH_EXTENSION_NAME` is set by PyTorch's build to match the extension module name (e.g. `target_rmsnorm_ops`):

```cpp
// runtime/production_kernels/target/rmsnorm/bindings.cpp

#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps);

torch::Tensor forward_py(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       float eps) {
    py::gil_scoped_release release;   // don't hold GIL during CUDA launch
    return rmsnorm_forward(input, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_py, "Fused RMSNorm forward pass");
}
```

Embedding follows the same pattern with a different export name:

```cpp
// runtime/production_kernels/target/embedding/bindings.cpp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward_py,
          "Qwen embedding gather (CUDA, fp16)");
}
```

Attention exports **multiple** sub-ops from one module:

```cpp
// runtime/production_kernels/target/attention/bindings.cpp (excerpt)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_proj_forward",    &qkv_proj_forward,    "Fused QKV projection (cuBLAS, fp16)");
    m.def("rope_kv_write_forward", &rope_kv_write_forward, "RoPE-fused KV write");
    m.def("fused_attn_forward",  &fused_attn_forward,  "Fused causal SDPA, prefill");
    m.def("decode_attn_forward", &decode_attn_forward, "Decode attention, S==1");
    m.def("o_proj_forward",      &o_proj_forward,      "Output projection (cuBLAS, fp16)");
}
```

### Step 3: AOT build produces a `.so` module

After `bash scripts/build_kernels.sh rmsnorm`, Python can import the raw extension:

```python
import target_rmsnorm_ops

x = torch.randn(1, 128, 3584, dtype=torch.float16, device="cuda")
w = torch.ones(3584, dtype=torch.float16, device="cuda")
out = target_rmsnorm_ops.forward(x, w, 1e-6)
```

The inference engine should **not** import extensions directly — it goes through `ops.py` (next step).

### Step 4: `ops.py` — host ABI

The stable contract between the inference engine and CUDA:

```python
# runtime/production_kernels/target/rmsnorm/ops.py

_BUILD_HINT = "Run: bash scripts/build_kernels.sh rmsnorm"

def _load_ext():
    try:
        import target_rmsnorm_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    if "torch_extensions" in getattr(ext, "__file__", ""):
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    return ext

def init() -> None:
    """Verify the prebuilt extension imports (no runtime compile)."""
    _load_ext()

def workspace_bytes(*, batch: int, seq_len: int, hidden_size: int) -> int:
    return 0  # RMSNorm needs no scratch buffer

def forward(input, weight, eps: float):
    return _load_ext().forward(input, weight, eps)
```

Embedding is the same pattern with a different function name and module:

```python
# runtime/production_kernels/target/embedding/ops.py

def embedding_forward(input_ids, weight):
    return _load_ext().embedding_forward(input_ids, weight)
```

Residual ops exposes two entry points from one extension:

```python
# runtime/production_kernels/target/residual_ops/ops.py

def residual_add_forward(a, b):
    return _load_ext().residual_add_forward(a, b)

def lm_head_forward(hidden, weight):
    return _load_ext().lm_head_forward(hidden, weight)
```

### Step 5: `__init__.py` — package re-exports

Callers import from the package path, never from the raw `.so`:

```python
# runtime/production_kernels/target/rmsnorm/__init__.py

from runtime.production_kernels.target.rmsnorm.ops import forward, init, workspace_bytes
__all__ = ["forward", "init", "workspace_bytes"]
```

Usage:

```python
from runtime.production_kernels.target.rmsnorm import forward, init
from runtime.production_kernels.target.embedding import embedding_forward
from runtime.production_kernels.target.residual_ops import residual_add_forward, lm_head_forward

init()  # optional sanity check that AOT build is present
```

## Op API reference

| Package | `ops.py` entry points | Extension symbols |
|---------|----------------------|-------------------|
| `rmsnorm` | `forward(input, weight, eps)`, `init()`, `workspace_bytes(...)` | `forward` |
| `embedding` | `embedding_forward(input_ids, weight)` | `embedding_forward` |
| `residual_ops` | `residual_add_forward(a, b)`, `lm_head_forward(hidden, weight)` | same names |
| `swiglu` | *(ops.py pending)* | `swiglu_forward`, `swiglu_act_forward` |
| `attention` | *(ops.py pending)* | 7 sub-ops (see below) |

### Attention — composing sub-ops

Attention has no single `forward()`. The executor (or benchmark `wrapper.py`) chains sub-ops. This is the actual prefill path from `attention/wrapper.py`:

```python
# runtime/production_kernels/target/attention/wrapper.py (excerpt)
# Uses JIT today; executor will call the same symbols via ops.py + AOT ext.

qkv = custom_ops.qkv_proj_forward(
    hidden_states.view(M, self.hidden_size), self.W_qkv, self.b_qkv)

q = qkv[:, :H_q].reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
k = qkv[:, H_q:H_q + H_kv].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
v = qkv[:, H_q + H_kv:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

cos, sin = position_embeddings
custom_ops.rope_kv_write_forward(k, v, cache_k, cache_v, write_pos=0, cos=cos, sin=sin)

o = custom_ops.fused_attn_forward(
    q, cache_k, cache_v, cur_len=S, softmax_scale=self.softmax_scale, cos=cos, sin=sin)

y = custom_ops.o_proj_forward(
    o.transpose(1, 2).contiguous().view(M, H_q), self.W_o)
```

Pipeline:

```
hidden → qkv_proj → split Q/K/V → rope_kv_write → fused_attn/decode_attn → o_proj → output
```

RoPE is fused into `rope_kv_write_forward` (on K) and the attention kernels (on Q) — no separate rope launch in the hot path.

## Testing

GPU parity tests import through the package path and compare against HuggingFace reference modules.

### Verify AOT build (no JIT cache)

```python
# runtime/tests/test_rmsnorm.py

import importlib

class TestRmsnormAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module("target_rmsnorm_ops")
        self.assertTrue(hasattr(ext, "forward"))
        self.assertNotIn("torch_extensions", ext.__file__)
```

### Parity against HF reference

```python
# runtime/tests/test_rmsnorm.py (excerpt)

from runtime.production_kernels.target.rmsnorm import forward, init
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

init()

cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().half()

x = torch.randn(4, 512, cfg.hidden_size, dtype=torch.float16, device="cuda")
with torch.no_grad():
    expected = hf(x)
    actual = forward(x, hf.weight, cfg.rms_norm_eps)
assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)
```

### Parity with real model weights

```python
# runtime/tests/test_rmsnorm.py (excerpt)

weights = load_weights(cfg, device="cuda")
w = weights["model.layers.0.input_layernorm.weight"]

x = torch.randn(1, 32, cfg.hidden_size, dtype=torch.float16, device="cuda")
with torch.no_grad():
    expected = hf(x)
    actual = forward(x, w, cfg.rms_norm_eps)
assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)
```

Run on a GPU node:

```bash
bash slurm/run_tests_gpu.sh runtime.tests.test_rmsnorm
bash slurm/run_tests_gpu.sh runtime.tests.test_embedding
bash slurm/run_tests_gpu.sh runtime.tests.test_residual_ops
```

## Inference engine integration (planned)

Phases 4–5 of `runtime/plan.md` wire kernels into a full decoder. The planned shape:

```python
# Planned usage (executor.py does not exist yet)

from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights
from runtime.production_kernels.target.rmsnorm import forward as rmsnorm_forward
from runtime.production_kernels.target.embedding import embedding_forward
from runtime.production_kernels.target.residual_ops import residual_add_forward, lm_head_forward

cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
weights = load_weights(cfg, device="cuda")
buffers = allocate_buffers(cfg, batch=1, max_seq_len=2048)  # Phase 4

executor = Qwen2Executor(cfg, weights, buffers)  # Phase 5
logits = executor.prefill(input_ids)             # runs layer_order loop
next_token = executor.decode(input_ids[-1])      # single-token path
```

### Layer order (from YAML)

Each model YAML defines the exact op sequence matching `Qwen2DecoderLayer`:

```yaml
# runtime/core/configs/qwen2.5-7b.yaml (excerpt)
layer_order:
  - input_rmsnorm
  - attention
  - residual_add
  - post_attn_rmsnorm
  - swiglu_mlp
  - residual_add
```

The executor maps each step to an `ops.py` call. Sketch of one decoder layer:

```python
# Planned executor pseudocode — one layer, prefill path

hidden = rmsnorm_forward(
    hidden,
    weights[f"model.layers.{i}.input_layernorm.weight"],
    cfg.rms_norm_eps,
)

attn_out = run_attention_prefill(hidden, weights, buffers, layer=i)  # chains attention sub-ops
hidden = residual_add_forward(hidden, attn_out)

hidden = rmsnorm_forward(
    hidden,
    weights[f"model.layers.{i}.post_attention_layernorm.weight"],
    cfg.rms_norm_eps,
)

mlp_out = swiglu_forward(hidden, gate_w, up_w, down_w)  # once swiglu/ops.py lands
hidden = residual_add_forward(hidden, mlp_out)
```

### Prefill vs decode

| Phase | Embedding | Attention path | LM head |
|-------|-----------|----------------|---------|
| Prefill (`S > 1`) | `embedding_forward(all_ids, embed_weight)` | `fused_attn_forward` | `lm_head_forward` on last hidden |
| Decode (`S = 1`) | `embedding_forward(one_id, embed_weight)` | `decode_attn_forward` + `rope_kv_write_forward` | same |

### Kernel set selection

Configs will carry `kernel_set: target` to pick which `production_kernels/<role>/` tree to import. Dynamic resolution avoids hard-coding when a `draft/` set arrives for speculative decoding.

## Legacy JIT path (development only)

Before AOT, each op had a `jit.py` that compiled at import time. Benchmarks still use this path via `wrapper.py`:

```python
# runtime/production_kernels/target/rmsnorm/jit.py (legacy — do NOT use in executor)

from torch.utils.cpp_extension import load

def get_ops():
    d = Path(__file__).resolve().parent
    return load(
        name="custom_rmsnorm_ops",
        sources=[d / "kernel.cu", d / "bindings.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"],
    )
```

JIT vs AOT at a glance:

| | AOT (`ops.py`) | JIT (`jit.py`) |
|--|----------------|----------------|
| When compiled | `build_kernels.sh` | first import / cache hit |
| Module location | project root `.so` | `~/.cache/torch_extensions/` |
| Inference path | yes | no |
| Benchmarks / HF patching | no | yes (via `wrapper.py`) |

If `ops.py` detects a JIT-loaded module, it raises:

```python
ImportError: Loaded JIT extension instead of AOT build. Run: bash scripts/build_kernels.sh rmsnorm
```

## Adding a new kernel

**1.** Create `kernel.cu` + `bindings.cpp` under `runtime/production_kernels/target/<op>/`.

**2.** Register in `setup.py`:

```python
KERNELS["my_op"] = {
    "module": "target_my_op_ops",
    "sources": ["kernel.cu", "bindings.cpp"],
}
```

**3.** Add `ops.py` + `__init__.py`:

```python
# ops.py template
def my_op_forward(...):
    return _load_ext().my_op_forward(...)
```

**4.** Build and test:

```bash
bash scripts/build_kernels.sh my_op
bash slurm/run_tests_gpu.sh runtime.tests.test_my_op
```

**5.** Wire into `executor.py` when the decoder loop needs the op.

See also `documentation/kernel_development_skill.md` for the partner workflow (CUDA authoring, micro-benchmarks). That doc covers the JIT + `wrapper.py` path; the runtime inference path uses AOT + `ops.py` instead.

## Current status

| Component | Status |
|-----------|--------|
| AOT build (`setup.py`, `build_kernels.sh`) | Done |
| `ops.py` for rmsnorm, embedding, residual_ops | Done |
| `ops.py` for swiglu, attention | Pending |
| GPU parity tests (rmsnorm, embedding, residual_ops) | Done |
| `kernel_set` in YAML / `RuntimeConfig` | Planned |
| `buffers.py`, `executor.py` | Planned (Phases 4–5) |

For the full roadmap, see `runtime/plan.md`.
