# System requirements and known build issues

Field notes from getting the first CUDA extension (`kernel_dev/target/kernels/embedding/`) to
build and run on Stanford ICME's `gpu-turing` partition (Quadro RTX 6000, SM 7.5,
nvcc 12.3 driver). Most of this is *not* about CUDA itself — it's about
PyTorch's JIT extension build colliding with the cluster's modulefile
environment. Read this before debugging from scratch.

---

## Required environment

### Lmod modules (load once per shell, or pin in `.bashrc`)

| Module | Why | Provides |
|---|---|---|
| `cuda/12.2` | Stock CUDA Toolkit for nvcc + headers. | `nvcc`, `cudart`, `<cuda_runtime.h>` |
| `course/cme213/nvhpc/24.1` | Course MPI / nvc++ for later assignments. *Can stay loaded* — `kernel_dev/target/kernels/*/jit.py` sanitizes its env-var pollution at build time. | `mpicc`, `nvc++`, NVHPC libs |
| `gnu12/12.3.0` | PyTorch 2.4 headers require **GCC ≥ 9**. System `/usr/bin/g++` is 8.5 → fails build with `#error "We need GCC 9 or later"`. | `g++ 12.3.0` |

```bash
module load gnu12/12.3.0     # cuda/12.2 + course module already loaded by default
module list                  # confirm all three appear
```

### Conda env

```bash
conda activate cme213        # python 3.11, torch 2.4.0, transformers 4.45.0
```

### One-time pip installs

```bash
pip install ninja            # see Bug 1 below
```

---

## Bug log — what we hit and how it's fixed

### Bug 1 — `RuntimeError: Ninja is required to load C++ extensions`

**Cause.** `torch.utils.cpp_extension.load()` (the JIT path) hard-requires
`ninja` as its build driver. Unlike `setup.py build_ext`, there is no
distutils fallback.

**Fix.** `pip install ninja`. ~200 KB, single binary.

---

### Bug 2 — `nvcc fatal : Unsupported NVHPC compiler found. nvc++ is the only NVHPC compiler that is supported.`

**Cause.** The `course/cme213/nvhpc/24.1` module prepends NVHPC's nvcc to
`PATH` (`/home/cme213/software/nvidia-hpc-sdk/2024_24.1/.../bin/nvcc`).
NVIDIA HPC SDK's nvcc *refuses every host compiler except nvc++*, and nvc++
is ABI-incompatible with the PyTorch wheel (built with g++ + libstdc++
old-ABI). Either of `setup.py build_ext` or the JIT path will pick this
nvcc and immediately error out.

**Fix.** Strip any `PATH` entry containing `nvidia-hpc-sdk` or `/nvhpc/`
before invoking the build. Implemented in `kernel_dev/target/kernels/embedding/jit.py:
_sanitize_env_for_nvcc()`.

---

### Bug 3 — NVHPC env vars beat `PATH`-based discovery

**Cause.** PATH-stripping alone wasn't enough. The course module also sets
`CC=nvc`, `CXX=nvc++`, `CUDA_HOME=<nvhpc cuda dir>`. PyTorch reads those
env vars directly and passes them through, e.g. `nvcc -ccbin /path/to/nvc`,
which then triggers Bug 2 again.

**Fix.** In `_sanitize_env_for_nvcc()`, also `del os.environ[var]` for
`CC`, `CXX`, `CUDA_HOME`, `CUDA_PATH`, `NVCC_CCBIN` — but only when the
value contains an NVHPC marker, so the function is safe whether or not the
course module is loaded.

---

### Bug 4 — `error: "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."`

**Cause.** Once NVHPC was out of the picture, distutils fell back to
`/usr/bin/g++`, which is **GCC 8.5.0** on this cluster's RHEL/Rocky 8 base
image. PyTorch 2.4's `c10/util/C++17.h` has a `__GNUC__` version check and
rejects anything below 9.

**Fix.** `module load gnu12/12.3.0` puts GCC 12.3 first on `PATH`. Once
`CXX` is unset (per Bug 3 fix), distutils picks it up automatically.

---

### Bug 5 — `nvcc` *still* resolved to NVHPC after env scrubbing

**Cause.** `torch.utils.cpp_extension` caches `CUDA_HOME` at **module-import
time** via a single call to `_find_cuda_home()`. Its top-level statement:

```python
CUDA_HOME = _find_cuda_home() if torch.cuda.is_available() else None
```

If `jit.py` did `from torch.utils.cpp_extension import load` *before*
sanitizing the env, that import call locked in NVHPC's nvcc path. Calling
`_sanitize_env_for_nvcc()` inside `load_embedding_ops()` afterward was too
late — torch never re-reads `CUDA_HOME`.

**Fix.** Two changes in `jit.py`:

1. Move `_sanitize_env_for_nvcc()` to **module-import time, before** the
   `from torch.utils.cpp_extension import load` line.
2. After cleaning `PATH`, explicitly set `os.environ["CUDA_HOME"]` by
   resolving `shutil.which("nvcc")` and taking its grandparent dir, so
   `_find_cuda_home()` returns a sane value even if its other fallback
   paths (e.g. `/usr/local/cuda`) point somewhere unexpected.

---

### Bug 6 — `identifier "C10_CUDA_KERNEL_LAUNCH_CHECK" is undefined`

**Cause.** This macro lives in `<c10/cuda/CUDAException.h>`, which
`<torch/extension.h>` does *not* transitively include. Easy to miss
because the rest of c10 is pulled in.

**Fix.** Add `#include <c10/cuda/CUDAException.h>` explicitly in
`kernel.cu`. (Alternative: replace with a hand-rolled
`cudaGetLastError()` + `TORCH_CHECK`. Same outcome, one less include.)

---

### Bug 7 — `RuntimeError: weight must be float16` against Qwen2.5 safetensors

**Cause.** Qwen2.5-7B-Instruct ships as **bfloat16** on disk
(`config.json: "torch_dtype": "bfloat16"`). Loading weights directly via
`safetensors.safe_open(...)` (as `benchmarks/correctness.py:load_weight`
does) returns `torch.bfloat16` tensors — *not* fp16.

The rest of the project assumes fp16 (`src/inference/autoregressive.py`,
`scripts/verify_env.py`, both pass `torch_dtype=torch.float16` to
`AutoModelForCausalLM.from_pretrained`, which *casts on load*). The
embedding kernel's `TORCH_CHECK(weight.scalar_type() == torch::kHalf)`
catches the mismatch.

**Fix.** Cast in user code:

```python
weight = load_weight(MODEL_PATH, EMBEDDING_WEIGHT_NAME, device=device)
if weight.dtype != torch.float16:
    weight = weight.to(torch.float16)
```

**Open question.** Whether to cast on every load or save an fp16-converted
copy of the shards once. The latter halves nothing in terms of disk (both
are 2 bytes/param) but eliminates the runtime cast and the bf16/fp16
footgun across all consumers. See "Convert weights to fp16 once" below if
you want to go that route.

---

## Self-test for a fresh shell

```bash
module load gnu12/12.3.0
conda activate cme213
which nvcc                                      # /opt/ohpc/.../cuda/12.2/bin/nvcc
which g++                                       # /opt/ohpc/.../gnu12/.../bin/g++
g++ --version | head -1                         # gcc (...) 12.3.0
python -c "import torch; print(torch.cuda.is_available())"  # True on a GPU node

# JIT-build + correctness check
rm -rf ~/.cache/torch_extensions/py311_cu121/qwen_embedding_kernel   # if a previous build failed
srun --partition=gpu-turing --gres=gpu:1 python kernel_dev/target/kernels/embedding/test.py
# Expected last line: PASS
```

If `nvcc` resolves under `nvidia-hpc-sdk/`, the env sanitization in
`jit.py` didn't fire in time — likely because some module imported
`torch.utils.cpp_extension` before `jit.py` ran. Check the import chain.

---

## Optional: convert weights to fp16 once

If the bf16→fp16 cast at load time becomes a pain across many kernels,
re-save the shards in fp16 form. One-time conversion script
(rough sketch, not committed):

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    "models/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)
m.save_pretrained("models/Qwen2.5-7B-Instruct-fp16", safe_serialization=True)
```

Trade-off: ~14 GB additional disk, but every downstream load (safetensors
or HF) now returns fp16 directly. `MODEL_PATH` would need to point at the
new dir.
