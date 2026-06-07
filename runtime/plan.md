---
name: Qwen2 Python Host + CUDA Kernels
overview: >-
  Python inference host (config, buffers, executor) calling pre-built C++/CUDA kernel
  extensions via thin ops.py wrappers. Phases 1–6 complete (HF parity on 7B).
  Phase 8 (speculative decoding): 7B target + 0.5B draft on separate GPUs via mpi4py.
  Next: Phase 8a target verify API, then MPI, CUDA graphs, draft kernels.
todos:
  - id: yaml-config-and-shapes
    content: Per-model YAML configs + RuntimeConfig loader + shape/memory helpers.
    status: completed
  - id: load-weights
    content: Load safetensors from model_path in YAML; validate shapes on device.
    status: completed
  - id: build-kernels-aot
    content: Add setup.py + scripts/build_kernels.sh — AOT-compile all target/ CUDA extensions (sm_75).
    status: completed
  - id: integrate-all-kernels
    content: Add ops.py in target/ for all five ops + GPU parity tests; drop runtime JIT (jit.py).
    status: completed
  - id: buffers-allocation
    content: Add runtime/buffers.py — KV cache, ping-pong hidden states, RoPE tables from plan_memory().
    status: completed
  - id: python-executor
    content: Add runtime/executor.py — layer loop, prefill() and decode_step() calling target/ kernel ops.
    status: completed
  - id: parity-harness
    content: Staged HF parity — layer → full logits → greedy decode (0.5B fast, 7B validate).
    status: completed
  - id: kernel-set-config
    content: Add kernel_set field to YAML + RuntimeConfig (default target); wire executor dynamic import.
    status: completed
  - id: update-readme
    content: Update runtime/README.md — Python host, AOT build, target/draft kernel layout.
    status: pending
  - id: cuda-graphs-decode
    content: "Phase 7/8b: CUDA Graph capture/replay for decode_step + fixed-γ verify_gamma. DONE for target (1.00x, memory-bound) and draft (1.64x, launch-bound). See graph_plan.md / draft_integration_plan.md."
    status: completed
  - id: spec-target-verify
    content: "Phase 8a: ForwardMode (PREFILL/VERIFY/DECODE), verify_gamma (small_q_attn), rollback_cache, p1_logits."
    status: completed
  - id: spec-host-sampler
    content: "Phase 8a: runtime/speculative/sampler.py — stochastic accept/reject per documentation/speculative_decoding.md."
    status: completed
  - id: spec-target-step
    content: "Phase 8a: target_step.py orchestrator + test_speculative_target.py."
    status: completed
  - id: spec-mpi
    content: "Phase 8c: mpi4py coordinator + slurm/run_speculative.sh (2 ranks, 1 GPU each)."
    status: pending
  - id: spec-draft
    content: "Phase 8d: Draft Qwen2Executor (kernel_set=draft) + DraftRunner γ-loop + draft CUDA graphs + single-process spec_decode. DONE — see draft_integration_plan.md (greedy spec == target greedy; end-to-end 1.18x)."
    status: completed
---

# Qwen2 Python Host + CUDA Kernels Plan

## Architecture Decision

**Python owns the host.** All orchestration — config, memory planning, weight loading, buffer allocation, decoder loop, `prefill()` / `decode_step()` — lives in Python under `runtime/core/` and a new `runtime/executor.py`.

**C++/CUDA owns compute.** Custom kernels stay as `kernel.cu` + `bindings.cpp` with pybind exports. They are **compiled ahead of time** (not JIT at import) and invoked from Python through a thin `ops.py` per op.

**Kernel sets by model role.** CUDA sources live under `production_kernels/<role>/` where `<role>` is `target` (main model) or `draft` (speculative draft model, future). The executor selects the kernel set from config — no flat `production_kernels/<op>/` namespace that would mix the two.

**Why Python host + AOT CUDA (not pure C++ host):** Writing a full inference loop in C++ often becomes a black hole of CMake and boilerplate. The PyTorch playbook — Python for orchestration, C++/CUDA for math — is the pragmatic choice given time constraints. Pre-built extensions + CUDA Graphs on decode (Phase 7) close most of the gap to a native C++ host without abandoning fast iteration.

**Explicitly abandoned:**
- C++ inference host (`runtime/host/`, CMake/LibTorch unified build, `qwen2_runtime` pybind module)
- Runtime JIT via `torch.utils.cpp_extension.load` in `jit.py` — compile once at build time instead

## Philosophy

This is a **scrappy research project**, not production code. Keep the runtime:

- **Config-driven** — one YAML per model, pass the path, everything else derives from it
- **Minimal files** — no enum frameworks, mock dispatch tables, or registry abstractions
- **Kernel-first** — CUDA ops under `production_kernels/<role>/`; Python host calls them via `ops.py`
- **Build vs run separated** — `scripts/build_kernels.sh` compiles extensions; inference imports prebuilt `.so` modules

## Target Topology

```mermaid
flowchart TD
    yaml[YAML config] --> cfg[RuntimeConfig Python]
    cfg --> shapes[shapes.py]
    cfg --> mem[plan_memory]
    cfg --> weights[load_weights_on_gpu]
    weights --> buffers[buffers.py]
    mem --> buffers
    buffers --> exec[executor.py Python]
    cfg -->|kernel_set target or draft| exec
    subgraph targetOps [production_kernels/target C++/CUDA]
        rmsnorm[rmsnorm]
        embed[embedding]
        attn[attention]
        swiglu[swiglu]
        residual[residual_ops]
    end
    subgraph draftOps [production_kernels/draft future]
        draftPlaceholder[draft ops TBD]
    end
    targetOps -->|"ops.py prebuilt ext"| exec
    draftOps -.-> exec
    exec --> logits[logits / decode]
```

## Runtime Layout

```
runtime/
├── core/                              # config, shapes, memory, weights (done)
├── buffers.py                         # Phase 4 — device buffer allocation ✅
├── executor.py                        # Phase 5 — decoder + model loop ✅
├── speculative/                       # Phase 8 — spec decode (sampler, MPI, orchestrator)
│   ├── sampler.py                     # host accept/reject + resample (CPU)
│   ├── target_step.py                 # target verify → sample → commit
│   ├── types.py                       # SpeculativeStepResult, wire format
│   └── mpi_coordinator.py             # mpi4py two-rank loop (Phase 8c)
├── production_kernels/
│   ├── target/                        # target (main) model kernels — KEEP HERE
│   │   ├── rmsnorm/
│   │   │   ├── kernel.cu
│   │   │   ├── bindings.cpp
│   │   │   └── ops.py                 # Python host ABI
│   │   ├── embedding/
│   │   ├── attention/
│   │   ├── swiglu/
│   │   └── residual_ops/
│   └── draft/                         # draft model kernels (future, same layout)
├── tests/
└── plan.md

scripts/
└── build_kernels.sh                   # AOT compile target/ (+ draft/ when present)

setup.py                               # CUDAExtension definitions for all ops
```

Partner dev artifacts removed — use AOT build + `runtime/tests/test_<op>.py` for correctness.

### YAML is the single source of truth

Each YAML contains:

- Model architecture dims (hidden, layers, heads, intermediate, vocab, rope, etc.)
- `model_path` — where safetensors live (resolved relative to project root)
- Runtime policy (`dtype: fp16`, `cuda_arch: sm_75`)
- Default inference limits (`max_batch`, `max_seq_len`)
- `kv_cache_layout` and `layer_order` (documentation + executor use)
- `kernel_set: target` (or `draft` when draft kernels exist) — which `production_kernels/<role>/` tree to use

Code derives `head_dim`, `kv_dim`, weight shapes, and buffer byte counts — nothing is duplicated.

### Usage

```python
from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.memory import plan_memory
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
weights, _ = load_weights_on_gpu(cfg, batch=1)
buffers = allocate_buffers(cfg, batch=1, max_seq_len=512, device="cuda")
executor = Qwen2Executor(cfg, weights, buffers)

logits = executor.prefill(input_ids)        # [B, S, vocab]
next_logits = executor.decode_step(token_id)  # [B, 1, vocab]
```

```bash
source setup.sh
bash scripts/build_kernels.sh          # once per code change / fresh checkout
bash slurm/run_tests_gpu.sh runtime.tests.test_executor.TestExecutorGpu
```

## What We Anchor To

- Reference forward order: `src/reference/modeling_qwen2.py`
- Reference algorithm: `documentation/speculative_decoding.md`
- Layer order in YAML: `input_rmsnorm → attention → residual_add → post_attn_rmsnorm → swiglu_mlp → residual_add`
- Target kernels: `runtime/production_kernels/target/` (all ops complete)
- GPU cluster: `.cursor/skills/gpu-cluster/` (SLURM, setup.sh, Turing FP16)

## Build Strategy: AOT, Not JIT

**JIT is not required.** Runtime JIT (`torch.utils.cpp_extension.load` at import) was removed in favor of AOT build.

**Preferred: ahead-of-time compile** via project-root `setup.py` + `scripts/build_kernels.sh`:

1. `setup.py` registers one `CUDAExtension` per op under `target/` with a **dotted module path** (e.g. `runtime.production_kernels.target.rmsnorm.target_rmsnorm_ops`)
2. `build_kernels.sh` sources `setup.sh` (conda + `gnu12`), runs `pip install -e . --no-build-isolation` or `python setup.py build_ext --inplace`
3. Each `ops.py` imports its prebuilt extension from the **same directory** via relative import:

```python
# production_kernels/target/rmsnorm/ops.py
from . import target_rmsnorm_ops as _ext

def forward(input, weight, eps):
    return _ext.forward(input, weight, eps)
```

Build output lands beside the kernel sources, e.g. `runtime/production_kernels/target/rmsnorm/target_rmsnorm_ops.cpython-311-x86_64-linux-gnu.so`.

4. Rebuild only when `kernel.cu` or `bindings.cpp` change; inference startup is instant import

**Why not CMake/LibTorch host build:** per-op `CUDAExtension` via setuptools keeps the Python host in Python while still giving prebuilt binaries. No yaml-cpp/LibTorch duplication.

## Current Status (Phases 1–6 complete)

| Phase | Status | Key artifacts |
|-------|--------|---------------|
| 1 Config + shapes | ✅ | `core/configs/*.yaml`, `RuntimeConfig`, `shapes.py`, `plan_memory()` |
| 2 Weights | ✅ | `load_weights()`, `vram_budget()`, `test_weights.py` |
| 3 Kernel integration | ✅ (3c pending) | `setup.py`, `build_kernels.sh`, five `target/<op>/ops.py`, GPU parity tests |
| 4 Buffers | ✅ | `buffers.py`, `test_buffers.py` (10/10) |
| 5 Executor | ✅ | `executor.py` (`prefill`, `decode_step`, `greedy_extend`, `run_decoder_layer`) |
| 6 Parity harness | ✅ (7B) | `parity_support.py`, `test_decoder_layer.py`, `test_parity_greedy.py`, `test_executor.py` |
| 7 Perf + CUDA graphs | 🔲 | Timing / tokens/sec baseline; graph decode (folded into 8b for verify) |
| 8 Speculative decoding | 🔲 8a done | 8a: verify + host sampler (no MPI); 8c–8d pending |

**Target kernels** (`runtime/production_kernels/target/`):

| Op | AOT `.so` | `ops.py` | GPU tests |
|----|-----------|----------|-----------|
| rmsnorm | ✅ | ✅ | `test_rmsnorm.py` |
| embedding | ✅ | ✅ | `test_embedding.py` |
| attention | ✅ | ✅ (6 sub-ops) | `test_attention.py` (8 tests, 7B / D=128) |
| swiglu | ✅ | ✅ | `test_swiglu.py` |
| residual_ops | ✅ | ✅ | `test_residual_ops.py` |

**GPU test suite on `gpu-turing`:** 37 tests — 25 per-op + 12 parity/executor (layer, greedy, full logits).

```bash
bash slurm/run_tests_gpu.sh runtime.tests.test_decoder_layer
bash slurm/run_tests_gpu.sh runtime.tests.test_parity_greedy
bash slurm/run_tests_gpu.sh runtime.tests.test_executor.TestExecutorGpu
```

**Still open:**
- **Phase 8a** — target `verify_gamma` + host stochastic sampler (can start now; mock draft payloads)
- `kernel_set: target` in YAML + `RuntimeConfig` (needed for Phase 8d draft)
- `runtime/README.md` refresh (buffers, executor, parity tests, spec decode, AOT layout)
- 0.5B draft blocked until `production_kernels/draft/` attention kernels (`head_dim=64`)
- Turing alignment validation in `RuntimeConfig.validate()` (Phase 1 checkbox)

Legacy `src/kernels/rmsnorm/` and per-op `jit.py` / `wrapper.py` / `benchmark.py` are removed; use AOT + `runtime/tests/` only.

## Kernel Host ABI (Python ↔ CUDA boundary)

Each op directory under `runtime/production_kernels/<role>/<op>/`:

```
<role>/<op>/
├── kernel.cu          # CUDA launchers (partner code, unchanged)
├── bindings.cpp       # pybind exports (unchanged)
└── ops.py             # Python host ABI — only file executor imports
```

**Standard single-op pattern** (RMSNorm, embedding, SwiGLU, residual_add, lm_head):

```python
# ops.py — imports prebuilt extension, no JIT
def forward(...) -> Tensor: ...
def workspace_bytes(...) -> int: ...   # 0 if none; optional helper
```

No `init()` required if extensions are prebuilt — import is sufficient. Keep `init()` only if lazy validation or one-time setup is needed.

**Attention is multi-op** — `ops.py` exposes the sub-ops in `target/attention/bindings.cpp`: `qkv_proj_forward`, `rope_kv_write_forward`, `fused_attn_forward` (prefill), `decode_attn_forward` (decode S=1), `small_q_attn_forward` (verify S∈[2,8]), `o_proj_forward`. The Python executor composes these; no monolithic `forward()`.

**Executor import pattern:**

```python
from runtime.production_kernels.target.rmsnorm import forward as rmsnorm_forward
from runtime.production_kernels.target.attention import ops as attn_ops
# future draft:
# from runtime.production_kernels.draft.rmsnorm import forward as draft_rmsnorm_forward
```

Or resolve dynamically from `cfg.kernel_set`.

### Bindings best practices (Phase 3)

Audit all `bindings.cpp` files when wiring `ops.py`:

- **Pass tensors by const reference** — accept `const torch::Tensor&`, not by value, to avoid atomic refcount bumps on every layer call
- **Release the GIL** — wrap the CUDA launch in `py::gil_scoped_release` inside each binding; cheap to add, prevents Python GC or background threads from stalling kernel launches

## Step-by-Step Plan

### Phase 1: YAML config + shape helpers ✅

- [x] `configs/qwen2.5-7b.yaml` and `configs/qwen2.5-0.5b.yaml`
- [x] `RuntimeConfig.from_yaml()` with derived properties
- [x] `shapes.py` and `plan_memory()` — plain functions, no class hierarchies
- [x] Tests in `runtime/tests/test_config.py` and `runtime/tests/test_shapes.py`
- [ ] **Turing alignment checks** — in `shapes.py` or `RuntimeConfig.validate()`, warn/assert that matmul inner dims (hidden, intermediate, head_dim) are multiples of 8 (prefer 16/32) so FP16 Tensor Cores on `sm_75` are not silently bypassed; Qwen2.5 vocab (152064) is already aligned — watch custom seq lengths and any pruning

### Phase 2: Weight loading ✅

- [x] `load_weights(cfg, device)` reads sharded or single safetensors from `cfg.model_path`
- [x] `validate_weights()` checks every expected HF key + shape (incl. q/k/v biases)
- [x] Casts to FP16 per YAML dtype policy; `startup_report()` for memory summary
- [x] Works for both models via whichever YAML is passed in
- [x] Tests in `runtime/tests/test_weights.py` (GPU tests via `bash slurm/run_tests_gpu.sh`)
- [x] `vram_budget()` / `max_seq_len_after_weights()` — compute max seq len from remaining HBM after weights

### Phase 3: Kernel integration (full) — **completed** (3c pending)

**3a. AOT build system** ✅

- [x] Root `setup.py` with dotted `CUDAExtension` per target op (`-arch=sm_75`, `-O3`)
- [x] `scripts/build_kernels.sh` (sources `setup.sh`, `BUILD_KERNEL=all|rmsnorm|…`)
- [x] `.so` colocated in each `target/<op>/` via `build_ext --inplace`
- [x] Removed `jit.py`, `wrapper.py`, `benchmark.py` from all `target/<op>/`

**3b. ops.py + parity tests** ✅ — 25/25 GPU tests on `gpu-turing`

| Op | extension module | Parity test |
|----|------------------|-------------|
| rmsnorm | `target_rmsnorm_ops` | `test_rmsnorm.py` |
| embedding | `target_embedding_ops` | `test_embedding.py` |
| swiglu | `target_swiglu_ops` | `test_swiglu.py` |
| residual_ops | `target_residual_ops` | `test_residual_ops.py` |
| attention | `target_attention_ops` | `test_attention.py` (7B only — kernels templated on `head_dim=128`) |

**3c. Config hook for kernel set** — **pending**

- [ ] Add `kernel_set: target` to YAML configs; expose on `RuntimeConfig`
- [x] Executor accepts `kernel_set=` kwarg but only `"target"` is wired today

**3d. Bindings audit** ✅

- [x] All `target/*/bindings.cpp`: `const torch::Tensor&` + `py::gil_scoped_release`

### Phase 4: Buffer allocation — **completed**

`runtime/buffers.py`:

- `RuntimeBuffers` dataclass + `allocate_buffers(cfg, batch, max_seq_len, device)` sized by `plan_memory()`
- KV cache: layer-major `[L, B, Hkv, S_max, D]`; row strides 16-byte aligned via `kv_cache_head_dim()`
- Ping-pong `hidden_a` / `hidden_b`; Q/K/V scratch; MLP gate; logits; RoPE cos/sin tables; device `cache_position` int64 scalar
- `build_rope_tables()`, `rope_embeddings()`, `swap_hidden()`, `kv_cache_*_layer()`, `memory_report()`, `buffer_fits_vram_budget()`
- `plan_memory()` extended for `rope_cos`, `rope_sin`, fixed `cache_position` bytes
- Tests: `runtime/tests/test_buffers.py` (10/10 CPU+GPU); `test_memory` linear-scaling fix for fixed overhead

### Phase 5: Python decoder + model executor — **completed**

`runtime/executor.py`:

- `Qwen2Executor(cfg, weights, buffers)` — loops `cfg.layer_order`, imports `production_kernels/target/` ops
- `prefill(input_ids)` → `[B, S, vocab]`; resets KV cache; fused attention + RoPE on Q/K
- `decode_step(token_id)` → `[B, 1, vocab]`; decode attention path; host `_cache_pos` mirror + device `cache_position` `fill_`/`add_` (no `.item()` in loop)
- Pre-stacked QKV weights per layer; `head_dim==128` gate (7B attention kernels)
- `greedy_extend(input_ids, n_new)` — correct greedy loop (prefill argmax, then `decode_step`)
- Tests: `runtime/tests/test_executor.py` — stages 3–4 (6 tests total with CPU structure)

### Phase 6: Parity harness — **completed** (7B)

Shared helpers in `runtime/tests/parity_support.py` (`load_hf_and_executor`, `capture_decoder_layer`, `greedy_decode_*`, tolerance constants).

| Stage | Status | Where |
|-------|--------|-------|
| 1. Per-op | ✅ | `test_rmsnorm.py` … `test_attention.py` (25 tests) |
| 2. Single decoder layer | ✅ | `test_decoder_layer.py` — layers 0, mid, last; HF hook vs `run_decoder_layer` |
| 3. Full model logits | ✅ | `test_executor.py` — prefill + decode_step vs HF |
| 4. Greedy decode trajectory | ✅ | `test_parity_greedy.py` — 8- and 16-token trajectories vs `generate()` |

**GPU test notes** (see `.cursor/skills/gpu-cluster/SKILL.md`):

- Run via `bash slurm/run_tests_gpu.sh <module>` — partition max **30 min**
- Parity tests: load **one** HF model copy on GPU (not safetensors + HF — OOM on 24GB)
- Compare logits/hidden in `.float()` (HF fp32 vs kernel fp16)
- Greedy: first new token from **prefill logits**; only later tokens use `decode_step`

### Phase 8: Speculative decoding — **pending**

**Goal:** Accelerate 7B (target) inference using 0.5B (draft) on a **second GPU**, coordinated via **mpi4py**. Algorithm: [documentation/speculative_decoding.md](../documentation/speculative_decoding.md). Partner owns `production_kernels/draft/` kernels; **target-side work can start now** with mock draft payloads.

**Topology:** Rank 0 → 7B target on GPU 0; Rank 1 → 0.5B draft on GPU 1 (`srun --ntasks=2 --gres=gpu:1`). One model copy per GPU (no OOM).

**Design decisions (agreed + corrections):**

| Topic | Decision |
|-------|----------|
| Target logits for verify | Need **γ+1** distributions `p_1…p_{γ+1}`: `p_1` = last prefill logit; `verify_gamma` returns `[B, γ, vocab]` = `p_2…p_{γ+1}` |
| Verify attention | Use **`small_q_attn_forward`** (S∈[2,8], **γ≤7**), not `fused_attn_forward`, after cached prefill |
| Cache rollback | Set `_cache_pos` / device scalar to `prefix_len + n`; stale KV in rejected slots is **logically dead** (optional zero for debug) |
| **Bonus handling** | Sample bonus after verify; **defer** (no target forward on bonus until next iter). Next iter prepends bonus + runs one VERIFY on ``[bonus, d_1…d_γ]``; accept/reject **drafts only**. ``flush_pending_bonus()`` after final iter. |
| Sampling | **Stochastic** accept/reject on **CPU** (D2H target logits, softmax); no sampling in CUDA kernels |
| Forward modes | **Three explicit stages** — do not conflate VERIFY (S=γ) with DECODE (S=1); see below |
| Module layout | Extend `executor.py` for forward ops; add `runtime/speculative/` for sampler, orchestrator, MPI |
| CUDA graphs | **One fixed-γ VERIFY graph** at S=γ+1 with ``leading_bonus_valid`` masking (see below). Optional prefill graph; S=1 flush after last iter only. **Never graph:** sampling, MPI, rollback |
| Execution order | **8a** eager verify + sampler → **8c** MPI → **8b** CUDA graphs → **8d** draft kernels |

#### Three forward stages (not just prefill vs decode)

The executor today uses a boolean `decode` flag; speculative decoding needs a **third, explicitly labeled stage**. Do not treat `verify_gamma` as “decode with a longer seq len” — it is a **verification forward** with different attention semantics and cache behavior.

| Stage | `ForwardMode` | Query len S | Attention kernel | KV / cache | Public API |
|-------|---------------|-------------|------------------|------------|------------|
| Prefill | `PREFILL` | prompt len | `fused_attn_forward` | Reset KV; write `[0, S)` | `prefill(input_ids)` |
| **Verify** | **`VERIFY`** | **γ** (1st iter) or **γ+1** (later) | **`small_q_attn_forward`** | Speculative KV write; may rollback | **`verify_gamma(draft_ids, leading_bonus=…)`** |
| Decode (flush) | `DECODE` | **1** | **`decode_attn_forward`** | Final iteration only | **`flush_pending_bonus()`** |

Implementation: replace `decode: bool` in `_run_attention` / `_forward_stack` with `mode: ForwardMode`. Logs, tests, and CUDA graph capture keys should name the stage (`VERIFY`, not `decode=True`).

```python
class ForwardMode(Enum):
    PREFILL = "prefill"   # fused_attn, reset KV
    VERIFY = "verify"     # small_q_attn, S=γ, speculative KV write
    DECODE = "decode"     # decode_attn, S=1, commit token
```

#### Host sampling and MPI (what crosses the wire)

**All accept/reject/resample math runs on the target rank’s CPU** — not in CUDA kernels. Reasons:

1. Algorithm needs RNG, ratio tests, and `norm(max(0, p − q))` (see spec doc).
2. Draft `q` distributions must be on the target to compare against target `p`.

**D2H on target is for local sampling, not because target logits are MPI’d back to draft:**

| Direction | Payload | Where used |
|-----------|---------|------------|
| Draft → target (MPI) | `draft_token_ids[γ]`, `draft_logits[γ+1, vocab]` (fp16) | Target host compares `q` vs `p` |
| Target GPU → target CPU (D2H) | `p_1…p_{γ+1}` after `prefill` + `verify_gamma` | Target host sampler only |
| Target → draft (MPI) | `n_accepted`, `bonus_token`, `prefix_len`, `cache_pos_after` (tiny) | Draft sync + append deferred bonus for γ gen |

Target never MPIs its full `p` matrix to the draft. Draft logits arrive via MPI already; target logits are copied to host for **local** stochastic verification. A GPU sampler would not remove the need for draft `q` on target — it would only avoid D2H of `p`, at the cost of a custom kernel we explicitly avoid.

#### Speculative iteration protocol (target + draft)

Both ranks **prefill** the same prompt of length `L` once. Every iteration the draft MPI payload is exactly **γ tokens** (bonus is never counted toward γ).

| Step | Target | Draft |
|------|--------|-------|
| **1 — first iter** | `prefill(L)`; recv γ draft ids + `q` logits | `prefill(L)`; autoregress γ drafts from end of prefix; send payload |
| **2 — verify + sample** | `verify_gamma(γ drafts)` (S=γ); accept `n` of γ; sample bonus `t`; rollback; **store `t`, no forward on `t`** | (waiting) |
| **3 — sync** | MPI send `n_accepted`, `bonus_token`, `prefix_len`, `cache_pos_after` | Recv; append accepted + `t` to prefix; **run γ draft steps continuing from `t`** |
| **4 — later iters** | Recv γ drafts; `take_pending_bonus()` → `verify_gamma(γ, leading_bonus=t_prev)` one forward on `[t_prev, d_1…d_γ]`; accept/reject **from `d_1` only** (`t_prev` not re-sampled) | Send γ drafts (not including `t_prev`) |
| **5 — end** | `flush_pending_bonus()` once (last deferred bonus) | mirror target prefix |

Eager path today uses S=γ on step 2 first iter and S=γ+1 on step 4 — logically identical to one masked graph (below).

#### Single CUDA VERIFY graph (Phase 8b — masking, not two graphs)

Capture **one** graph at fixed query length **S = γ+1** (`MAX_VERIFY_SEQ_LEN = 8`):

| Device input | Role |
|--------------|------|
| `verify_input_ids[γ+1]` | slot 0 = leading bonus id (don't-care when invalid); slots 1…γ = draft ids |
| `leading_bonus_valid` (0/1) | 0 = first post-prefill iter; 1 = prepend bonus |
| `p1_logits[vocab]` | used by host sampler row 0 when `leading_bonus_valid=0` |

Kernel / executor behavior when **`leading_bonus_valid=0`**: skip KV write for slot 0 (or mask first query); run effective verify over γ draft queries; sampler uses saved **`p1_logits`** + logits rows for drafts (same as eager first iter).

When **`leading_bonus_valid=1`**: full S=γ+1 forward; slot 0 commits deferred bonus; sampler uses logits rows 0…γ for `p_1…p_{γ+1}` on **γ drafts only** (`speculative_acceptance` never sees the leading bonus id).

No second graph for “first iter vs later” — only the scalar mask and sampler row selection differ.

#### Phase 8a — Target verify API + host stochastic sampler — **completed** (no MPI)

**Goal:** Target-only correctness without MPI; mock draft tokens/logits passed directly.

**Delivered:**

- `ForwardMode` enum in `runtime/speculative/types.py` — PREFILL / VERIFY / DECODE
- `runtime/executor.py`: `verify_gamma(..., leading_bonus=)`, `take_pending_bonus`, `defer_bonus_token`, `flush` via `commit_pending_bonus`
- `runtime/speculative/target_step.py` — protocol in module docstring; `had_leading_bonus` on result
- Tests: `test_speculative_sampler.py` (CPU, dummy logits); `test_speculative_target.py` (GPU verify vs HF, rollback, deferred bonus, dummy-q step); `TestNoMpiInPhase8a` asserts no mpi4py in speculative package

```bash
python -m unittest runtime.tests.test_speculative_sampler -v
bash slurm/run_tests_gpu.sh runtime.tests.test_speculative_target
```

#### Phase 8b — CUDA graphs — **pending** (after 8c eager)

Graph **GPU forward paths only** — never sampling, MPI, or cache rollback.

| Priority | Graph captures | Rationale |
|----------|----------------|-----------|
| **Primary** | **Fixed S=γ+1 VERIFY** + `leading_bonus_valid` mask | One graph for all post-prefill iters |
| **Optional flush** | **`flush_pending_bonus`** (S=1) | Once after final iteration only |
| **Optional** | **`prefill`** (`ForwardMode.PREFILL`) | Runs once per prompt; variable length → bucketed graphs or stay eager |
| **Never** | Host sampler, MPI, `rollback_cache` | CPU / control flow |

Flag: `use_cuda_graph=True, graph_gamma=4` — capture **VERIFY** graph only when `len(draft_ids) == graph_gamma`. Implement after MPI eager path is parity-checked (Phase 8c).

#### Phase 8c — MPI coordinator — **pending**

- `runtime/speculative/mpi_coordinator.py` + `slurm/run_speculative.sh`
- **Draft → target:** `draft_token_ids[γ]`, `draft_logits[γ+1, vocab]` (fp16)
- **Target → draft:** `n_accepted`, `bonus_token`, `prefix_len`, `cache_pos_after` — send **immediately after defer**
- **Target loop:** `take_pending_bonus()` → `verify_gamma(γ, leading_bonus=?)` → sample (drafts only) → rollback → defer → MPI
- **End of generation:** `flush_pending_bonus()` on target (only S=1 in entire run is here + optional prefill)

#### Phase 8d — Draft executor + kernels — **pending** (partner)

When `production_kernels/draft/` lands:

- AOT build draft ops in `setup.py` / `build_kernels.sh` (`head_dim=64`)
- `Qwen2Executor(kernel_set="draft")` — mirror target API: `prefill`, `decode_step`, γ sequential draft loop
- Draft rollback: same `_cache_pos` rule after target returns `n_accepted`; apply bonus to draft prefix on MPI recv (draft `decode_step` on bonus while target idle)
- Completes Phase 3c `kernel_set` YAML hook

**Phase 8 risks:** γ+1 logit indexing bugs (explicit unit test); verify kernel must HF-parity before MPI; keep single-process `target_speculative_step` as golden path for debugging.

### Phase 7: Performance + profiling — **pending**

Only after Phase 6 is stable:

- Timing hooks around embedding, per-layer, lm_head
- Compare tokens/sec vs HF eager baseline on Turing (`gpu-turing`, batch=1)
- Optional: lightweight profiler script in `runtime/tests/` (not in `production_kernels/`)

**CUDA Graphs (Phase 7 + 8b)**

Python launch overhead is the biggest threat to this architecture. Each GPU forward runs ~30 layers × 5+ ops ≈ **150+ kernel launches from Python**, each incurring ~10–50µs pybind/interpret overhead.

Because Phase 4 uses static buffers (fixed addresses, no `torch.empty`), capture repeatable forwards into **CUDA Graphs** and replay:

- **Spec-decode:** graph **`ForwardMode.VERIFY`** (`verify_gamma`) first — highest ROI in the spec loop
- **Commit:** bundled in masked **VERIFY** graph at S=γ+1; not a separate autoregressive step mid-loop
- **Prefill:** optional / bucketed — lower ROI (once per prompt, variable length)
- **Do not graph:** host sampling, MPI, rollback

Requirements:

- Static tensor addresses and kernel arguments across replay steps
- `cache_position` updated on device — graph reads device scalar; no host int per replay
- Draft token ids: write into fixed device buffer slot(s) before VERIFY graph replay
- Warmup: one eager forward per mode before capture; then `torch.cuda.graph()` + `replay()`
- Label graph capture by **`ForwardMode`** (VERIFY ≠ DECODE)

Implement behind `use_cuda_graph=True` once eager spec-decode parity passes (Phase 8a/8c).

## Cleanup

- [x] Removed per-op `jit.py`, `wrapper.py`, `benchmark.py` under `target/`
- [ ] Remove leftover `runtime/host/build/` artifacts if still on disk
- [ ] Update `runtime/README.md` — buffers, executor, AOT build, `production_kernels/target/` layout

## Deferred (add only when needed)

- Pluggable execution policies / op registries
- CMake unified build or LibTorch C++ host

(Multi-GPU speculative decoding is **active** — Phase 8 above. Draft kernels: `production_kernels/draft/`.)

## Immediate Next Actions

1. **Phase 8c:** mpi4py coordinator + `slurm/run_speculative.sh` (mock draft first)
2. **Phase 3c:** `kernel_set: target|draft` in YAML + `RuntimeConfig`
3. **Phase 8b / 7:** CUDA graphs for fixed-γ `verify_gamma` + `decode_step` commit
4. **Phase 8d:** draft executor when `production_kernels/draft/` lands
5. **README:** update `runtime/README.md` with spec decode layout and test commands
