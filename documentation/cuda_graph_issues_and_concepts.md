# CUDA Graph Issues & Concepts — Implementation Journal

A running log of concrete problems hit while adding CUDA graphs to `runtime/`, each written up
with **the concept you need to understand it**, the **symptom**, the **diagnosis**, and the **fix**.

This is the chronological "what actually went wrong and why" companion to:
- `documentation/cuda_graphs_explained.md` — the upfront mental model (read that first), and
- `graph_plan.md` — the phase-by-phase TODO list.

Newest issues at the bottom.

---

## Issue #1 — Custom kernels launched on the default stream (capture recorded nothing)

**Phase:** 1 · **Status:** fixed & validated (2026-06-07) · **Severity:** blocker for *all* graph work

### The concept: streams, and what "capture" actually listens to

A **CUDA stream** is an ordered queue of GPU work. When the CPU "launches a kernel," it really
*enqueues* that kernel onto a stream; the GPU runs the stream's items in order, asynchronously from
the CPU. A program can have many streams; work on different streams may overlap.

There is a special default stream (**stream 0**, the "legacy"/"null" stream). If you launch a CUDA
kernel in C++ as `my_kernel<<<grid, block>>>(...)` **without** naming a stream, it goes to **stream
0**.

**CUDA graph capture works by recording one specific stream.** `torch.cuda.graph(g)` creates a
fresh, *non-default* stream, switches PyTorch's "current stream" to it, flips that stream into
*capture mode*, runs your Python forward, and stops. While a stream is capturing, every kernel
**enqueued onto that stream** is *recorded into the graph* instead of being run now. Work sent to a
**different** stream is **not** part of the recording — and, by CUDA's rules, doing ordinary work on
the legacy default stream *while another stream is capturing is illegal*.

So there's an ironclad requirement: **every kernel that should be part of the graph must be launched
onto the stream that is currently capturing** — i.e. PyTorch's *current* stream, obtained in C++ via
`at::cuda::getCurrentCUDAStream()`. PyTorch's own ops and cuBLAS already do this. Hand-written
`<<<...>>>` launches do **not**, unless you pass the stream explicitly.

### The symptom

Capturing the decode forward and replaying it produced **garbage that grew with each replay**, and
eventually **NaN** — even with identical inputs at a fixed sequence position (so positions/inputs
were *not* the cause). A localization probe (`runtime/benchmarks/phase0_graph_diag.py`) captured
progressively larger chunks:

| Captured | Replay result (before fix) |
|---|---|
| single `rmsnorm` | "matched" eager — but PyTorch warned **`CUDA Graph is empty… wrong device or stream`** |
| one decoder layer | replay diverged and **grew**: −2.4 → −1.6 → −34 → −47 |
| full stack / forward | **NaN** |

The **"CUDA Graph is empty"** warning was the giveaway: the recording contained (almost) nothing.

### The diagnosis

Every one of our custom kernels launched with a bare `<<<grid, block>>>` — no stream argument — so
they all ran on **stream 0**, never on the capture stream. During "capture":

- the kernels executed for real on stream 0 (so the output tensor *looked* right immediately after
  capture — a red herring), but
- **nothing they did was recorded** into the graph.

Only PyTorch's own tensor ops inside the forward (the `q/k/v` `.transpose().contiguous()`, reshapes)
use the current stream, so *those* got recorded. On `replay()`, the graph re-ran just those copy ops
over intermediate buffers that the (unrecorded) compute kernels were supposed to fill — reading
stale/uninitialized pool memory. As that pool memory churned between replays, the error grew; over
28 layers it became NaN. The single `rmsnorm` "passed" only because its graph was *entirely empty*,
so `replay()` was a no-op and the output kept its capture-time value.

Grep confirming the root cause — none of the launch sites named a stream:

```
embedding/kernel.cu:67     rmsnorm/kernel.cu:155     residual_ops/kernel.cu:92
attention/kernel.cu:200,593,829     swiglu/kernel.cu:136
```

Note this also kills the previous attempt's theory that `torch::empty` allocations break replay.
They don't: the lone op that *was* exercised cleanly through the caching allocator (rmsnorm, once
fixed) replays bit-exactly. Allocations were never the problem; streams were.

### The fix

Launch every kernel on the current stream by passing it as the 4th `<<<>>>` argument (the 3rd is
dynamic shared-memory bytes, here `0`):

```cpp
// before
my_kernel<<<grid, block>>>(...);
// after
my_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);
```

Applied to all 7 launch sites above; added `#include <ATen/cuda/CUDAContext.h>` to `embedding` and
`rmsnorm` (the other three kernels already included it). This is **harmless in eager mode** — when
nothing is capturing, the "current stream" *is* the default stream, so behavior is unchanged.

### Validation

After rebuild, re-running the diagnostic: **no "empty graph" warning**, and every stage's
`replay()` matched eager **exactly** (full forward: min −9.133 / max +9.867, identical to eager).
This both fixes capture and confirms the allocation hypothesis.

### Concept follow-up: "capture records, it does not execute"

After the fix, a subtle and initially confusing detail appeared: immediately *after* capture but
*before* the first `replay()`, the output tensor read **all zeros**.

That is correct and worth internalizing. Once the kernels are properly recorded (not run) during
capture, the captured operations don't produce their results until the graph is **replayed**. The
output buffer just holds whatever was in that pool slot at capture time (zeros). **A captured
graph's outputs are only meaningful after `replay()`.**

(Before the stream fix, this same buffer held *real* values right after capture — precisely
*because* the kernels were wrongly executing on stream 0 instead of being recorded. So "real-looking
output straight after capture" was itself a symptom of the bug.)

Practical consequence: any test or benchmark must call `replay()` before reading a graph's output,
and must not treat the post-capture/pre-replay buffer as valid. (`phase0_graph_probe.py` was updated
to judge correctness only after replay.)

---

## Concept #2 — Device-scalar position arguments (un-freezing the graph)

**Phase:** 2 · **Status:** done & validated (2026-06-07) · this is a *design requirement*, not a bug

### The concept: a captured scalar argument is frozen forever

From `cuda_graphs_explained.md` §1.5: a CUDA graph records the **exact argument values** of each
kernel launch, including plain scalars. Our attention kernels took the sequence position as **host
`int`s**:

- `rope_kv_write_forward(..., int64_t write_pos, ...)` — *where* in the KV cache to scatter the new
  K/V (which row).
- `decode_attn_forward(..., int64_t cur_len, ...)` / `small_q_attn_forward(...)` — *how many* cached
  positions to attend over (and, via `q_pos_base = cur_len - S`, the query's absolute position for
  causal masking + RoPE).

If we capture a decode at sequence position `p`, the graph bakes in `write_pos = p` and
`cur_len = p+1`. Replaying it for the *next* token would still scatter K/V into row `p` (clobbering
it) and attend over only `p+1` positions — i.e. the graph is **frozen at the capture-time
position**. That is exactly the "growing 614 → 3978" error the old handoff saw once streams were
(accidentally) working: a position-frozen replay. Phase 0's probe avoided it only by holding the
position *fixed*.

### The fix: read the position from device memory

The cure follows §1.5 rule 3: a value that must change per step can't be a host scalar argument; it
must live in **device memory at a fixed address**, so we update its *contents* before each replay
without changing the recorded *argument* (which becomes just a pointer).

So each kernel gained an **optional device-scalar override**: a `const int64_t* ..._ptr` parameter.
When it's null, the kernel uses the host int (old eager behavior, untouched). When it's non-null,
the kernel reads the position from that 0-d int64 CUDA tensor on the device:

```cpp
// in the kernel, before the value is used:
if (cur_len_ptr) cur_len = static_cast<int>(*cur_len_ptr);
```

New host-facing ops expose this: `rope_kv_write_forward_dev`, `decode_attn_forward_dev`,
`small_q_attn_forward_dev`. Internally one launcher/dispatch serves both paths (host int **or**
device pointer), so there's no duplicated kernel body or dispatch table. The runtime already
maintains exactly such a device scalar — `buffers.cache_position`, advanced with `fill_`/`add_`
without a `.item()` sync — which the graph path will point these kernels at (Phase 3).

### Subtlety: you lose host-side bound checks

The eager path asserts `cur_len <= max_seq` and `write_pos + S <= max_seq` on the host. The `_dev`
path **cannot** — reading the device scalar on the host would require a `cudaStreamSynchronize`,
which is illegal during graph capture and defeats the point. So the `_dev` ops validate only the
*tensor* (must be a 0-d int64 CUDA scalar) and shift responsibility for the position *value* to the
caller. This is the general pattern with device scalars: **shape/type checks stay on the host;
value-range checks move to the caller's contract** (or a device-side assert, which we avoid here).

### Why `S` (query length) stays a host value

Only the *position* is a device scalar. The query length `S` (1 for decode, γ+1 for verify) stays a
host int because it's a **compile-time/shape** quantity: it selects the kernel template
(`decode_attn_kernel<Q_TOKENS=S>`) and the launch grid, both of which are *fixed across replays* for
a given graph. Things that change per step → device memory; things that are constant for the graph's
lifetime → fine as host values baked in at capture.

### Validation

`runtime/tests/test_attention_dev_scalar.py`: for decode (S=1), small_q (S∈{2,4,5}), and
rope_kv_write, the `_dev` op equals the host-int op **bit-exactly** (`torch.equal`) across several
positions — with and without RoPE — and the `_dev` ops reject a non-CUDA / non-int64 scalar. Eager
parity of the refactored host path is unchanged (`test_attention`, `test_executor`).

### Operational note (build memory)

The attention `kernel.cu` (WMMA + 8 `decode_attn` template instantiations) is memory-heavy to
compile; building it on the shared **login node** can get `cicc` **OOM-killed** (`signal 9`). Build
it on a compute node instead: `srun --partition=gpu-turing --gres=gpu:1 --mem=32G bash
scripts/build_kernels.sh attention`.

---

## Concept #3 — The prepare/forward split *is* the capture/replay boundary

**Phase:** 3 · **Status:** done & validated (2026-06-07) · design pattern, not a bug

### The concept: what runs once at capture vs. every replay

A CUDA graph captures a **fixed sequence of GPU work over fixed addresses** (§1.4–1.6). But real
decoding needs *something* to change each token: the input id, the position, the RoPE row. The
resolution (from §1.5 rules 2–3) is to split every decode step into two halves:

1. **Update step** — runs *eagerly, outside* the graph. Writes the per-step values **into the
   fixed-address buffers in place**: the new token, the device position scalars, the RoPE row.
2. **Forward step** — the part that is **captured once and replayed**. It only *reads* those fixed
   buffers (plus the persistent weights / KV cache) and does the 150-launch transformer math.

The graph never changes; only the *contents* of the static buffers do, and only the cheap update
step touches them. This split is the whole reason the static decode forward exists.

### How it maps onto the executor

| Half | Method | Runs | Touches |
|---|---|---|---|
| Update | `_prepare_decode_static(token_id)` | eager, per step (per replay in Phase 4) | `static_input_ids.copy_`, `cache_position.fill_` (write_pos), `static_cur_len.fill_` (= write_pos+1), `refresh_decode_rope(pos)` |
| Forward | `_decode_forward_static()` | captured once, replayed | reads only those static buffers + weights + KV cache, via the `_dev` ops |

`decode_step_static` = update + forward + advance, and is the **eager** stand-in we validate now;
Phase 4 will capture `_decode_forward_static` and call `_prepare_decode_static` before each
`replay()`. Because the two halves are already cleanly separated, the Phase 4 change is small.

### Why everything else is already static (and the allocation point, again)

The forward reassigns `hidden = op(...)` to a fresh `torch::empty` each layer — but those come from
the caching allocator, which under capture hands out **deterministic pool addresses** (Concept #1 /
§1.7). So they're effectively static across replays; we do **not** pre-allocate per-op buffers. The
only things that genuinely had to become fixed-address-and-updated-in-place were the *inputs*: the
token id, the two position scalars, and the RoPE row.

### What stays outside the graph (and why)

The RoPE refresh (`refresh_decode_rope`) is a host-side slice + `copy_` in the **update** step. It's
correct there, but it's one extra per-step launch we'd like to eliminate eventually by indexing the
full rope table with the device position scalar *inside* the kernel (tracked under "Future
optimizations" in `graph_plan.md`). Sampling/argmax also stay outside — they involve host decisions
and must not be in the frozen region (§1.5 rule 4).

### Validation

`runtime/tests/test_decode_static.py`: `decode_step_static` reproduces eager `decode_step`
**bit-exactly** (`torch.equal`) over a 6-step greedy trajectory — confirming the static path is a
faithful, capture-ready mirror of the eager decode before we wrap it in a graph.

> Test-isolation gotcha: `test_decode_static` keeps a 7B model resident (`setUpClass`). Don't run it
> in the same job as the `test_buffers` GPU tests (which load a *second* 7B) — two 7B copies OOM a
> 24 GB card. Run the modules in separate jobs.

---

## Concept #4 — Launch-bound vs. memory-bound: the graph works, but 7B decode doesn't need it

**Phase:** 4 · **Status:** measured (2026-06-07) · **the most important finding so far**

### The result

The CUDA-graph decode is correct — graphed `decode_step_graph` reproduces eager `decode_step`
**bit-exactly** over a trajectory, and one captured graph is reused across prompts. But the speedup
over eager is **1.00×**:

```
decode tokens/sec (7B, batch=1, Quadro RTX 6000)
  eager : 29.73 ms/tok   33.6 tok/s
  graph : 29.74 ms/tok   33.6 tok/s     speedup 1.00x
```

The graph is not broken and it *does* cut host work — measuring CPU time to *queue* one step (no GPU
wait) shows eager **28.1 ms** vs graph **20.6 ms**. But both are **below** the 29.7 ms the GPU needs
per token, so the host never becomes the bottleneck and wall-clock doesn't move.

### The concept: two different bottlenecks

A decode step can be limited by either side:

- **Launch-bound (host-bound):** the CPU can't *issue* kernels fast enough; the GPU sits idle
  waiting for the next launch. This is what CUDA graphs fix (collapse ~150 launches → 1 replay).
- **Memory/compute-bound (GPU-bound):** the GPU itself is the bottleneck; the CPU finishes issuing
  all the launches with time to spare. Graphs save host work that was *already hidden*, so wall-clock
  is unchanged.

**7B batch-1 decode is firmly GPU-bound, specifically memory-bandwidth-bound.** Every token must
read all ~15 GB of fp16 weights from HBM at least once:

```
15 GB / 672 GB/s (RTX 6000 GDDR6 peak)  ≈ 22 ms/token   (hard floor)
measured: 29.7 ms/token                 ≈ 74% of peak bandwidth
```

The arithmetic intensity at batch=1 is tiny (each weight is multiplied by a single activation vector,
then thrown away), so the GEMMs are bandwidth-starved, not launch-starved. ~150 launches × ~10–50 µs
≈ 1.5–7 ms of host work hides easily behind 29.7 ms of GPU work.

### Why the original "super launch-bound" premise was wrong here

It's a reasonable hypothesis — "150 tiny kernels per token, surely the launches dominate" — but it
must be **measured**, not assumed. At batch=1 on a 7B model the kernels aren't tiny in *time*: the
weight-matrix reads dominate. Launch overhead only dominates when the per-kernel GPU work is small
relative to launch latency.

### Where CUDA graphs *will* pay off (so the work isn't wasted)

The infrastructure (stream-correct kernels, device-scalar positions, static buffers, capture/replay)
is correct and reusable. It helps wherever decode is actually launch-bound:

1. **The 0.5B draft model (speculative decoding, Phase 8d).** ~14× less weight memory
   (~1 GB → ~1.5 ms/token of HBM traffic), so the same ~1.5 ms of host launch overhead is now
   *comparable to* GPU time → the draft's γ-step autoregressive loop is genuinely launch-bound.
   This is the natural home for the decode graph. (Blocked on the draft `head_dim=64` kernels.)
2. **Larger batch / quantized (int8/int4) weights** — both raise arithmetic intensity / cut weight
   traffic, shifting the balance toward launch-bound.
3. **Not the 7B VERIFY forward:** verify runs γ+1 query rows through the *same* single weight read,
   so it's even more compute-dense per launch than decode → graphs help it even less. A 7B VERIFY
   graph would also be ~1.00×.

### Takeaway

Measure the bottleneck before optimizing. CUDA graphs are the right tool for **launch-bound**
inference (small models, the draft side of spec-decode, high batch), not for **memory-bound**
single-stream 7B decode. The Phase 1–4 machinery is correct and validated; its payoff is on the
draft model, not the 7B target.
