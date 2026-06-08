# BUG: split-KV decode attention produces all-NaN logits under CUDA-graph capture

**Status:** OPEN. Root cause not yet found. The feature works eagerly and in an
isolated graph, but corrupts (all logits NaN) when the split-KV decode kernel runs
inside the **full-model** captured decode graph.

## TL;DR

We grafted the split-KV "flash-decoding" decode path
(`decode_attn_split_kernel` + `decode_attn_combine_kernel` + `choose_num_splits`)
into `runtime/production_kernels/{draft,target}/attention/kernel.cu`, wiring it
behind the existing eager and device-scalar (`_dev`) attention ops with **no Python
/ executor ABI change**. The kernel is numerically correct eagerly. But when the
**device-scalar (`_dev`) path** runs inside the captured S=1 decode graph
(`decode_step_graph`), every one of the 151936 output logits comes back `NaN`
(`inf=0`, so it is **not** an fp16 overflow). The single-block decode kernel
(the previously-validated 1.64× decode graph) is unaffected — the corruption is
specific to swapping single-block → split inside the captured forward.

## The feature being integrated

- New, more-optimized decode-attention kernels arrived in `new_draft_attention.cu`
  / `new_target_attention.cu` (repo root). They add split-KV flash-decoding (KV
  axis partitioned across `grid.x`, per-split online-softmax partials in fp32
  scratch, then a combine kernel) to fill the RTX 6000's 72 SMs at B=1.
- Those root files were forked from a pre-CUDA-graph snapshot and dropped the
  device-scalar `_dev` variants + `getCurrentCUDAStream()` launches. We **merged**
  the split kernels into the production kernels instead of copying, keeping the
  graph scaffolding. Plan: `/home/cme213/eliwand/.claude/plans/breezy-twirling-umbrella.md`.
- Scope chosen by the user: eager + graph both, both models, all S (decode S=1 +
  verify S=2..8) through the shared `dispatch_decode_attn`.

## Symptom / how to reproduce

```bash
bash slurm/run_python.sh runtime/benchmarks/draft_graph_decode.py
#   ... speedup: 2.26x   (timing is valid — graph replay works, it's just NaN)
#   correctness: graph vs eager next-token logits max|Δ| = nan  (CHECK)

bash slurm/run_python.sh runtime/benchmarks/diag_split_decode.py
#   (B) eager (single-block): nan=0 inf=0   |   graph (split-KV): nan=151936 inf=0
```

The decode loop does not crash (no illegal access; exit 0); the GPU does the full
work each replay, so the 2.26× wall-clock number is real — the **logits are just
all NaN**, so the printed `max|Δ|` is `inf-inf`/`nan`.

## Evidence (what the diagnostics established)

`runtime/benchmarks/diag_split_decode.py` — (A) kernel vs PyTorch SDPA, (B) full
forward eager vs graph:

```
(A) draft decode_attn_forward (EAGER) vs SDPA, GQA 14/2, D=64, max_seq=512
 cur_len  splits  out.nan  out.inf   max|Δ| vs SDPA
      33       1    False    False          0.00024   OK
     400       2    False    False          0.00006   OK     (all 6 rows OK)
  -> kernel isolation: PASS (finite, matches SDPA)

(B) eager vs graph decode logits (cur_len=33; eager splits=1, graph splits=2)
  eager (single-block) : nan=    0 inf=    0   (clean)
  graph (split-KV)     : nan=151936 inf=    0   (ALL logits NaN)   <-- both random & real prompts
```

`runtime/benchmarks/diag_graph_split_localize.py` — graph captured around ONE
`decode_attn_forward_dev` call:

```
  control: single-block (max_seq=128)         splits=1  eager.nan=False  graph.nan=False  max|Δ|=0.00000
  suspect: split-KV     (max_seq=512)         splits=2  eager.nan=False  graph.nan=False  max|Δ|=0.00000
  suspect: split-KV     (max_seq=512, len=400)splits=2  eager.nan=False  graph.nan=False  max|Δ|=0.00000
```

### Ruled OUT
- **Split/combine math** — eager output matches SDPA to ~2e-4 at num_splits 1 and 2 (A).
- **fp16 logit overflow** — `inf=0`; it is genuine NaN, not saturation. Eager
  (single-block) is clean on the same prompts.
- **The split op under graph in isolation** — a graph around a single
  `decode_attn_forward_dev` is bit-exact and finite (localizer).
- **Scratch allocation strategy** — the first version used per-call `torch::empty`
  for `partial_o/m/l`; the second used **persistent scratch** (allocated once
  outside capture, reused). **Both produce the identical `nan=151936`.** So
  in-capture alloc/free of scratch is NOT the cause. (The persistent-scratch
  attempt is currently in the source — see below — and was verified to be the
  loaded build: draft `kernel.cu` 17:26 → `kernel.o`/`.so` 17:29, `grep s_po->data_ptr` = 1.)

### Narrowed to
The bug requires **(split-KV kernel) × (full-model graph capture)** simultaneously:
- split-KV eager (full model or isolated): clean.
- split-KV under graph, single op: clean.
- single-block under graph, full model: clean (the validated 1.64× path).
- **split-KV under graph, full model (24 layers): all-NaN.**

The only code change between the working single-block graph and the broken split
graph is `launch_decode_attn` choosing the split+combine path (two kernels +
fp32 scratch + `grid=(num_splits,h_q,B)`) instead of one `decode_attn_kernel`
launch. Everything upstream/downstream (qkv_proj, rope_kv_write_dev, o_proj,
residual, rmsnorm, swiglu, lm_head, the device `cur_len`/`cache_position`
scalars, static RoPE gather) is unchanged and works with single-block.

## Current code & build state (IMPORTANT — source/.so are out of sync for target)

- `runtime/production_kernels/draft/attention/kernel.cu` — split graft + **persistent
  scratch** dev-path attempt. **`.so` REBUILT** from this (still NaNs).
- `runtime/production_kernels/target/attention/kernel.cu` — same edits. **`.so` NOT
  rebuilt** with the persistent-scratch version: the rebuild OOM-killed `cicc`
  (`nvcc error: 'cicc' died due to signal 9`) on the **login node**. The target
  `.so` currently in place is the *first* build (split with per-call `torch::empty`
  scratch). Target source has persistent scratch; target `.so` does not. **Rebuild
  target on a compute node**, e.g.
  `srun --partition=gpu-turing --gres=gpu:1 --mem=32G --time=00:30:00 bash scripts/build_kernels.sh target attention`
  (login-node `cicc` OOMs intermittently on these heavy WMMA + 8×Q_TOKENS files).
- Both `.so` flavors (per-call and persistent scratch) reproduce the NaN.
- Eager paths and prefill are unaffected. `bindings.cpp` / `ops.py` / `__init__.py`
  / `executor*.py` / `buffers.py` are unchanged.

## Key code

- Split path lives in `launch_decode_attn<Q_TOKENS>()` in both `kernel.cu`. Behind
  `decode_attn_forward` (eager) and `decode_attn_forward_dev` / `small_q_attn_forward_dev`
  (graph). Eager sizes `num_splits` from `cur_len`; the `_dev` path sizes it from
  `max_seq = cache_k.size(2)` (fixed) and reads `cur_len` from a device int64
  scalar; both compute `split_len` in-kernel.
- The captured forward is `runtime/executor_graph.py::_decode_forward_static`
  (→ `_run_attention(static_attn=True)` in `executor.py`, which calls
  `decode_attn_forward_dev`). Capture/replay + side-stream warmup in
  `_capture_decode_graph`. One decode graph serves all S=1 steps.

## Open hypotheses (untested)

1. **Two-kernel WAR/RAW dependency under capture.** The split path is
   `split_kernel` (writes scratch) → `combine_kernel` (reads scratch) → result in
   `o`. In the full captured forward, with many interleaved ops/allocations, the
   scratch read/write dependency, or the combine→o→o_proj chain, may not be
   ordered/captured as expected on replay. (Single-op localizer wouldn't expose a
   cross-op interleaving effect.)
2. **Graph-pool / address reuse of `q` or `o` specific to having an extra kernel
   between the qkv output and o_proj.** The split path inserts a kernel boundary
   (combine) that the single-block path doesn't; a transient `q`/`o` block may be
   recycled differently under capture.
3. **A capture-time issue triggered only by the split launch pattern** (e.g.
   `grid.x=num_splits`, or the fp32 scratch dtype) at full-model scale.

## NEXT STEP (the missing bisection — do this first)

Add `decode_step_static` to the repro. It runs the **`_dev` (split) ops through the
full forward, eagerly, with NO graph capture** (`runtime/executor_graph.py::decode_step_static`
→ `_decode_forward_static`). This separates "dev/split path broken in the full
24-layer chain" from "graph-capture-specific":

```python
# in diag_split_decode.py reproduce(), between the eager and graph steps:
logits = ex.prefill(p); tok = int(logits[0,-1].argmax())
lg_s = ex.decode_step_static(torch.tensor([tok], device=DEVICE)).clone()  # eager _dev/split, NO graph
print("  static (eager _dev):", _logit_stats(lg_s))
```

- **If `lg_s` is NaN** → the split dev path is broken in the full-model chain even
  eagerly (a real kernel/launcher bug, not capture). Then bisect by layer count /
  inspect the first NaN layer; suspect the shared persistent scratch across 24
  layers, or `num_splits=2` + device `cur_len` in the chained context.
- **If `lg_s` is clean** → the bug is strictly graph-capture-specific. Then probe
  capture interactions: try forcing `num_splits=1` on the `_dev` path (should
  match single-block and be clean → confirms it's the split/combine-under-capture),
  and try the pre-allocated-in-`RuntimeBuffers` scratch variant (pass explicit
  buffers through the ABI) to rule out any allocator/pool interaction definitively.

## Other follow-ups noted along the way

- `runtime/tests/test_draft_decode_graph.py` / `test_decode_graph.py` use
  `MAX_SEQ=128` → `choose_num_splits=1` → they **never exercise the split path**
  (need `MAX_SEQ ≥ 256`). They also assert `torch.equal` (bit-exact eager==graph),
  which the split path breaks by design (eager uses `num_splits` from `cur_len`,
  graph from `max_seq`). A split-exercising test with a finite-robust metric
  (argmax match / max|Δ| over finite entries) is needed once the NaN is fixed.
- `draft_graph_decode.py`'s correctness line compares raw fp16 logits and is
  fragile; switch to argmax-match or a finite-masked diff.

## Repro / diagnostic files

- `runtime/benchmarks/diag_split_decode.py` — (A) kernel-vs-SDPA, (B) eager-vs-graph logits.
- `runtime/benchmarks/diag_graph_split_localize.py` — graph around a single split op.
- `runtime/benchmarks/draft_graph_decode.py` — original benchmark that surfaced it.
