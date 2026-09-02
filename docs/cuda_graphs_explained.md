# CUDA Graphs, Explained — Concepts → This Codebase

A from-scratch conceptual guide to *why* we want CUDA graphs and *what* they require,
followed by a walkthrough of how our current `runtime/` engine maps onto those concepts.
Read this before `cuda_graph_issues_and_concepts.md` (the implementation journal).

No code changes here — this is the mental model.

---

## Part 1 — The concepts

### 1.1 How the GPU actually runs our code (host vs. device)

Two processors are involved in every inference step:

- **The host** (CPU, running our Python + PyTorch + the pybind C++ shims).
- **The device** (the GPU, running the actual `.cu` kernels).

When Python calls `rmsnorm_forward(x, w, eps)`, the CPU doesn't compute anything itself. It
**enqueues** a kernel onto a *CUDA stream* — think of a stream as a to-do list the GPU works
through in order. The CPU says "GPU, when you get to it, run `rmsnorm_kernel` with these
pointers and sizes," and then *immediately returns* to Python to issue the next op. The GPU
chews through the stream asynchronously.

This enqueue step is **a kernel launch**, and it is *not* free on the CPU side. Each launch pays:

- Python interpreter overhead (the call, arg marshalling),
- pybind11 / `torch` dispatch overhead,
- the CUDA driver building and submitting the launch command.

Call it **~5–50 µs of CPU time per launch**. The GPU compute for a single small op at batch=1,
S=1 might itself only take a few µs. See the asymmetry?

### 1.2 "Launch-bound" — the core problem

Picture the timeline for one decode step (one new token):

```
CPU:  [launch k1][launch k2][launch k3] ............ [launch k150]
GPU:        [k1] [k2] [k3] .............................  [k150]
                ^ GPU finishes k1 long before CPU has finished
                  enqueuing k2, k3, ... so it sits IDLE waiting.
```

When the CPU can't enqueue work fast enough to keep the GPU busy, you are **launch-bound** (a.k.a.
CPU-bound / overhead-bound). The GPU is mostly *idle*, waiting for the next launch command. Adding
a faster GPU wouldn't help — the bottleneck is the host issuing 150 separate launches.

Our engine does roughly **30 layers × ~5 ops ≈ 150+ launches per forward**, and at decode time
(batch=1, sequence length 1) each kernel is tiny. This is the textbook launch-bound regime.

### 1.3 The fix in one sentence

**Record the entire sequence of 150 launches *once*, then re-issue all of them with a *single*
CPU command.** That recording is a **CUDA graph**.

### 1.4 What a CUDA graph actually is

A CUDA graph is a **frozen recording of a sequence of GPU operations** — the kernels, their
arguments, and the dependencies between them — captured as one replayable object.

- **Capture**: you put a stream into "capture mode," run your forward pass once (the launches are
  *recorded*, not just executed), and end capture. You now hold a `cudaGraph` / `torch.cuda.CUDAGraph`.
- **Replay**: `graph.replay()` issues the *whole* recorded sequence to the GPU with **one** launch
  command from the CPU. 150 launches' worth of CPU overhead collapses to ~1.

```
Without graph:  CPU does 150 enqueue calls every token.   (launch-bound)
With graph:     CPU does 1 replay call every token.        (now GPU-bound — good!)
```

The Python-level layer loop, the per-op pybind calls, the dispatch — all of that CPU work happens
**only during the one capture**, never again on replay.

### 1.5 The catch: a graph is *frozen*, including its arguments

This is the part that makes graphs tricky, and it's the source of every bug in the previous
attempt. When you capture a kernel launch, the graph records the **exact argument values** that
were passed — and that **includes raw pointer addresses and scalar values**.

So a recorded launch is literally: "run `attn_kernel` reading from address `0x7f...A`, writing to
address `0x7f...B`, with `cur_len = 5`." On replay it does *exactly that* again. It does **not**
re-evaluate your Python; it does **not** notice that you'd like `cur_len` to be 6 now, or that the
new token lives at a different address.

This gives us the three golden rules of graph-friendly code:

1. **Static addresses.** Every tensor the graph reads or writes must live at a *fixed* address
   across replays. You cannot hand the graph a freshly-allocated tensor each step — the recorded
   launch points at the *old* address.

2. **Update inputs in place, never by re-passing.** To feed a new token in, you must **write it
   into the same buffer** the graph recorded as its input (`static_input.copy_(new_token)`), then
   replay. The graph reads whatever is at that fixed address *now*.

3. **No host values baked into the recording for things that must change.** A scalar argument like
   `cur_len=5` is frozen at 5 forever. If the value must vary per step, it cannot be a host scalar
   argument — it has to live in **device memory at a fixed address** that the kernel dereferences,
   so we can update the *contents* before replay (rule 2) without changing the recorded *argument*
   (which is just the pointer).

There's also a fourth, implicit rule:

4. **No host-dependent control flow inside the captured region.** An `if token.item() > 0:` or a
   data-dependent loop count can't be recorded — the branch/length is decided once at capture and
   baked. (A *fixed* Python `for layer in range(30)` is fine: it just unrolls into the recording.)

### 1.6 Capture happens on a *stream* — and your kernels must launch on it

This is the rule that actually broke our first attempt, so it gets its own section.

Recall (§1.1) that a kernel launch is an *enqueue onto a stream*. Capture works by putting **one
specific stream** into "capture mode": every operation enqueued **onto that stream** while it's
capturing gets recorded into the graph instead of (only) running. `torch.cuda.graph(...)` creates
and switches to a dedicated **non-default** capture stream for you, runs your Python body, and ends
capture.

The catch: a kernel only gets recorded if it is launched **onto the stream that is capturing**. In
CUDA C++, a launch written as `my_kernel<<<grid, block>>>(...)` with no stream argument goes to the
**default stream (stream 0)** — *not* the current/capture stream. So a custom kernel written that
way will:

- run for real on stream 0 during the capture (so your output tensor *looks* correct right after
  capture), but
- **never be recorded into the graph**, and
- worse, doing GPU work on the default stream *while another stream is capturing* is illegal —
  CUDA/PyTorch will warn "**CUDA Graph is empty… captured on wrong device or stream**."

Then `graph.replay()` re-issues only whatever *did* get recorded (e.g. PyTorch's own
reshape/transpose/copy ops, which correctly use the current stream) — reading from intermediate
buffers that the *unrecorded* kernels were supposed to fill. The result is garbage, and because the
pool memory those buffers point at keeps changing, the error **grows on each replay**.

5. **Every kernel must launch on the current stream.** In each launcher, pass
   `at::cuda::getCurrentCUDAStream()` as the 4th `<<<>>>` argument:
   `my_kernel<<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(...)`. (PyTorch's own ops
   and cuBLAS already do this; only hand-written `<<<>>>` launches are at risk.) This is harmless in
   eager mode — when nothing is capturing, the current stream *is* the default stream.

### 1.7 The allocation problem — and why PyTorch already solves it

Rule 1 sounds fatal for PyTorch code, because almost every op allocates its output with
`torch.empty` / `torch::empty`. Two issues:

- The underlying `cudaMalloc` is **not capturable** (you can't record a malloc into a graph).
- Even if it were, you'd get a *different* address each run, violating "static addresses."

PyTorch's **caching allocator + a private graph memory pool** solves both:

- During capture, PyTorch routes every `torch::empty` to a **private pool** reserved for that
  graph, using its own bookkeeping (no live `cudaMalloc` during capture).
- Crucially, it assigns those allocations **deterministically**, so on replay the *same* logical
  `torch::empty` gets the *same* address. The intermediate buffers are effectively static.

**Consequence for us:** ops that allocate their outputs with `torch::empty` (which is *all* of
ours) are **graph-safe** as long as we capture inside `torch.cuda.graph(...)`. We do **not** need
to rewrite every kernel to write into pre-allocated `_out` buffers. (This is the key disagreement
with the previous (reset-away) attempt, whose stated root cause — "torch::empty breaks replay" — is
**wrong**. Phase 0 confirmed it empirically: a single rmsnorm op captures and replays correctly
through the caching allocator. The real blocker was the stream rule in §1.6, not allocations.)

What is **not** safe, and would genuinely break capture, is a kernel doing a raw `cudaMalloc`, a
`cudaMemcpy`, a `.item()`, or a `cudaDeviceSynchronize` mid-forward. We grepped — **none of our
kernels do any of that.** They only use the caching allocator. Good.

### 1.8 Summary of what graphs demand of us

| Concept | Requirement | Why |
|---|---|---|
| Launch overhead | Replay 150 launches as 1 | The whole point |
| **Stream** | Every kernel launches on `getCurrentCUDAStream()` | Default-stream launches aren't recorded (§1.6) |
| Frozen arguments | Inputs at **fixed addresses**, updated **in place** | Graph records pointers, not your intent |
| Varying scalars | Position/length must be a **device tensor**, not a host int | Host ints get baked into the recording |
| Allocations | OK if via caching allocator under `torch.cuda.graph` | Private pool gives deterministic addresses |
| Control flow | No `.item()` / data-dependent branches in captured region | Branches are baked at capture |
| Warmup | Run the forward a few times before capturing | Initializes cuBLAS workspaces etc. so capture is clean |

---

## Part 2 — Our codebase, read through these concepts

Now let's trace an actual decode step and label every piece as *static / graph-safe* vs. *the
thing we have to fix*. File references are to the current tree.

### 2.1 The forward we want to capture

`Qwen2Executor.decode_step` (`runtime/executor.py:467`) is the per-token forward — the launch-bound
hot loop. It does:

```
decode_step(token_id):
    input_ids = token_id.unsqueeze(1)                 # make [batch, 1]
    hidden = embedding_forward(input_ids, embed_w)     # 1 kernel
    for layer in range(30):                            # _forward_stack -> _run_decoder_layer
        input_rmsnorm                                  # 1 kernel
        attention:                                     #   qkv_proj, rope_kv_write, attn, o_proj
        residual_add                                   # 1 kernel
        post_attn_rmsnorm                              # 1 kernel
        swiglu_mlp                                     # ~3 kernels
        residual_add                                   # 1 kernel
    final rmsnorm                                      # 1 kernel
    lm_head                                            # 1 kernel
    return logits
```

That `for layer in range(30)` and all the pybind op calls are **pure host work** — exactly the
overhead a graph eliminates by recording it once. The math itself is what we want to keep replaying.

### 2.2 What's already graph-friendly (the good news)

- **Persistent weights.** All weights live at fixed addresses for the whole run, and the stacked
  QKV weights are precomputed once in `__init__` (`runtime/executor.py:104-109`). Static. ✅
- **Persistent KV cache & RoPE tables.** `buffers.kv_cache_k/v`, `buffers.rope_cos/sin` are
  allocated once (`runtime/buffers.py:179-196`) and never reallocated. Static addresses. ✅
- **A device-side position scalar already exists.** `buffers.cache_position` is a 0-d int64 CUDA
  tensor, and `executor._advance_cache_pos` (`runtime/executor.py:192-197`) already updates it on
  device via `fill_`/`add_` *without* an `.item()` sync. This is precisely the device-scalar
  pattern graphs need — it's just **not consumed by the kernels yet** (see 2.4).
- **Intermediate allocations.** Every op output is a `torch::empty` via the caching allocator
  (rmsnorm `kernel.cu:138`, residual `kernel.cu:117`, lm_head `kernel.cu:152`, swiglu `kernel.cu:50,161`,
  embedding `kernel.cu:62`, attention `kernel.cu:60,578,820,939`). Under `torch.cuda.graph` these
  get deterministic pool addresses. Graph-safe. ✅ (Phase 0 confirmed this empirically — see 2.5.)

### 2.3 What changes every step (so it must become a static input we update in place)

- **The token id.** `decode_step` builds `input_ids = token_id.unsqueeze(1)`
  (`runtime/executor.py:481`) — a *fresh* tensor each call, and `embedding_forward` reads it. A
  captured graph would forever read the *first* token's address. → We need a **static
  `input_ids` buffer**; each step `copy_` the new token in, then replay. (Plan Phase 3.)

### 2.4 The second blocker (after streams): host-int positions baked into the recording

> Read 2.5 first — the **#1** blocker Phase 0 found is that our kernels launch on the default
> stream. The host-int positions below are the *next* thing to fix, and only matter once capture
> actually records our kernels.

Inside `_run_attention` (`runtime/executor.py:216-276`):

```python
write_pos = self._cache_pos                                  # a Python int
cos, sin = buf.rope_embeddings(write_pos, seq_len)           # HOST slice by position
ops.rope_kv_write_forward(k, v, cache_k, cache_v, write_pos, cos, sin)   # write_pos is host int64
...
cur_len = write_pos + seq_len                                # a Python int
attn_ctx = ops.decode_attn_forward(q, cache_k, cache_v, cur_len, ...)    # cur_len is host int64
```

Two distinct violations of rule 3 / rule 1:

1. **`write_pos` and `cur_len` are host `int64` arguments** to the kernels
   (`attention/bindings.cpp:6-18`, passed as `static_cast<int>(...)` into the launch in
   `attention/kernel.cu`). The graph **bakes these values in.** Replay would forever scatter K/V
   at the captured slot and attend over the captured length — frozen at the position where we
   happened to capture, producing exactly the "garbage that grows as the sequence advances"
   symptom the previous attempt reported. → The kernels need **`_dev` variants that read the position from
   a 0-d int64 CUDA tensor** (we already maintain such a tensor: `cache_position`). Then the
   recorded *argument* is just a pointer; we change the *contents* before each replay. (Plan Phase 2.)

2. **RoPE `cos/sin` are host-sliced by position.** `buffers.rope_embeddings`
   (`runtime/buffers.py:119-133`) returns `self.rope_cos[start:start+length]` — a *view* whose
   `data_ptr` depends on `write_pos`. Capturing bakes that one slice's address in, so replay always
   applies the rotation for the *captured* position. → Fix by keeping a small **static
   `static_cos`/`static_sin` buffer** and refreshing it for the current position before replay.
   *(Update: this was later improved — the RoPE gather is now done **inside** the captured graph via
   `index_select` driven by the `cache_position` scalar, so `static_cos/sin` and the host refresh
   were removed. See `cuda_graph_issues_and_concepts.md` Concept #3.)* (Plan Phase 3.)

### 2.5 What Phase 0 actually found (and how it corrects the previous attempt)

Phase 0 (`runtime/benchmarks/phase0_graph_probe.py` + `phase0_graph_diag.py`) captured the *current*
eager decode forward at a **fixed** position and replayed it with **identical inputs** — so the
host-int positions of 2.4 were correct-and-constant, isolating everything else. Capturing
progressively larger chunks gave:

| Stage captured | Result |
|---|---|
| single `rmsnorm` | "matched" eager — but only because the graph was **empty** (PyTorch warned `CUDA Graph is empty… wrong device or stream`); replay was a no-op |
| one decoder layer | capture already diverged from eager, and replay **grew**: −2.4 → −1.6 → −34 → −47 |
| full stack / forward | **NaN** |

Per §1.6 this is the **stream** rule, not allocations. Every custom kernel launches with bare
`<<<grid, block>>>` (`embedding/kernel.cu:67`, `rmsnorm/kernel.cu:155`, `residual_ops/kernel.cu:92`,
`attention/kernel.cu:200,593,829`, `swiglu/kernel.cu:136`) — i.e. on the **default stream**, never
on the capture stream. During capture they execute on stream 0 but are **not recorded**; replay
re-runs only the captured PyTorch view/copy ops over buffers nothing fills → garbage that grows as
the pool churns. The lone rmsnorm "passed" precisely because *nothing* of it was recorded.

So:

- **The allocation theory is dead.** rmsnorm proves the caching-allocator path works; `torch::empty`
  is fine. We do **not** rewrite ops allocation-free (contra the previous attempt).
- **The real fix is the stream rule** (§1.6): launch every kernel on `getCurrentCUDAStream()`. That
  is the new **Plan Phase 1**, and it must land before capture can record anything.
- The host-int positions (2.4) are still a genuine bug — they'd produce the previous attempt's growing
  `614 → 3978` "frozen at position `p`" error *after* streams are fixed — so they're **Plan Phase 2**.

The previous attempt likely hit *both* problems at once (it never fixed the stream launches) and blamed
the whole thing on allocations.

### 2.6 The capture/replay loop we're building toward (concept, not code)

```
ONCE, after a prefill:
  warm up the static decode forward a few times      # init cuBLAS workspaces; settle the pool
  with torch.cuda.graph(g, pool=shared_pool):
      logits = decode_forward_static()               # reads ONLY static buffers + persistent weights
  keep handles to: static_input_ids, cache_position(device), static_cos/sin, logits

EACH TOKEN:
  static_input_ids.copy_(next_token)                 # update input in place (rule 2)
  cache_position.add_(1) ; cur_len.add_(1)           # advance device scalars (contents, not args)
  refresh static_cos/sin for the new position        # cheap eager copy, outside the graph
  g.replay()                                          # 150 launches -> 1 CPU command
  read logits   (sampling/argmax stays OUTSIDE the graph)
```

Everything that varies is communicated through **fixed-address device memory updated in place**;
the recording itself never changes. Sampling, rollback, and MPI stay eager — they involve host
decisions and must not be inside the frozen region (rule 4).

### 2.7 Mapping concepts → plan phases

| Concept (Part 1) | Where it bites us (Part 2) | Plan phase |
|---|---|---|
| Allocations are fine via the pool | 2.2 / 2.5 — confirmed in Phase 0 | Phase 0 ✅ |
| Kernels must launch on the capture stream (§1.6) | 2.5 — all kernels on default stream | Phase 1 |
| Varying scalar → device tensor | 2.4(1) — `write_pos`/`cur_len` host ints | Phase 2 |
| Inputs at fixed addresses | 2.3 — fresh `input_ids`; 2.4(2) — host-sliced rope | Phase 3 |
| Record once, replay as one launch | 2.6 — capture/replay machinery | Phase 4 |
| Frozen recording ⇒ must verify parity | replay vs eager vs HF | Phase 5 |
| Same machinery, bigger fixed shape | the speculative `verify_gamma` (S=γ+1) | Phase 6 |

---

## TL;DR

- We're slow because the **CPU** issues ~150 tiny kernel launches per token while the **GPU**
  idles — "launch-bound."
- A **CUDA graph** records those launches once and replays them as a **single** CPU command.
- A graph is **frozen**: it remembers exact **addresses and scalar values**. So anything that
  changes per step must be (a) written into a **fixed-address buffer in place**, or (b) read by
  the kernel from a **device-side scalar**, never passed as a host int that gets baked in.
- A graph also only records work issued **onto the capturing stream**. Phase 0 found our **#1
  blocker**: every custom kernel launches on the **default stream**, so capture recorded nothing
  and replay produced growing garbage / NaN. Fix = launch each kernel on `getCurrentCUDAStream()`.
- Our kernels already allocate only through PyTorch's caching allocator, so the graph's **private
  pool handles intermediates** — no need to rewrite every op (contra the previous attempt; Phase 0
  confirmed rmsnorm captures/replays cleanly).
- So the work, in order: **(1) stream-correct the kernels**, **(2) device-scalar positions**
  (`write_pos`/`cur_len`), **(3) static inputs** (token id + rope cos/sin), then **capture/replay**
  — first for DECODE (S=1), then the speculative VERIFY (S=γ+1). This is the order the implementation followed.
- Eager baseline to beat: **33.8 tok/s** (7B, batch=1, RTX 6000).
