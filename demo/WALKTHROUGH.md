# Inference Path Walkthrough

This walks through the single-GPU stack that prefills a query and generates response tokens with speculative decoding. Every piece has a matching `WALKTHROUGH [NAME]` comment: search for that marker if line numbers have changed. The local comments explain the engineering choice at the point where it is implemented.

We will follow a fresh query with **four draft tokens per round** (`gamma = 4`). We first cover setup, because by the time Compare is clickable, the weights are loaded and the draft decode graph already exists. Then we trace the request, walk one complete decoder layer down into CUDA, and return through three illustrative speculative rounds to the page. We use made up token IDs for the walkthrough.

The serving setup is batch size 1, FP16 weights/activations, greedy token selection, a 7B target and a 0.5B draft. Both models stay on the same GPU. Initially, the kernels were optimized for a Quadro RTX 6000, which didn't support BF16. The current deployment runs Python and CUDA on the Runpod host; the laptop hosts the browser and an SSH port forward. The initial project hosted the draft and target models on different GPUs to simulate a larger-inference service.

## 1. Orient yourself: the boundaries we cross

```text
Laptop browser: demo/index.html
    POST /api/chat through the local SSH-forwarded port
GPU host CPU: Handler.do_POST -> KernelEngine.generate
    target-only run, then speculative run
    speculative_chunks -> DraftRunner + greedy_target_step
    Qwen2Executor -> role-specific ops.py -> bindings.cpp
    C++ host functions in kernel.cu enqueue work
GPU: custom CUDA kernels + cuBLAS GEMMs + PyTorch tensor operations
    logits -> greedy choices -> accepted token IDs
GPU host CPU: tokenizer.decode -> JSON line -> flush
Laptop browser: stream parser -> card.answer.textContent
```

A `.cu` file contains both CPU host launch functions and GPU functions. `__global__` identifies a GPU kernel. A call to a C++ function such as `qkv_proj_forward` is not itself a GPU thread: that function validates tensors, allocates an output, and asks cuBLAS or a CUDA launch to do the work.

A **model token** is a vocabulary ID and may represent a word, part of a word, punctuation, or part of an encoded character. A **speculative block** here means the accepted draft prefix plus one target-selected token.

## 2. Before the demo: compile the extensions

**Open:** `demo/setup.sh:18` [—](setup.sh#L18) `build both roles` (search `BUILD`) and `setup.py:61` [—](../setup.py#L61) `_make_extension` (search `EXTENSION`).

`bash demo/setup.sh` runs on the CUDA host. It checks for `nvcc`, creates a virtual environment inheriting the image's CUDA-enabled PyTorch, installs the demo dependencies, and sets `TORCH_CUDA_ARCH_LIST` from the GPU unless it was already supplied. On the validated RTX 4090 that architecture is 8.9.

The two `build_ext --inplace` calls build target and draft separately. In `setup.py`, `_selected_extensions` chooses roles and operations; `_make_extension` creates a `CUDAExtension` from each operation's `bindings.cpp` and `kernel.cu`. `BuildExtension` coordinates the host C++ compiler, CUDA compiler, and linking. This produces **ten shared libraries**: five operations for each model role. For example, target attention becomes a platform-tagged `target_attention_ops*.so` beside its `ops.py`. The residual module is named `target_residual_ops`, without a second `_ops` suffix.

The C++ flags use `-O3`; CUDA also uses `--use_fast_math`. PyTorch's extension build chooses architecture flags using the environment/visible GPU. This is ahead-of-time compilation: the server later loads an existing library. It does not compile kernels when you submit a prompt. An incremental build can reuse unchanged objects. Changing CUDA code, the relevant ABI/toolchain, or GPU architecture may require rebuilding.

Next the setup script downloads the two pinned checkpoint revisions. Their safetensors files contain parameters, and their tokenizer/config files contain data. Downloading weights does not build the CUDA extensions.

**Keep three stages separate:** compiling CUDA creates executable GPU code; importing the extension makes its functions callable from Python; capturing a CUDA graph records a sequence of those already-compiled operations for replay. None of these is training.

## 3. Start the process and construct `KernelEngine`

**Open:** `demo/server.py:311` [—](server.py#L311) `main` (search `STARTUP`), then `demo/server.py:53` [—](server.py#L53) `KernelEngine.__init__` (search `ENGINE`).

The command `.venv/bin/python -m demo.server` enters `main`. It parses the model paths, port, and context capacity. The default context is 2048 tokens; the command accepts 32–4096, although fitting a larger configuration is a separate memory question. `KernelEngine` is fully constructed before `DemoServer` binds its loopback socket.

The engine creates:

- A target tokenizer, target generation configuration, and EOS token-ID set. Transformers supplies these utilities; it does not execute the model forward here.
- A target `Qwen2Executor`, holding target weights and buffers, with `use_cuda_graph=False`.
- A draft `Qwen2Executor`, holding draft weights and buffers, with `use_cuda_graph=True`.
- A `DraftRunner` around the draft executor to manage proposal state and target feedback.
- A lock preventing two HTTP requests from changing these shared caches simultaneously.

The target tokenizer defines prompt IDs for **both** models. Startup compares every entry in the draft tokenizer vocabulary against the corresponding target token ID. The embedding matrices have different row counts, so there is also a prompt bound check and a later bonus-token bound check. Token-ID compatibility does not mean the models have identical distributions.

## 4. Load architecture, weights, and persistent GPU state

**Open:** `runtime/core/config.py:44` [—](../runtime/core/config.py#L44) `RuntimeConfig.from_yaml` (search `CONFIG`), `runtime/core/configs/qwen2.5-7b.yaml:8` [—](../runtime/core/configs/qwen2.5-7b.yaml#L8) `model dimensions and layer_order` (search `TARGET_CONFIG`), `runtime/core/configs/qwen2.5-0.5b.yaml:8` [—](../runtime/core/configs/qwen2.5-0.5b.yaml#L8) `model dimensions and layer_order` (search `DRAFT_CONFIG`), `runtime/core/weights.py:83` [—](../runtime/core/weights.py#L83) `load_weights` (search `WEIGHTS`).

The runtime reads its own YAML architecture specifications. The server replaces the configured checkpoint paths and context capacity. It does not instantiate a Hugging Face model and ask that model to run layers.


| Quantity                        | Target   | Draft   |
| ------------------------------- | -------- | ------- |
| Decoder layers                  | 28       | 24      |
| Hidden width `H`                | 3584     | 896     |
| MLP intermediate width `I`      | 18944    | 4864    |
| Query heads                     | 28       | 14      |
| Key/value heads                 | 4        | 2       |
| Head width `D`                  | 128      | 64      |
| Vocabulary rows `V`             | 152064   | 151936  |
| Embedding/LM-head weights tied? | No       | Yes     |
| Kernel role                     | `target` | `draft` |


`load_weights` finds the shard index or single safetensors file, reads named tensors on the host, checks required names/shapes, and copies expected tensors to CUDA in FP16. Transfers happen tensor by tensor, although `_load_raw` first builds the host dictionary for the checkpoint. Parameters stay resident across requests. For the draft, the executor later reuses the input embedding matrix as the output vocabulary projection because embeddings are tied.

**Continue to:** `runtime/core/memory.py:9` [—](../runtime/core/memory.py#L9) `plan_memory` (search `PLAN`), `runtime/core/shapes.py:24` [—](../runtime/core/shapes.py#L24) `kv_cache` (search `KV_SHAPE`), `runtime/buffers.py:179` [—](../runtime/buffers.py#L179) `allocate_buffers` (search `BUFFERS`), `runtime/buffers.py:27` [—](../runtime/buffers.py#L27) `build_rope_tables` (search `ROPE_TABLE`).

Each model receives its own K and V cache with shape:

```text
[layers, batch=1, KV heads, context capacity, head width]
```

For example, a target layer's K cache is `[1, 4, 2048, 128]`. Seven query heads use each KV head in both models. This grouped-query layout saves storage compared with storing K/V separately for every query head. K and V are not shared between target and draft, or between layers.

`allocate_buffers` also creates RoPE tables, a device-side cache-position scalar, fixed graph input/length tensors, and reserved activation buffers. RoPE tables are calculated in FP32 from positions and inverse frequencies, then stored in FP16. Their two half-width sections match Qwen's `rotate_half` convention.

Be precise about allocation: the buffer structure reserves activation arrays, but the current eager executor does **not** route every output into those arrays. Many wrappers allocate new tensors, including full vocabulary logits for all prefill rows. The CUDA graph maintains its captured intermediates through a graph memory pool. `plan_memory` is therefore a reservation estimate, not total peak VRAM. Packed QKV weights, eager intermediates, allocator/library workspaces, and graph storage also matter.

## 5. How the kernels become callable through the engine

**Open:** `runtime/executor.py:72` [—](../runtime/executor.py#L72) `Qwen2Executor.__init__` (search `EXECUTOR`), `runtime/executor.py:45` [—](../runtime/executor.py#L45) `_stack_qkv` (search `PACK_QKV`), `runtime/executor.py:18` [—](../runtime/executor.py#L18) `_import_kernels` (search `IMPORT`).

`KernelEngine` hands configuration, weights, and buffers to `Qwen2Executor`. The executor validates that the role's compiled attention head width matches the configuration: target 128, draft 64. `_import_kernels` imports five Python modules from the selected role and returns a dictionary of callable operations. The attention entry is a module because it exposes several projection/cache/attention variants.

For each decoder layer, `_stack_qkv` concatenates the Q, K, and V projection weights along their output rows and concatenates their biases. This happens once at startup. Packing lets the layer issue one combined QKV projection rather than three separate GEMMs. These packed tensors are additional GPU storage; the original weight dictionary remains present.

Here is the complete binding chain for an attention projection:

```text
KernelEngine.executor
  -> Qwen2Executor._ops["attn"].qkv_proj_forward(...)
  -> target/attention/ops.py: _load_ext().qkv_proj_forward(...)
  -> target_attention_ops shared library
  -> bindings.cpp: registered qkv_proj_forward_py
  -> kernel.cu: qkv_proj_forward
  -> cublasGemmEx on the GPU
```

**Open:** `runtime/production_kernels/target/attention/ops.py:13` [—](../runtime/production_kernels/target/attention/ops.py#L13) `_load_ext and public op wrappers` (search `TARGET_ATTENTION_WRAPPER`) and `runtime/production_kernels/target/attention/bindings.cpp:120` [—](../runtime/production_kernels/target/attention/bindings.cpp#L120) `PYBIND11_MODULE` (search `TARGET_ATTENTION_BINDING`).

`ops.py` loads the already-built module with a regular Python import and checks its location. Missing extensions raise an error with a build command; there is no fallback JIT compilation. Although `_load_ext` is called by wrappers repeatedly, Python caches imported modules. `PYBIND11_MODULE` registers the callable names. Its C++ wrappers release the Python GIL during dispatch. CUDA tensors carry device pointers across this boundary; the large arrays do not make a CPU round trip.

The same three-file structure exists for embedding, RMSNorm, residual/LM head, and SwiGLU. The draft imports independently named draft libraries, so target and draft can coexist in one process. The source index at the end gives every wrapper and binding stop.

## 6. Warmup, graph capture, and browser readiness

**Open:** `demo/server.py:101` [—](server.py#L101) `KernelEngine warmup` (search `WARMUP`); the graph implementation is detailed in stop 16.

The engine prepares a short `Hello` prompt and consumes both generation paths with an eight-token budget. This initializes the execution paths and triggers the draft's lazy one-token CUDA graph capture. A GPU synchronization finishes the warmup before readiness metadata is published. Model load, this warmup, and graph capture are outside the displayed per-query timers.

**Open:** `demo/server.py:235` [—](server.py#L235) `Handler.do_GET` (search `GET`) and `demo/index.html:120` [—](index.html#L120) `health fetch` (search `HEALTH`).

After construction, the server listens on the GPU host's `127.0.0.1:7860`. The laptop's SSH tunnel forwards its own `127.0.0.1:7860` to that remote loopback socket. Opening the local URL retrieves `index.html` through `Handler.do_GET`. Its JavaScript fetches `/health`; success enables Compare and displays the GPU/context information. The hostname in the browser is local, but inference executes on the remote GPU.

The page is plain HTML/CSS/JavaScript. There is no frontend build step or browser-side model runtime.

## 7. Press Compare: construct and transmit the request

**Open:** `demo/index.html:60` [—](index.html#L60) `form.onsubmit` (search `SUBMIT`).

The submit handler trims the prompt, creates a turn with two empty cards, and builds `pending` from completed conversation history plus this new user message. It installs an `AbortController`, disables input controls, and sends one JSON request:

```json
{
  "messages": [{"role": "user", "content": "Explain why a KV cache helps."}],
  "max_new_tokens": 128,
  "gamma": 4
}
```

The relative URL `/api/chat` goes to the same server that served the page. The browser sends text and controls, not token IDs or model parameters. The same request will generate both cards. The right card waits while the left card runs.

## 8. HTTP checks, ownership of the engine, and tokenization

**Open:** `demo/server.py:252` [—](server.py#L252) `Handler.do_POST` (search `HTTP`), `demo/server.py:24` [—](server.py#L24) `validate_request` (search `VALIDATE`), `demo/server.py:121` [—](server.py#L121) `KernelEngine.prepare` (search `TOKENIZE`).

`Handler.do_POST` checks the route and, when present, requires Origin to match the page's host. It tries to acquire the engine lock without waiting. A concurrent request gets HTTP 409 because the engine has one mutable pair of caches, even though the HTTP server can create multiple handler threads.

Inside the lock, the handler validates the body size (up to 65536 bytes), JSON content type, and payload. Messages must alternate user/assistant, start and end with user, contain nonempty bounded text, and total at most 40 messages. The token budget must be 1–256; gamma must be 1–8. The browser offers a smaller set of convenient choices.

`prepare` prepends the system message, applies the target tokenizer's chat template, and requests an assistant-generation prefix. Role delimiters and that prefix are part of the prompt tokens. It rejects `prompt_length + output_budget > context_capacity`, then rejects IDs outside the smaller draft vocabulary. Only then does it create a CUDA int64 tensor of shape `[1, P]`, where `P` is prompt length.

Both generation paths receive this same tensor. The server sends success headers for a newline-delimited JSON stream, creates the generation iterator, and begins consuming it. Invalid inputs before headers get an ordinary HTTP error; failures after streaming starts use an `error` event where possible.

## 9. The left card runs first

**Open:** `demo/server.py:188` [—](server.py#L188) `KernelEngine.generate` (search `COMPARE`), `demo/server.py:138` [—](server.py#L138) `KernelEngine.target_chunks` (search `BASELINE`), `runtime/executor.py:488` [—](../runtime/executor.py#L488) `decode_step` (search `DECODE`), `demo/server.py:155` [—](server.py#L155) `KernelEngine.run_mode` (search `TEXT`).

`KernelEngine.generate` enters `torch.inference_mode()` and loops over `target`, then `speculative`. These are serialized executions on the shared GPU. The engine yields a `start` event before each mode. Model work is lazy: creating a generator does not execute its body until it is advanced.

For target-only generation, `target_chunks` calls `executor.prefill(prompt)` and chooses the argmax of its final logit row. That is the first response token. It yields that one ID immediately. Unless EOS or the output limit has been reached, it passes the chosen ID through `decode_step`, whose new logits predict the following token. It repeats one token at a time.

This matters for both algorithms: **the logits produced while processing token X predict the token after X.** You do not run an extra decode to get the first token after the prompt; prompt prefill already supplies its scores. Likewise, the final emitted token needs no subsequent decode if generation stops.

`run_mode` converts IDs to cumulative text and emits the events explained in stop 21. The left card finishes, its cache is reset, and its token IDs are retained in a CPU list for comparison. The speculative pass then starts from the original prompt again. It does not reuse the left run's answer or its prompt cache.

## 10. Start the right card: prefill both models

**Open:** `demo/decoding.py:8` [—](decoding.py#L8) `speculative_chunks` (search `SPEC_LOOP`), `runtime/executor.py:396` [—](../runtime/executor.py#L396) `Qwen2Executor.prefill` (search `PREFILL`), `runtime/speculative/draft_runner.py:19` [—](../runtime/speculative/draft_runner.py#L19) `DraftRunner.prefill` (search `DRAFT_PREFILL`).

`speculative_chunks` first calls `target.prefill(prompt)`, then `draft.prefill(prompt)`. These are full prompt forwards, each using its own weights and caches. `prefill` resets its cache, checks bounds, embeds the IDs, runs the decoder stack, saves the last logit row as `p1_logits`, and advances the cache cursor to `P`.

The target retains `p1_logits` for verifying the first proposed token. `DraftRunner.prefill` clones its own final row into `_last_logits` so it can choose the first proposal. Both caches now represent exactly the prompt.

We now pause the lifecycle at this first model forward and trace every stage. In the next stops, `S` is the number of input rows: `P` for prefill, 1 for ordinary decode, and gamma or gamma+1 for target verification. Unless stated otherwise, shapes below describe the target. The draft follows the same sequence with the smaller dimensions in the table.

## 11. Embedding and input RMSNorm

**Open:** `runtime/production_kernels/target/embedding/ops.py:9` [—](../runtime/production_kernels/target/embedding/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_EMBEDDING_WRAPPER`), `runtime/production_kernels/target/embedding/bindings.cpp:13` [—](../runtime/production_kernels/target/embedding/bindings.cpp#L13) `PYBIND11_MODULE` (search `TARGET_EMBEDDING_BINDING`), `runtime/production_kernels/target/embedding/kernel.cu:22` [—](../runtime/production_kernels/target/embedding/kernel.cu#L22) `embedding_kernel and embedding_forward` (search `TARGET_EMBED`).

Embedding maps `[1, S]` token IDs into `[1, S, H]` hidden vectors. The C++ host function checks CUDA placement, int64 IDs, FP16 weights, and contiguous storage; allocates the output; then launches `embedding_kernel` with one block per token and 128 threads. The threads stride across the selected vocabulary row. Adjacent threads copy adjacent channels, allowing coalesced memory accesses. This is an indexed gather, not a matrix multiplication.

The result enters `_forward_stack`, which calls `_run_decoder_layer` once per layer.

**Open:** `runtime/executor.py:306` [—](../runtime/executor.py#L306) `_run_decoder_layer` (search `LAYER`), `runtime/production_kernels/target/rmsnorm/ops.py:9` [—](../runtime/production_kernels/target/rmsnorm/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_RMSNORM_WRAPPER`), `runtime/production_kernels/target/rmsnorm/bindings.cpp:12` [—](../runtime/production_kernels/target/rmsnorm/bindings.cpp#L12) `PYBIND11_MODULE` (search `TARGET_RMSNORM_BINDING`), `runtime/production_kernels/target/rmsnorm/kernel.cu:44` [—](../runtime/production_kernels/target/rmsnorm/kernel.cu#L44) `rmsnorm_forward_kernel_vectorized and rmsnorm_forward` (search `TARGET_NORM`).

The layer first saves `residual_1 = hidden`. Its first RMSNorm creates a normalized copy for attention:

```text
y[j] = x[j] * rsqrt(mean(x*x) + epsilon) * weight[j]
```

This is a mean square, not mean subtraction. Each token row is normalized independently across its hidden channels. One GPU block handles one row. The kernel moves eight FP16 elements at a time through 16-byte `float4` loads/stores; this reinterpretation is a memory-access technique, not conversion of the entire tensor to FP32. The arithmetic unpacks half pairs into FP32 registers.

Each thread sums squares for its elements. Warp shuffle reductions combine partial sums, warp leaders write shared memory, and the first warp finishes the block reduction. Thread 0 computes the reciprocal square root and shares it. Threads reread input and learned scale weights, normalize, and store FP16 results. For target width 3584, the launcher selects 448 threads: eight values per thread. The draft width uses the same algorithm with the launcher's fallback thread count.

The unnormalized residual is preserved because the branch output will be added back to it.

## 12. QKV projection, head layout, RoPE, and KV writes

**Open:** `runtime/executor.py:222` [—](../runtime/executor.py#L222) `_run_attention` (search `ATTENTION`), `runtime/production_kernels/target/attention/kernel.cu:35` [—](../runtime/production_kernels/target/attention/kernel.cu#L35) `qkv_proj_forward` (search `TARGET_QKV`).

`_run_attention` flattens `[1, S, H]` to `[S, H]` and projects with packed QKV weights. For the target:

```text
X:       [S, 3584]
W_qkv:   [4608, 3584]       (3584 Q + 512 K + 512 V channels)
Y:       [S, 4608]
```

The host function allocates `Y`, fills it with the broadcast bias, and calls `cublasGemmEx` with beta=1 so the matrix product is added to that bias. FP16 inputs/output use FP32 GEMM accumulation. cuBLAS chooses its implementation; this projection is not a handwritten GEMM. The transpose flags and swapped matrix dimensions reconcile PyTorch's row-major tensors with cuBLAS's column-major interface without explicitly transposing the weights in memory.

Python slices the result into Q, K, V, reshapes them into heads, transposes sequence/head axes, and calls `.contiguous()`:

```text
Q: [1, 28, S, 128]
K: [1,  4, S, 128]
V: [1,  4, S, 128]
```

These layout conversions are part of the workload; some require GPU copies rather than metadata-only views.

**Continue to:** `runtime/production_kernels/target/attention/kernel.cu:100` [—](../runtime/production_kernels/target/attention/kernel.cu#L100) `rope_kv_write_kernel and host launchers` (search `TARGET_KV_WRITE`) and the earlier RoPE-table stop.

For an eager forward, `write_pos` comes from the executor's host cache cursor. `rope_embeddings(write_pos, S)` selects the appropriate absolute-position rows from the precomputed tables. The RoPE/cache kernel writes the new K/V into this layer's cache interval `[write_pos, write_pos + S)`.

For a paired first-half/second-half key component `(x0, x1)`, RoPE computes `(x0*c - x1*s, x1*c + x0*s)`. The kernel performs this rotation as it writes K, saving a separate rotated-K tensor and kernel pass. V is copied unchanged using packed memory accesses. Q remains unrotated at this boundary; the attention kernel rotates it after loading it into shared memory.

There is one launch block per batch/KV-head pair and 16-row sequence tile. Short decode inputs use one tile; long prefills expose many independent blocks. The cache cursor is advanced by the executor after the **whole model forward**, not once per layer: every layer writes the same token positions into its own cache slice.

## 13. Attention: select the correct workload

**Open:** the dispatch branches in `runtime/executor.py:222` [—](../runtime/executor.py#L222) `_run_attention` (search `ATTENTION`), then `runtime/production_kernels/target/attention/kernel.cu:259` [—](../runtime/production_kernels/target/attention/kernel.cu#L259) `flash_attention_kernel and fused_attn_forward` (search `TARGET_PREFILL_ATTN`), `runtime/production_kernels/target/attention/kernel.cu:668` [—](../runtime/production_kernels/target/attention/kernel.cu#L668) `decode_attn_kernel` (search `TARGET_SMALL_ATTN`), and `runtime/production_kernels/target/attention/kernel.cu:1074` [—](../runtime/production_kernels/target/attention/kernel.cu#L1074) `choose_num_splits, launch_decode_attn, dispatch_decode_attn` (search `TARGET_DISPATCH`).

The executor always supplies new Q rows and the layer's cached K/V. It selects:


| Model operation    | Query rows `S`                 | Attention entry point                          |
| ------------------ | ------------------------------ | ---------------------------------------------- |
| Prompt prefill     | Prompt length                  | `fused_attn_forward`                           |
| Target-only decode | 1                              | `decode_attn_forward`                          |
| First verification | gamma                          | `small_q_attn_forward`, or decode entry if S=1 |
| Later verification | gamma+1, including prior bonus | `small_q_attn_forward`                         |
| Draft graph replay | 1                              | `decode_attn_forward_dev`                      |


All compute causal scaled dot-product attention: each query receives a weighted sum of permitted values, with weights from softmax of `QKᵀ / sqrt(D)`. Query head `h` maps to KV head `h // 7` here. Storing fewer KV heads saves cache space; the current attention blocks can still load the shared KV data separately for different query heads.

### Prompt prefill

`fused_attn_forward` launches the custom `flash_attention_kernel`. The target uses 64 query rows and 32 KV rows per tile, 128 threads (four warps), head width 128, and 16×16×16 WMMA matrix fragments. The draft uses head width 64.

For each query tile, the kernel:

1. Loads Q into shared memory with packed accesses and rotates it there using RoPE.
2. Iterates over tiles of cached K/V, staging them in shared memory.
3. Uses WMMA Tensor Core products to compute the score tiles in FP32 accumulators.
4. Applies scale and causal masking: future keys are assigned negative infinity.
5. Updates a running row maximum, exponential sum, and weighted-output accumulator.
6. Uses a second matrix product to accumulate the probability-weighted V contribution.
7. Normalizes the completed accumulator and writes FP16 attention output.

The running softmax state avoids materializing an entire prompt-length-squared score/probability matrix in global memory. When a later tile increases the maximum, prior sums and outputs are rescaled to the new reference maximum. Shared storage for K is reused for probabilities after K is no longer needed; barriers protect those reuse points. Padding/masks handle partial tiles.

This code accesses WMMA fragment elements explicitly, an architecture-sensitive implementation detail. Existing RTX 4090 validation supports that tested setup; the mere ability to compile for another architecture does not prove this indexing remains correct.

### Single-token decode and small verification batches

`decode_attn_kernel` handles only a few query rows and uses FP32 scalar dot products rather than the large WMMA query tile. It stages KV tiles in shared memory, computes scores, maintains online softmax state, accumulates values, and normalizes. The small-query launch dispatch specializes `Q_TOKENS` for S=1 through S=9 at build time. Selecting a case during inference does not invoke the compiler.

Causality is essential during verification. New K/V for **all** proposed rows have already been written, but query row `r` may see only keys at or before its absolute position:

```text
query_position = cur_len - S + r
allow key_position <= query_position
```

This lets a single forward process multiple known candidate input IDs without allowing a candidate to look at future candidates. It is why we can evaluate candidate continuations together, but cannot invent ordinary autoregressive tokens in parallel before their IDs are known.

### Split the cache when one block per head is too little work

**Continue to:** `runtime/production_kernels/target/attention/kernel.cu:869` [—](../runtime/production_kernels/target/attention/kernel.cu#L869) `decode_attn_split_kernel` (search `TARGET_SPLIT`) and `runtime/production_kernels/target/attention/kernel.cu:1022` [—](../runtime/production_kernels/target/attention/kernel.cu#L1022) `decode_attn_combine_kernel` (search `TARGET_COMBINE`).

`launch_decode_attn` can divide the attended KV range among multiple thread blocks. Each split computes an unnormalized partial output, its local maximum, and its exponential sum. The combine kernel rescales all partials to the common maximum, sums them, and divides by the combined denominator. Simply averaging normalized partial outputs would be incorrect.

The eager path chooses split count from the current length. The graph path chooses it from context **capacity**, keeping scratch shapes and launch geometry constant across replays; the kernel still partitions the actual current length on the device. Empty splits are handled in the combine step. A short eager context falls back to the single-block-per-head path.

The heuristic still targets roughly two waves on a historical 72-SM device. It is not an automatic RTX 4090 occupancy tuner. Correctness and good speed on a test prompt do not establish optimal launch tuning for every GPU/context.

## 14. Output projection, residuals, and the MLP

**Open:** `runtime/production_kernels/target/attention/kernel.cu:1296` [—](../runtime/production_kernels/target/attention/kernel.cu#L1296) `o_proj_forward` (search `TARGET_O_PROJ`), `runtime/production_kernels/target/residual_ops/ops.py:9` [—](../runtime/production_kernels/target/residual_ops/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_RESIDUAL_OPS_WRAPPER`), `runtime/production_kernels/target/residual_ops/bindings.cpp:19` [—](../runtime/production_kernels/target/residual_ops/bindings.cpp#L19) `PYBIND11_MODULE` (search `TARGET_RESIDUAL_OPS_BINDING`), `runtime/production_kernels/target/residual_ops/kernel.cu:49` [—](../runtime/production_kernels/target/residual_ops/kernel.cu#L49) `residual_add_kernel and residual_add_forward` (search `TARGET_ADD`).

Attention returns `[1, query_heads, S, D]`. Python transposes and reshapes that into `[S, H]`, then `o_proj_forward` calls cuBLAS for the bias-free output projection. Its result returns to `[1, S, H]`.

The first residual add combines this attention branch with `residual_1`. `residual_add_kernel` loads eight FP16 elements at a time, uses four `half2` additions, and writes a fresh output. This is a standalone add; it is not fused with the following normalization. The result is saved as `residual_2`.

Post-attention RMSNorm now applies a different learned scale using the same RMSNorm kernel described above. Its normalized result feeds the MLP.

**Continue to:** `runtime/production_kernels/target/swiglu/ops.py:9` [—](../runtime/production_kernels/target/swiglu/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_SWIGLU_WRAPPER`), `runtime/production_kernels/target/swiglu/bindings.cpp:25` [—](../runtime/production_kernels/target/swiglu/bindings.cpp#L25) `PYBIND11_MODULE` (search `TARGET_SWIGLU_BINDING`), `runtime/production_kernels/target/swiglu/kernel.cu:179` [—](../runtime/production_kernels/target/swiglu/kernel.cu#L179) `swiglu_forward` (search `TARGET_MLP`), `runtime/production_kernels/target/swiglu/kernel.cu:86` [—](../runtime/production_kernels/target/swiglu/kernel.cu#L86) `swiglu_act_kernel` (search `TARGET_ACT`).

The SwiGLU wrapper flattens the batch/sequence dimensions. The C++ orchestration does:

```text
gate = X @ W_gate.T                 [S, H] -> [S, I]
up   = X @ W_up.T                   [S, H] -> [S, I]
gate = silu(gate) * up              [S, I], overwrite gate
out  = gate @ W_down.T              [S, I] -> [S, H]
```

The three projections are cuBLAS GEMMs. The custom `swiglu_act_kernel` fuses SiLU and multiplication, moving eight halves at a time and calculating `g / (1 + exp(-g)) * up` in FP32 registers before storing FP16. Writing into the gate buffer avoids an additional activation output allocation. There is no separate sigmoid tensor. “Fused SwiGLU” here does not mean all three GEMMs and activation are one GPU kernel.

The wrapper restores `[1, S, H]`. The second residual add combines the MLP result with `residual_2`. This completes **one** decoder layer:

```text
input -> RMSNorm -> attention -> add input residual
      -> RMSNorm -> SwiGLU MLP -> add post-attention residual
```

It then repeats this sequence for every layer, using that layer's parameters and K/V slice: 28 times for target or 24 for draft. Embedding runs once before the stack, not once per layer.

## 15. Final normalization, vocabulary logits, and the return to Python

**Open:** `runtime/executor.py:373` [—](../runtime/executor.py#L373) `_forward_stack` (search `STACK`) and `runtime/production_kernels/target/residual_ops/kernel.cu:135` [—](../runtime/production_kernels/target/residual_ops/kernel.cu#L135) `lm_head_forward` (search `TARGET_HEAD`).

After the last decoder layer, `_forward_stack` applies the final RMSNorm with `model.norm.weight`, then `lm_head_forward`. The vocabulary projection is another bias-free cuBLAS GEMM:

```text
[S, H] @ [V, H].T -> [S, V] -> [1, S, V]
```

The target uses its distinct LM-head matrix. The draft uses its embedding matrix. Logits are scores, not normalized probabilities. Greedy selection only needs the largest score, so there is no vocabulary softmax/sampling stage in the demo. Attention softmax still exists inside the attention kernels.

The current implementation projects all S rows, even during prefill when only the last row is ultimately needed for generation. This is a concrete possible optimization, not something already implemented.

C++ returns tensors whose storage is on the GPU. Most launches are asynchronous from the CPU's perspective, and use PyTorch's current CUDA stream so dependent operations execute in order. `argmax` is a PyTorch GPU operation; `.item()` or `.tolist()` brings a small result to Python and requires the needed GPU work to finish. Python coordinates those decisions. There is no custom fused greedy sampler in this branch.

The prefill call now advances its cache cursor to P. Target saves its first-next-token logits, and draft saves its own copy. We resume the speculative lifecycle.

## 16. Generate four draft tokens with a reusable CUDA graph

**Open:** `runtime/speculative/draft_runner.py:26` [—](../runtime/speculative/draft_runner.py#L26) `DraftRunner.generate_drafts` (search `DRAFT`), `runtime/executor_graph.py:47` [—](../runtime/executor_graph.py#L47) `_prepare_decode_static` (search `GRAPH_INPUT`), `runtime/executor_graph.py:62` [—](../runtime/executor_graph.py#L62) `_decode_forward_static` (search `GRAPH_FORWARD`), `runtime/executor_graph.py:83` [—](../runtime/executor_graph.py#L83) `_capture_decode_graph` (search `GRAPH_CAPTURE`), `runtime/executor_graph.py:114` [—](../runtime/executor_graph.py#L114) `decode_step_graph` (search `GRAPH_REPLAY`), `runtime/buffers.py:153` [—](../runtime/buffers.py#L153) `RuntimeBuffers.static_rope` (search `GRAPH_ROPE`).

`generate_drafts(4)` records the draft's starting cache position in `_prefix_len` and takes argmax of `_last_logits` to choose `d1`. It calls `_decode(d1)`, which creates a one-token CUDA tensor and invokes `decode_step_graph`. That forward writes d1's K/V and returns logits predicting d2. The loop repeats until it has d1…d4. It also decodes d4, so **all four** candidates have tentative entries in the draft cache, even though it does not need d4's next-token logits for another proposal in this round.

There is serial dependence between draft tokens. The CUDA graph reduces repeated host launch overhead; it does not make those four token choices independent or merge four model forwards into one.

For every graph decode:

1. `_prepare_decode_static` copies the chosen ID into a fixed-address input tensor and updates device scalars for `cache_position` and `static_cur_len`.
2. On the first use only, `_capture_decode_graph` warms the static forward three times on a side stream, synchronizes, and records it using `torch.cuda.graph`. Startup warmup normally pays this cost before the page is ready.
3. `_decode_graph.replay()` submits the recorded operation sequence.
4. The executor advances the cache cursor by one and returns the captured logits buffer.

The captured region includes embedding, all layers, the final norm, and LM head. `static_rope` gathers the correct rows inside the graph using the current device position. RoPE/cache writes use the `_dev` entry point, and attention reads `cur_len` from device memory. Passing these as ordinary capture-time host integers would freeze their values and corrupt later positions.

**GPU counterpart:** `runtime/production_kernels/draft/attention/kernel.cu:1251` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1251) `decode_attn_forward_dev` (search `DRAFT_DEV`), `runtime/production_kernels/draft/attention/kernel.cu:100` [—](../runtime/production_kernels/draft/attention/kernel.cu#L100) `rope_kv_write_kernel and host launchers` (search `DRAFT_KV_WRITE`), and `runtime/production_kernels/draft/attention/kernel.cu:1068` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1068) `choose_num_splits, launch_decode_attn, dispatch_decode_attn` (search `DRAFT_DISPATCH`).

Graph replay does not rerun the Python layer loop or C++ launch decisions. Fixed shapes/addresses and the graph pool keep the intermediate storage usable. Work outside the captured region still exists: creating/copying the token tensor, changing scalar values, replay submission, greedy argmax, and CPU decisions. Returned graph logits are a **live reused buffer**, so the draft consumes each argmax before the next replay. It clones logits that must survive into a later round.

At the end, `generate_drafts` returns `[1, 4]` IDs on CUDA. They are guesses, not yet text sent to the user.

## 17. Verify a block with one target forward

**Open:** `runtime/speculative/target_step.py:9` [—](../runtime/speculative/target_step.py#L9) `greedy_target_step` (search `VERIFY`), `runtime/executor.py:422` [—](../runtime/executor.py#L422) `verify_gamma` (search `VERIFY_FORWARD`).

`greedy_target_step` takes any pending bonus from the target. In the first round there is none, and the target's cache cursor is P. It calls `verify_gamma([d1,d2,d3,d4])`.

`verify_gamma` embeds the four known candidate IDs, runs the same complete target stack, and advances the cursor tentatively to P+4. The sequence dimension is 4, so attention uses the small-query verification path. Dense projections process four rows together, reusing target weights across those rows within the GEMM. The target does not perform four separate Python-level decode forwards.

The logit alignment is easy to get wrong:


| Logit source                            | Predicts                      |
| --------------------------------------- | ----------------------------- |
| Saved final prompt row (`p1_logits`)    | d1                            |
| Verification output after processing d1 | d2                            |
| Verification output after processing d2 | d3                            |
| Verification output after processing d3 | d4                            |
| Verification output after processing d4 | A token after all four drafts |


For the first round, the verifier prepends `p1_logits` to the returned rows. For each i, it accepts di only if it equals the target argmax of the row **before** di. It stops at the first mismatch. Rows conditioned on a rejected token cannot validate later tokens on the accepted path.

If n draft tokens match, the verifier chooses a target argmax at row n as the **bonus**. If n<gamma, this replaces the first rejected draft. If n=gamma, this is an additional token after the full accepted block. Even when n=0, the target supplies one token, so a successful round always makes progress.

No acceptance-ratio sampling or random draws occur here. Both paths are greedy. With matching numerical choices, the verified sequence follows target-only greedy decoding. Different target attention/GEMM shapes can still cause small FP16 numerical differences around near-tied logits; the demo therefore compares actual generated IDs instead of assuming equality.

## 18. Roll back, defer the target bonus, and yield

**Open:** `runtime/executor.py:183` [—](../runtime/executor.py#L183) `rollback_cache` (search `ROLLBACK`), `runtime/executor.py:161` [—](../runtime/executor.py#L161) `defer_bonus_token` (search `BONUS`), then return to `demo/decoding.py:8` [—](decoding.py#L8) `speculative_chunks` (search `SPEC_LOOP`).

The target has temporarily written K/V for every input candidate. It rolls the valid cursor back to the accepted prefix, excluding the new bonus. Rollback changes the host cursor and its device mirror; it does not copy/compact accepted data or clear rejected rows. Later attention is bounded by the valid length and causal mask, and new writes overwrite stale suffix slots.

The target stores the bonus as a pending Python integer. It has logically chosen that output token but has **not run the token through the target to produce its K/V**. In the next round, verification will prepend this bonus to the next candidates. This saves a separate target forward just to commit the bonus.

`speculative_chunks` builds a CPU ID list from the accepted draft prefix followed by the bonus. It trims to the remaining output budget, then through the first EOS if present. It yields that verified block with accepted/proposed/round counters. The word “token” in a wire event does not imply this block has length one.

The generator suspends at `yield`. Text conversion and the HTTP write happen before it resumes and applies draft feedback. This ordering gets verified text on its way to the browser earlier.

## 19. Follow the first three blocks with exact cache positions

This is an **illustrative** trace with prompt length P and gamma=4. Letters are token IDs; assume no EOS and enough output budget. “Valid cache” counts tokens with materialized K/V, not merely tokens displayed on screen.

### Round 1: accept two and replace the third

Draft proposes `[a,b,x,y]`. Target's first choices agree on a and b but prefer c where the draft proposed x.

- Draft tentatively has P+4 tokens in cache.
- Target verifies four rows, temporarily reaching P+4.
- The verifier accepts 2, selects bonus c, rolls target back to P+2, and marks c pending.
- First output block is `[a,b,c]`. The right card can now display its decoded text.

**Open:** `runtime/speculative/draft_runner.py:46` [—](../runtime/speculative/draft_runner.py#L46) `DraftRunner.apply_target_feedback` (search `FEEDBACK`).

When the generator resumes, `apply_target_feedback(2,c)` rolls draft back to P+2, decodes c through the draft graph, and clones the resulting next-token logits. Draft now has P+3 valid tokens. Target still has P+2 plus pending c. This one-token difference is intentional.

### Round 2: accept all four and add one more

Draft now proposes `[d,e,f,g]` conditioned on prompt+a+b+c. Its cache temporarily reaches P+7.

Target consumes pending c and performs **one** forward over `[c,d,e,f,g]`, temporarily reaching P+7. The output after c predicts d. There is no need to use the old prompt logits this round: the five output rows directly supply scores for d, e, f, g, and the following bonus.

Suppose all four match and the target's next token is h. The target rollback base includes the leading c, so it retains P+7 rows, defers h, and emits `[d,e,f,g,h]`. The browser replaces cumulative text with the decoding of `[a,b,c,d,e,f,g,h]`.

Draft feedback keeps the four accepted proposals and decodes h, reaching P+8. Target remains at P+7 plus pending h. All-accepted rounds can therefore advance output by gamma+1 tokens.

### Round 3: reject the first proposal

Suppose draft proposes `[z,u,v,w]` from P+8. Target forwards `[h,z,u,v,w]` from P+7, temporarily reaching P+12. Its first output row, after h, prefers i over z.

It accepts zero, retains only the already-chosen leading h (valid cursor P+8), defers i, and emits `[i]`. No later draft is accepted even if a later row happens to match. Feedback rolls draft from P+12 back to P+8, then decodes i to reach P+9.


| After block, once draft feedback finishes | Cumulative output count | Target valid cursor | Target pending | Draft valid cursor |
| ----------------------------------------- | ----------------------- | ------------------- | -------------- | ------------------ |
| `[a,b,c]`                                 | 3                       | P+2                 | c              | P+3                |
| `[d,e,f,g,h]`                             | 8                       | P+7                 | h              | P+8                |
| `[i]`                                     | 9                       | P+8                 | i              | P+9                |


At each next target verification, its leading bonus closes the one-token gap. The bonus is already trusted target output and is **not re-verified** as a draft. `_last_logits` on the draft describes the full accepted output prefix, so its next proposals use the correct context.

If a target bonus is outside the draft embedding vocabulary, feedback raises a clear error. This can happen after a verified block has already been streamed; the UI then reports an incomplete comparison. Shared tokenizer IDs are checked, but the matrix row-count difference remains a real constraint.

## 20. Remaining budget, EOS, and the final bonus

Return to `demo/decoding.py:8` [—](decoding.py#L8) `speculative_chunks` (search `SPEC_LOOP`) and `runtime/executor.py:149` [—](../runtime/executor.py#L149) `reset_kv_cache` (search `RESET`).

Each round uses `proposed = min(gamma, limit - emitted)`. The previous bonus already counts in emitted; if it is pending on target, the target cache is one row behind and bundling that bonus restores the alignment. The proposal count and context bound keep the tentative forward within the allowed capacity.

A verifier can return gamma+1 IDs, so the adapter trims the block to the remaining budget. It also ends the block at the first EOS. Accepted/proposed counters measure the verification decision; in the final trimmed block they need not count only visibly retained tokens. EOS IDs count toward token totals even though text decoding skips special tokens.

On a finished block, the generator breaks without draft feedback. Its `finally` resets both caches. There is no reason to spend a final model forward committing a bonus into a cache that is about to be discarded. Ending with a pending target bonus is therefore intentional.

## 21. From verified IDs to a visible right-hand response

**Open:** `demo/server.py:155` [—](server.py#L155) `KernelEngine.run_mode` (search `TEXT`), `demo/server.py:282` [—](server.py#L282) `Handler streaming writes` (search `WIRE`), `demo/index.html:71` [—](index.html#L71) `response stream parser` (search `READ_STREAM`), `demo/index.html:87` [—](index.html#L87) `panel event rendering` (search `RENDER`).

After `speculative_chunks` yields, `run_mode`:

1. Records the first produced-block time if this is the first chunk.
2. Appends IDs to its cumulative `generated` list and adds round counters.
3. Calls the target tokenizer's `decode(generated, skip_special_tokens=True)` on the CPU.
4. Emits a `token` event containing the **whole response so far**, unless its text currently ends in the Unicode replacement character `\ufffd`.

Decoding the full prefix handles token pieces whose text only becomes complete after subsequent tokens. Skipping an incomplete trailing character may delay a visible update beyond the first internally produced block. A final `done` event decodes the full result again.

A representative wire event is:

```json
{"type":"token","mode":"speculative","text":"The response so far","output_tokens":8,"accepted":6,"proposed":8,"rounds":2}
```

The handler serializes the event, appends a newline, writes it to the socket, and flushes. It is ordinary newline-delimited JSON, not WebSocket or Server-Sent Events. The tunnel carries it back to the laptop. Network buffering can combine several events or split one event across reads; a flush does not guarantee a separate browser paint for every block.

The browser's `TextDecoder` preserves UTF-8 state across reads. Its string buffer retains an unfinished JSON line. Each complete line is parsed; `result.mode` selects a card. `start` shows Thinking, `token` replaces `card.answer.textContent` and updates counters, and `done` marks the card complete and shows final metrics.

Because text is cumulative, assigning it replaces the earlier prefix rather than appending the same words again. `textContent` displays model text literally rather than interpreting it as HTML. The browser may paint the first three verified blocks as three updates or combine updates if events arrive faster than rendering.

## 22. Timing, comparison, and the next query

Return to `demo/server.py:155` [—](server.py#L155) `KernelEngine.run_mode` (search `TEXT`), `demo/server.py:188` [—](server.py#L188) `KernelEngine.generate` (search `COMPARE`), and `demo/index.html:110` [—](index.html#L110) `submit finally/history` (search `HISTORY`).

`run_mode` synchronizes CUDA before starting its timer and after consuming the iterator. Its elapsed time includes that mode's prefill, generation, text conversion, the streaming work while the generator is suspended, and generator cleanup/cache reset. Tokenization in `prepare`, model loading, graph capture during startup, and waiting for the other mode are outside this timer. It is an application-level measurement, not a sum of isolated CUDA kernel times or a browser paint measurement.

TTFT measures when the first internal ID chunk becomes available, before its tokenizer decode/network delivery. On the target side that chunk has one token; on the speculative side it is the first verified block. Skipped incomplete-character updates can make the first visible text later. Throughput divides generated token count by this elapsed time; it includes prefill and counts special IDs even if hidden in text.

`generate` saves each mode's complete token-ID list internally, removes those private IDs from the streamed `done` object, then compares the two lists exactly. Its final `comparison` event reports that boolean and target elapsed divided by speculative elapsed. The UI shows either speedup or slowdown and warns if tokens differ. Draft overhead and acceptance rate mean speculative decoding is not guaranteed to win on every prompt.

The page adds only a completed target-only answer to conversation history. Even if the speculative run was stopped later, a completed baseline can become the next assistant message. A partial baseline does not. Follow-ups resend full text history, which is tokenized and prefills both models again. There is no persistent conversational KV session or prefix-cache reuse across HTTP requests.

## 23. Stop, errors, and cleanup

The Stop button calls `AbortController.abort` in the browser (next to the submit marker). The server discovers the disconnect on socket I/O; this is not an immediate GPU-kernel interrupt. An in-flight model operation can finish before the server notices.

The handler closes the generation iterator in `finally` and then releases the lock. Nested `closing(...)` and generator `finally` blocks reset target/draft caches even when generation stops early. `reset_kv_cache` clears cache contents, cursors, saved target first-token logits, and pending bonus state. The weights, RoPE tables, and captured draft graph remain resident for reuse. The next draft prefill replaces its runner-level logits/state.

A streaming error leaves the baseline card visible if it completed and labels unfinished work accordingly. A future request starts from fresh cache state. One request owns the engine until this cleanup completes.

## 24. A route through the code for an interview

Once you have read the full path, the following order makes a shorter live walkthrough possible:

1. Submit a query and show the serialized target/speculative cards.
2. Open `KernelEngine.generate` and `speculative_chunks` to identify the two algorithms and their shared prompt.
3. Show `DraftRunner.generate_drafts`, `greedy_target_step`, and the three-round cache table. Explain why accepted prefixes reproduce greedy target choices and why the bonus is deferred.
4. Show `Qwen2Executor._run_decoder_layer` as the model structure, then `_run_attention` as the mode switch.
5. Pick one custom kernel and show its Python wrapper → binding → launcher → device code. Then show the distinct prefill/small-query attention paths and the cuBLAS boundary.
6. Show graph inputs and replay to explain reducing repeated launch overhead without freezing positions.
7. Return to `run_mode` and the stream parser to connect verified IDs to visible output and interpret the timings accurately.

The longer lifecycle above remains the reference for what happens between those presentation stops.

## Source index: target and draft implementation counterparts

The lifecycle above walks the target implementation once. Draft prefill and graph decode use the corresponding draft files below, with smaller dimensions and the same overall math. These stops carry matching comments too. Wrapper/binding markers apply to the public functions immediately below/around them; device markers cover the named kernel and associated host launch path.


| Source boundary      | Target                                                                                                                                                                                                                    | Draft                                                                                                                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| embedding wrapper    | `runtime/production_kernels/target/embedding/ops.py:9` [—](../runtime/production_kernels/target/embedding/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_EMBEDDING_WRAPPER`)                               | `runtime/production_kernels/draft/embedding/ops.py:9` [—](../runtime/production_kernels/draft/embedding/ops.py#L9) `_load_ext and public op wrappers` (search `DRAFT_EMBEDDING_WRAPPER`)                               |
| embedding binding    | `runtime/production_kernels/target/embedding/bindings.cpp:13` [—](../runtime/production_kernels/target/embedding/bindings.cpp#L13) `PYBIND11_MODULE` (search `TARGET_EMBEDDING_BINDING`)                                  | `runtime/production_kernels/draft/embedding/bindings.cpp:13` [—](../runtime/production_kernels/draft/embedding/bindings.cpp#L13) `PYBIND11_MODULE` (search `DRAFT_EMBEDDING_BINDING`)                                  |
| embed                | `runtime/production_kernels/target/embedding/kernel.cu:22` [—](../runtime/production_kernels/target/embedding/kernel.cu#L22) `embedding_kernel and embedding_forward` (search `TARGET_EMBED`)                             | `runtime/production_kernels/draft/embedding/kernel.cu:22` [—](../runtime/production_kernels/draft/embedding/kernel.cu#L22) `embedding_kernel and embedding_forward` (search `DRAFT_EMBED`)                             |
| rmsnorm wrapper      | `runtime/production_kernels/target/rmsnorm/ops.py:9` [—](../runtime/production_kernels/target/rmsnorm/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_RMSNORM_WRAPPER`)                                     | `runtime/production_kernels/draft/rmsnorm/ops.py:9` [—](../runtime/production_kernels/draft/rmsnorm/ops.py#L9) `_load_ext and public op wrappers` (search `DRAFT_RMSNORM_WRAPPER`)                                     |
| rmsnorm binding      | `runtime/production_kernels/target/rmsnorm/bindings.cpp:12` [—](../runtime/production_kernels/target/rmsnorm/bindings.cpp#L12) `PYBIND11_MODULE` (search `TARGET_RMSNORM_BINDING`)                                        | `runtime/production_kernels/draft/rmsnorm/bindings.cpp:12` [—](../runtime/production_kernels/draft/rmsnorm/bindings.cpp#L12) `PYBIND11_MODULE` (search `DRAFT_RMSNORM_BINDING`)                                        |
| norm                 | `runtime/production_kernels/target/rmsnorm/kernel.cu:44` [—](../runtime/production_kernels/target/rmsnorm/kernel.cu#L44) `rmsnorm_forward_kernel_vectorized and rmsnorm_forward` (search `TARGET_NORM`)                   | `runtime/production_kernels/draft/rmsnorm/kernel.cu:44` [—](../runtime/production_kernels/draft/rmsnorm/kernel.cu#L44) `rmsnorm_forward_kernel_vectorized and rmsnorm_forward` (search `DRAFT_NORM`)                   |
| attention wrapper    | `runtime/production_kernels/target/attention/ops.py:13` [—](../runtime/production_kernels/target/attention/ops.py#L13) `_load_ext and public op wrappers` (search `TARGET_ATTENTION_WRAPPER`)                             | `runtime/production_kernels/draft/attention/ops.py:13` [—](../runtime/production_kernels/draft/attention/ops.py#L13) `_load_ext and public op wrappers` (search `DRAFT_ATTENTION_WRAPPER`)                             |
| attention binding    | `runtime/production_kernels/target/attention/bindings.cpp:120` [—](../runtime/production_kernels/target/attention/bindings.cpp#L120) `PYBIND11_MODULE` (search `TARGET_ATTENTION_BINDING`)                                | `runtime/production_kernels/draft/attention/bindings.cpp:120` [—](../runtime/production_kernels/draft/attention/bindings.cpp#L120) `PYBIND11_MODULE` (search `DRAFT_ATTENTION_BINDING`)                                |
| qkv                  | `runtime/production_kernels/target/attention/kernel.cu:35` [—](../runtime/production_kernels/target/attention/kernel.cu#L35) `qkv_proj_forward` (search `TARGET_QKV`)                                                     | `runtime/production_kernels/draft/attention/kernel.cu:35` [—](../runtime/production_kernels/draft/attention/kernel.cu#L35) `qkv_proj_forward` (search `DRAFT_QKV`)                                                     |
| kv write             | `runtime/production_kernels/target/attention/kernel.cu:100` [—](../runtime/production_kernels/target/attention/kernel.cu#L100) `rope_kv_write_kernel and host launchers` (search `TARGET_KV_WRITE`)                       | `runtime/production_kernels/draft/attention/kernel.cu:100` [—](../runtime/production_kernels/draft/attention/kernel.cu#L100) `rope_kv_write_kernel and host launchers` (search `DRAFT_KV_WRITE`)                       |
| prefill attn         | `runtime/production_kernels/target/attention/kernel.cu:259` [—](../runtime/production_kernels/target/attention/kernel.cu#L259) `flash_attention_kernel and fused_attn_forward` (search `TARGET_PREFILL_ATTN`)             | `runtime/production_kernels/draft/attention/kernel.cu:251` [—](../runtime/production_kernels/draft/attention/kernel.cu#L251) `flash_attention_kernel and fused_attn_forward` (search `DRAFT_PREFILL_ATTN`)             |
| small attn           | `runtime/production_kernels/target/attention/kernel.cu:668` [—](../runtime/production_kernels/target/attention/kernel.cu#L668) `decode_attn_kernel` (search `TARGET_SMALL_ATTN`)                                          | `runtime/production_kernels/draft/attention/kernel.cu:663` [—](../runtime/production_kernels/draft/attention/kernel.cu#L663) `decode_attn_kernel` (search `DRAFT_SMALL_ATTN`)                                          |
| split                | `runtime/production_kernels/target/attention/kernel.cu:869` [—](../runtime/production_kernels/target/attention/kernel.cu#L869) `decode_attn_split_kernel` (search `TARGET_SPLIT`)                                         | `runtime/production_kernels/draft/attention/kernel.cu:863` [—](../runtime/production_kernels/draft/attention/kernel.cu#L863) `decode_attn_split_kernel` (search `DRAFT_SPLIT`)                                         |
| combine              | `runtime/production_kernels/target/attention/kernel.cu:1022` [—](../runtime/production_kernels/target/attention/kernel.cu#L1022) `decode_attn_combine_kernel` (search `TARGET_COMBINE`)                                   | `runtime/production_kernels/draft/attention/kernel.cu:1016` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1016) `decode_attn_combine_kernel` (search `DRAFT_COMBINE`)                                   |
| dispatch             | `runtime/production_kernels/target/attention/kernel.cu:1074` [—](../runtime/production_kernels/target/attention/kernel.cu#L1074) `choose_num_splits, launch_decode_attn, dispatch_decode_attn` (search `TARGET_DISPATCH`) | `runtime/production_kernels/draft/attention/kernel.cu:1068` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1068) `choose_num_splits, launch_decode_attn, dispatch_decode_attn` (search `DRAFT_DISPATCH`) |
| dev                  | `runtime/production_kernels/target/attention/kernel.cu:1260` [—](../runtime/production_kernels/target/attention/kernel.cu#L1260) `decode_attn_forward_dev` (search `TARGET_DEV`)                                          | `runtime/production_kernels/draft/attention/kernel.cu:1251` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1251) `decode_attn_forward_dev` (search `DRAFT_DEV`)                                          |
| o proj               | `runtime/production_kernels/target/attention/kernel.cu:1296` [—](../runtime/production_kernels/target/attention/kernel.cu#L1296) `o_proj_forward` (search `TARGET_O_PROJ`)                                                | `runtime/production_kernels/draft/attention/kernel.cu:1287` [—](../runtime/production_kernels/draft/attention/kernel.cu#L1287) `o_proj_forward` (search `DRAFT_O_PROJ`)                                                |
| residual ops wrapper | `runtime/production_kernels/target/residual_ops/ops.py:9` [—](../runtime/production_kernels/target/residual_ops/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_RESIDUAL_OPS_WRAPPER`)                      | `runtime/production_kernels/draft/residual_ops/ops.py:9` [—](../runtime/production_kernels/draft/residual_ops/ops.py#L9) `_load_ext and public op wrappers` (search `DRAFT_RESIDUAL_OPS_WRAPPER`)                      |
| residual ops binding | `runtime/production_kernels/target/residual_ops/bindings.cpp:19` [—](../runtime/production_kernels/target/residual_ops/bindings.cpp#L19) `PYBIND11_MODULE` (search `TARGET_RESIDUAL_OPS_BINDING`)                         | `runtime/production_kernels/draft/residual_ops/bindings.cpp:19` [—](../runtime/production_kernels/draft/residual_ops/bindings.cpp#L19) `PYBIND11_MODULE` (search `DRAFT_RESIDUAL_OPS_BINDING`)                         |
| add                  | `runtime/production_kernels/target/residual_ops/kernel.cu:49` [—](../runtime/production_kernels/target/residual_ops/kernel.cu#L49) `residual_add_kernel and residual_add_forward` (search `TARGET_ADD`)                   | `runtime/production_kernels/draft/residual_ops/kernel.cu:50` [—](../runtime/production_kernels/draft/residual_ops/kernel.cu#L50) `residual_add_kernel and residual_add_forward` (search `DRAFT_ADD`)                   |
| head                 | `runtime/production_kernels/target/residual_ops/kernel.cu:135` [—](../runtime/production_kernels/target/residual_ops/kernel.cu#L135) `lm_head_forward` (search `TARGET_HEAD`)                                             | `runtime/production_kernels/draft/residual_ops/kernel.cu:136` [—](../runtime/production_kernels/draft/residual_ops/kernel.cu#L136) `lm_head_forward` (search `DRAFT_HEAD`)                                             |
| swiglu wrapper       | `runtime/production_kernels/target/swiglu/ops.py:9` [—](../runtime/production_kernels/target/swiglu/ops.py#L9) `_load_ext and public op wrappers` (search `TARGET_SWIGLU_WRAPPER`)                                        | `runtime/production_kernels/draft/swiglu/ops.py:9` [—](../runtime/production_kernels/draft/swiglu/ops.py#L9) `_load_ext and public op wrappers` (search `DRAFT_SWIGLU_WRAPPER`)                                        |
| swiglu binding       | `runtime/production_kernels/target/swiglu/bindings.cpp:25` [—](../runtime/production_kernels/target/swiglu/bindings.cpp#L25) `PYBIND11_MODULE` (search `TARGET_SWIGLU_BINDING`)                                           | `runtime/production_kernels/draft/swiglu/bindings.cpp:25` [—](../runtime/production_kernels/draft/swiglu/bindings.cpp#L25) `PYBIND11_MODULE` (search `DRAFT_SWIGLU_BINDING`)                                           |
| mlp                  | `runtime/production_kernels/target/swiglu/kernel.cu:179` [—](../runtime/production_kernels/target/swiglu/kernel.cu#L179) `swiglu_forward` (search `TARGET_MLP`)                                                           | `runtime/production_kernels/draft/swiglu/kernel.cu:179` [—](../runtime/production_kernels/draft/swiglu/kernel.cu#L179) `swiglu_forward` (search `DRAFT_MLP`)                                                           |
| act                  | `runtime/production_kernels/target/swiglu/kernel.cu:86` [—](../runtime/production_kernels/target/swiglu/kernel.cu#L86) `swiglu_act_kernel` (search `TARGET_ACT`)                                                          | `runtime/production_kernels/draft/swiglu/kernel.cu:86` [—](../runtime/production_kernels/draft/swiglu/kernel.cu#L86) `swiglu_act_kernel` (search `DRAFT_ACT`)                                                          |




## Current implementation boundaries worth being able to explain

- This is greedy speculative decoding. There are no temperature controls, stochastic rejection corrections, or MPI workers in this path.
- Both models remain resident, but draft proposal and target verification execute sequentially. The browser baseline and speculative runs are also serialized.
- Target decode and verification are eager. Draft one-token decode is graph-replayed; both model prefills are eager.
- Custom CUDA handles embedding, normalization, residual add, the fused SwiGLU activation, fused RoPE/cache writes, and attention. cuBLAS handles the dense projections; PyTorch also supplies layout/copy/indexing, argmax, and scalar/tensor operations.
- This runtime allocates eager intermediates and projects every prefill row to logits. It is not allocation-free, and its memory planner is not a peak-VRAM oracle.
- A bonus can be visible before its target KV row exists. Logical output, target cache position, and draft cache position are related but not always equal.
- Runtime success and identical outputs on tests validate those cases. FP16 near ties, vocabulary limits, context capacity, inherited launch heuristics, and architecture-sensitive WMMA indexing still constrain claims about correctness/performance on new configurations.
