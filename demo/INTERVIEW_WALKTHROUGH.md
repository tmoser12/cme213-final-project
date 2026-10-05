# Interview walkthrough: a custom Qwen inference engine

Use sections 1–6 as a spoken script, then section 7 to guide the live demo. The complete version is a roughly 25–30 minute rehearsal including the demo. For a shorter interview slot, summarize the layer sequence in section 4 and offer the individual kernel explanations on request; choose two optimization stories and keep the others ready for questions. Bracketed directions are rehearsal notes, not words to say. The final sections are backup answers and references.

This describes the current single-GPU demo: Qwen2.5-7B-Instruct target, Qwen2.5-0.5B-Instruct draft, FP16, batch one, custom CUDA kernels plus cuBLAS, and a captured draft decode graph. Use “we” for the project and identify your own contributions accurately if asked about ownership. The optimization chronology below follows the development process you described; only the recent measured results have explicit before/after numbers here.

## 1. Motivation and scope

“The question behind this project was: how much of interactive language-model inference could we understand and improve by building the execution path ourselves?

“I wanted to connect two levels of performance work. At the algorithm level, speculative decoding can reduce how often we need to run a large model sequentially. At the GPU level, those savings only matter if the draft is cheap and the kernels execute efficiently.

“We built an inference runtime for a 7B Qwen target and a 0.5B draft. It loads pretrained weights, executes the transformer layers, manages their KV caches, generates and verifies draft tokens, and streams the result into a demo. We didn’t train the models. Transformers supplies the tokenizer and checkpoint-related utilities; our runtime performs the model forward using custom CUDA operations and cuBLAS.

“I’ll first explain the design choices and the kernels, then a few optimization stories, and finish by showing the same prompt generated with and without speculation.”

## 2. The three ideas that shape the design



### KV caching: reuse the unchanged prefix

“Autoregressive generation adds one token at a time. In a causal transformer, an earlier token cannot attend to a later token. With a fixed prefix and positions, extending the sequence therefore doesn’t change the earlier tokens’ hidden states or the keys and values derived from them.

“We compute those keys and values once per layer and keep them. For the next token, we compute its new query, key, and value, append the new key and value, and attend over the valid cache.

“That saves recomputing the entire prefix through the model on every step. It doesn’t make attention constant time: the new query still has to read the historical keys and values, so decode attention gets more expensive as context grows.

“Each layer has its own cache, and target and draft have separate caches. Both models use grouped-query attention: seven query heads share each KV head. We store the smaller set of KV heads, rather than duplicating the cache for every query head.”

### Speculative decoding: spend cheap work to save expensive sequential work

“The small model proposes several tokens serially. Once those candidate IDs are known, the target can evaluate the candidate sequence in one causally masked forward. That lets the target reuse its weights across several token positions instead of making a separate forward call for each position.

“In our greedy implementation, we accept the longest prefix whose tokens match what the target would choose. At the first disagreement, we emit the target’s choice and discard the remaining guesses. If every guess matches, the target supplies an additional token.

“For example, suppose the draft proposes A, B, C, D. If the target agrees with A and B but would choose X next, we emit A, B, X. We cannot accept D using the result conditioned on C, because C is no longer part of the accepted prefix.

“The correctness argument is an induction over that accepted prefix: each emitted token is the target’s greedy choice conditioned on the tokens already accepted. In finite precision, different execution shapes can perturb nearly tied logits, so the demo checks the actual output IDs rather than assuming they match.

“The speedup depends on how many useful tokens we get per target verification relative to the cost of drafting and verification. Low acceptance can make speculation slower.”

[If asked for the logit alignment: the final prompt logit predicts A; the output after processing A predicts B. See the backup answers. This explanation is specific to our greedy verifier. The [speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html) also gives a stochastic algorithm that preserves the target distribution; we have not implemented that sampler in this demo.]

### Batch one: optimize the experience of one waiting user

“We focused on batch size one because the workload is interactive generation for one request. The objective was the latency that user sees, rather than maximum aggregate throughput across many independent requests.

“That exposes an interesting regime: the model does relatively little arithmetic for each weight it reads, and small operations can have very little parallel work. Launch overhead, memory traffic, and GPU occupancy become important.

“A larger request batch can improve utilization and reuse model weights across requests, but it changes the workload and can introduce waiting to assemble a batch. It doesn’t remove the dependency between successive tokens within one request.

“Speculation creates some extra work across candidate positions within that one request. Four proposed tokens are still one request, not four independently batched users. The demo server intentionally handles one generation request at a time.”

## 3. What we implemented, and what we delegated

“With finite time, we used cuBLAS for the standard dense projections and concentrated custom work on the surrounding operations and their integration with Qwen’s architecture.

“The custom pieces include embedding lookup, RMSNorm, RoPE and cache writes, attention for both prefill and decode, residual addition, and the SwiGLU activation. We also built the layer executor, cache-position management, speculative verification and rollback, graph capture, and the serving demo.

“cuBLAS handles QKV projection, attention’s output projection, the three MLP projections, and the final vocabulary projection. Those are established matrix-multiplication workloads. Reimplementing a competitive general GEMM would have consumed a large part of the project without addressing the rest of the inference path.

“That choice still leaves room for Tensor Cores inside our custom attention. There, the matrix products are part of a tiled algorithm with masking and online softmax; keeping them inside the attention kernel avoids materializing intermediate matrices.”

## 4. One transformer layer, kernel by kernel

[Offer this verbally: “I can give the blocking strategy for each operation, and we can go deeper on whichever is most interesting.” Embedding runs before the layer stack; final normalization and the vocabulary head run after it.]

### Embedding lookup

“Embedding is a gather, not a matrix multiplication. One block owns one token, and 128 threads copy its embedding row across the hidden dimension. Adjacent threads copy adjacent FP16 values, which gives coalesced accesses. There’s no reduction or need for shared memory. This kernel uses scalar copies; it isn’t one of the explicitly vectorized kernels.”

### RMSNorm

“One block owns one token’s hidden vector. Threads load eight FP16 elements at a time, accumulate squares in FP32, reduce within warps, then combine the warp totals through shared memory. We compute the inverse root mean square, apply the learned scale, and write the normalized row.

“The target uses 448 threads for its 3,584 channels. The draft uses 128 threads for 896 channels, rounding up to complete warps so the reduction’s full-warp mask is valid. Vectorization reduces memory-instruction overhead; it doesn’t guarantee bandwidth saturation, especially for a single row.”

### QKV projection, RoPE, and cache writes

“We concatenate Q, K, and V weights once at startup, so one cuBLAS projection produces all three. Bias addition and tensor-layout preparation are separate operations in the runtime.

“RoPE rotates pairs of query and key components according to token position. We rotate K while writing it into the cache, and copy V unchanged. The current cache-write kernel gives each block one batch/KV-head pair and up to 16 sequence rows, with 256 threads. That exposes more blocks for a long prompt while retaining a small launch for decode.

“Q is rotated inside attention when it is staged in shared memory. This removes separate global-memory passes for rotated Q and K.”

### Prefill attention

“Prefill has many query positions available at once. A block owns one query head and a tile of 64 query rows, with 128 threads, and streams through 32-row KV tiles. The target head dimension is 128; the draft’s is 64.

“We use 16-by-16-by-16 WMMA Tensor Core operations for the score and weighted-value matrix products. Between them, we apply scale, causal masking, and online softmax.

“Online softmax keeps a running maximum, denominator, and weighted output for each query. When a new tile changes the maximum, we rescale the previous state before adding the new contribution. That lets us process the keys in tiles without writing a full sequence-by-sequence attention matrix to global memory.”

### Decode and verification attention

“One-token decode and small speculative verification blocks have too few query rows to use that same 64-row tile efficiently. We specialize a separate scalar attention path for one through nine query rows. Nine covers eight proposals plus a deferred target token.

“Each block owns a query head and streams through 32-row KV tiles with 128 threads. It keeps the query and online-softmax state while reusing shared memory for the next KV tile. For verification, every query has its own causal boundary, even though all candidate keys have already been written.

“For longer contexts, we also divide the KV range between blocks. Each block produces a partial maximum, denominator, and weighted output. A small combine kernel uses one block per head and one thread per output channel to rescale and merge those partials. We can’t just average independently normalized outputs.

“Streaming tiles bounds working storage; splitting the KV range exposes more blocks. Those solve different problems. Splitting also adds scratch traffic and a combine launch, so short contexts use the unsplit path.”

### Output projection and residual addition

“cuBLAS projects the attention result back to the hidden width. The residual kernel then adds the branch output to the incoming hidden state. It uses flat chunks, 256 threads per block, and eight FP16 values per thread, with packed half-pair additions. Each result is independent, so there is no reduction. Residual addition and normalization are still separate kernels.”

### SwiGLU and the final vocabulary head

“After the next RMSNorm, cuBLAS computes the MLP gate and up projections. Our activation kernel calculates SiLU of the gate times the up value in one pass. Like residual addition, it uses 256-thread blocks and eight FP16 elements per thread, but the nonlinear calculation happens in FP32. It overwrites the gate buffer, then cuBLAS performs the down projection. A second residual add completes the layer.

“After all layers, we run a final RMSNorm and a cuBLAS vocabulary projection. Greedy selection is a PyTorch argmax, not a custom sampler. We don’t need a vocabulary softmax to find the largest logit; attention still has its own softmax.”

## 5. Optimization stories



### Story A: make repeated draft execution cheap

“The draft is small, and we run it repeatedly for every speculative round. That makes repeatedly dispatching the same sequence of operations a useful optimization target.

“We capture a one-token draft forward in a CUDA graph and replay it. The interesting part was making changing state compatible with a fixed graph. Input addresses stay fixed, but token ID, cache position, and valid context length change. Those values live in persistent device tensors, and the kernels read them during replay. If we captured an ordinary host integer for the cache position, later replays would keep using the old position.

“We also keep launch geometry stable by sizing the graph’s attention split layout from cache capacity, while the kernels read the actual length on the device. The captured output is reused, so anything that must survive another replay has to be consumed or copied first.

“The graph removes repeated dispatch work. It doesn’t remove the serial dependency between draft tokens, and the host still coordinates greedy decisions. In the demo, the draft uses the graph; the target remains eager.”

### Story B: change the unit of attention work

“For decode, we moved to streaming the cached keys and values in fixed-size tiles. A query stays resident, a KV tile comes through shared memory, and we update the online-softmax state before moving on. The working storage doesn’t grow with the entire context.

“Then there’s a separate occupancy question: batch one and one block per query head can leave much of the GPU idle. Splitting the KV range creates independent partial attention problems that we can merge exactly using their softmax statistics.

“The tradeoff is extra scratch and a merge kernel. The lesson was to choose a decomposition for the actual query count and context length, rather than assume the prefill strategy would also be good for decode.”

### Story C: fuse RoPE where the data is already moving

“RoPE is a small amount of arithmetic, but a separate implementation adds launches and reads and writes entire tensors. We already have to write K into the cache and load Q for attention. Putting the rotations at those two points removes intermediate traffic.

“The subtle part is position bookkeeping: the source tensor uses local row indices, but its rotation must correspond to absolute sequence positions. That gets particularly important when speculative decoding rolls back a rejected suffix or the draft graph replays at a new position. I’d explain both the saved memory passes and the invariant that keeps the fusion correct.”

### Story D: move from manual tuning to a measured agent loop

“We started by implementing and optimizing kernels by hand, which gave us an understanding of the layouts, correctness constraints, and likely bottlenecks. Then we built setups where coding agents could edit, compile, run correctness checks, benchmark, and use the result to guide the next attempt.

“That helped explore changes such as wider memory accesses and Tensor Core operations inside prefill attention. The useful part was closing the loop around executable feedback. A plausible-looking optimization wasn’t enough; it had to preserve the computation and improve measured performance.

“The current kernels reflect that division of work: vectorized accesses in normalization, residuals, activation, and attention staging; Tensor Cores in prefill attention; and a specialized scalar path for small query counts. I’d distinguish the experiment design and review decisions I owned from implementation changes proposed by an agent.”

[Do not assign a numerical historical speedup to graph capture, streaming, fusion, or the agent workflow unless you have the corresponding experiment. The next story has a saved paired comparison.]

### Optional recent example: a layout fix with a measurable result

“A recent pass made two focused changes. We padded the shared-memory K rows by two FP16 elements to prevent column accesses from repeatedly mapping to the same bank. We also tiled RoPE cache writes across sequence rows instead of assigning an entire prompt to one block per KV head.

“On the RTX 4090, at batch one and a 1,742-token context, the cold-cache target RoPE operation went from 168 to 16 microseconds. Nine-token attention went from 141 to 92 microseconds, and single-token attention from 56 to 51 microseconds.

“But full target-only decode improved by only about 0.6 to 0.9 percent. That’s an important distinction: a large local speedup can affect only a small part of the model’s total time. We checked 63 paired kernel cases for bitwise equality and passed 61 runtime and demo tests.”

[These are isolated operation timings, not total speculative-decoding speedups. The bank-conflict explanation comes from the access pattern; we did not collect hardware bank-conflict counters. See the [paired optimization report](../benchmarking/optimizations/attention_v1/REPORT.md).]

## 6. Limitations and deliberate tradeoffs



### A fixed, fully allocated KV cache

“We reserve contiguous K and V storage for the whole configured context up front. That gives us simple indexing, stable storage for graph replay, and cheap rollback: move the valid-length cursor back, then overwrite rejected slots later. We don’t need to erase the stale suffix as long as attention never reads beyond the valid prefix.

“The cost is reserving capacity a short request might never use, and imposing a fixed context limit. It becomes a bigger problem when serving many requests with different lengths. A natural extension is a block allocator and paged attention, where logical token positions map to physical cache blocks allocated as needed. That would require changes to cache management and attention addressing; it isn’t just replacing one allocation call.”

[The [PagedAttention paper](https://arxiv.org/abs/2309.06180) is the reference for this serving extension. Paging improves management and sharing of cache storage; it does not eliminate the K/V data needed for a given live context or guarantee faster batch-one attention.]

### Greedy decoding

“Greedy decoding gives us the clearest correctness demonstration: generate with the target alone, generate speculatively, and compare the exact output IDs. It also keeps the verification implementation small enough to inspect and test thoroughly.

“I wouldn’t claim greedy decoding necessarily gives the largest speculative speedup. Acceptance depends on the models and prompt. Supporting sampling while preserving the target distribution requires the proper acceptance and correction algorithm; matching argmax tokens is only the greedy version.”

### Target and draft vocabulary sizes

“Being in the same model family doesn’t mean every tensor dimension is identical. These checkpoints expose 152,064 target vocabulary rows and 151,936 draft rows. Our startup check verifies that tokens in the draft tokenizer use the same IDs in the target tokenizer, but the draft embedding table still has a smaller valid index range.

“If a prompt contains an ID outside that range, we reject it before inference. If the target selects such a token during generation, the draft cannot consume it to continue proposing, and the current demo reports an error. That is a model-compatibility limitation, not a normal speculative rejection.

“A robust next step would be to continue in target-only mode when the draft can’t represent the continuation, or use a pair with fully compatible supported IDs. I wouldn’t silently clamp the token or mask the target’s logits: that would change the target baseline we’re trying to preserve.”

[The vocabulary dimensions agree with the official [7B configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) and [0.5B configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json). Vocabulary matrix rows are not necessarily all distinct usable tokenizer entries. Do not claim the extra rows are “128 extra words,” or invent a reason the checkpoint authors chose the sizes. For a particular failing prompt, identify the actual error and ID before attributing it to this limitation.]

## 7. Transition to the live demo

“That’s the design. Now I’ll show the consequence for one request. Both models are already loaded and warmed up. The left side runs our target-only engine first; the right side then runs our speculative engine, starting from the same prompt with fresh caches. These runs are sequential on the same GPU.

“I’ll start with four draft tokens per round and look at three things: whether the output token IDs match, how many proposals the target accepts, and whether the saved target steps outweigh the draft work.”

[Use a prompt you have rehearsed on the current build. A possible starting point is “Explain why a KV cache helps language-model inference in three short paragraphs,” with an output budget of 128 and gamma 4. This is a suggested prompt, not a claim of measured acceptance or guaranteed speedup. Start a fresh chat for controlled comparisons.]

“The speculative side emits verified chunks, so it can appear burstier. The output tokens are confirmed by the target before they reach the user.

“This page’s speedup is end to end for the generation run. It includes prefill and the server’s generation and streaming work; loading and warmup are outside it. Its tokens-per-second number includes prefill, unlike a benchmark measuring only steady decode. The speculative first-output timer is time to the first verified chunk, which isn’t exactly the same event as the target’s first token.”

[At completion, point to exact-token agreement first, then acceptance and total elapsed time. If time allows, rerun the same fresh prompt with gamma 2 or 8.]

“Increasing the proposal length isn’t automatically better. We can amortize a target forward over more accepted tokens, but we pay for more serial draft work and potentially reject more of it. This gives us a direct way to see that tradeoff.”

[If speculation loses:]

“Here the saved target work didn’t pay for drafting, verification, and the extra prefill. I’d use the acceptance count and round count to explain that result, then check separate timings before assigning the cause to a kernel.”

[If IDs differ:]

“This is not a like-for-like correctness success. I would locate the first divergent ID and compare target logits and cache state there. Near ties in FP16 are one possible cause, but so are implementation bugs; the mismatch itself doesn’t tell us which.”

[If the vocabulary error appears:]

“This is the compatibility boundary I mentioned: the target chose an ID the smaller model can’t embed. The current demo stops that speculative request rather than silently changing the target’s choice.”

“The main thing I learned was to connect the algorithmic opportunity to the cost of the actual execution path: how many target steps we avoid, how we move the data, how much parallel work each launch exposes, and whether a local improvement matters to the user’s total latency.”

## Backup answers for questions



### Why can verification run in parallel if generation cannot?

“Ordinary generation doesn’t yet know the future input tokens. The draft supplies candidate IDs, so the target can process them as a known sequence with causal masking. Results after a disagreement are conditioned on the wrong prefix and must be discarded.”

### What happens to the cache after rejection?

“Keep the accepted prefix by rolling back the valid cursor. The target’s correction token is emitted but its KV write is deferred until the next verification, where it is prepended to the candidates. The draft consumes the correction immediately to get the next proposal logits. This saves a separate target forward just to commit that token.”

### How do you combine split attention correctly?

“For split s, save its maximum m_s, exponential sum l_s, and unnormalized weighted-value sum a_s. Set M to the maximum of all m_s. Rescale each contribution by exp(m_s − M), sum the rescaled a_s values, and divide by the sum of the rescaled l_s values. That reconstructs the softmax across the whole attended range.”

### What does a simple speculative break-even model look like?

“A round emits approximately accepted-count plus one tokens, before EOS and budget trimming. Its cost is draft proposal work, target verification, and bookkeeping, including the draft consuming the target correction. Compare that with the same number of ordinary target decode steps. High acceptance helps, but draft latency and verification scaling also matter.”

### Would larger batches fix the low roofline percentages?

“They can expose more blocks and improve weight reuse, especially for GEMMs. They won’t automatically fix an inefficient shared-memory layout, redundant work, or a poorly matched attention algorithm. They also answer a different latency/throughput question. Our recent padding and sequence-tiling gains were measured at batch one.”

### How should we interpret the attention roofline results?

“Our roofline is an optimistic time bound based on useful arithmetic and logical memory traffic, not a measured utilization counter. Small launches, serial dependencies, reductions, shared-memory behavior, repeated KV loads across query heads, and softmax all matter. For warm inputs, cache residency also changes the relevant memory bandwidth. I’d compare measured time with a strong implementation at the same shape and use profiling to distinguish causes.”

### Are the custom kernels faster than vLLM?

“The saved target-only comparison was competitive for the specific batch-one FP16 shapes we tested, but it wasn’t a general win. Before the latest attention changes, the eager custom engine was about 1% behind vLLM on the shorter prompts and about 5% behind at the longest tested prompt. The latest changes improved our own target-only decode by under 1%; we haven’t repeated the vLLM comparison against that build. None of that establishes production-serving parity.”

[The browser compares custom target-only with custom speculative generation, not with vLLM. The controlled engine benchmark excludes HTTP and text rendering. See [historical benchmark results](../benchmarking/results/REPORT.md).]

### What would you improve next?

“For this demo, I’d prioritize graceful target-only fallback for incompatible tokens and investigate any output mismatch. For latency, I’d profile the whole model and avoid projecting unused prefill rows through the vocabulary head. For scaling to many users, I’d add paged cache management and request scheduling. Attention split tuning still carries a heuristic for the original 72-SM GPU, so retuning it for the 4090 is another concrete experiment.”

### How portable is the implementation?

“It specializes to these model dimensions and has been validated on the current RTX 4090 setup. Some WMMA fragment indexing is architecture-sensitive. Successful compilation alone wouldn’t establish correctness or speed on a different GPU; I’d rerun numerical and performance validation.”

## Rehearsal and source map

Before the interview, rehearse one complete successful prompt and one less favorable case. Know the actual error message for any prompt you describe as a vocabulary failure. Keep measurements attached to their shape and timing definition. Describe wider loads as reducing instructions, not automatically saturating bandwidth, and describe an agent-produced change only to the extent you can explain and defend it.

Open code only when asked. These are useful entry points:


| Topic                                           | Local reference                                                                                                                                                                                                                                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full execution trace and source markers         | [Inference walkthrough](WALKTHROUGH.md)                                                                                                                                                                                                                                             |
| Model execution and cache rollback              | [Executor](../runtime/executor.py)                                                                                                                                                                                                                                                  |
| Fixed graph inputs and replay                   | [Graph executor](../runtime/executor_graph.py)                                                                                                                                                                                                                                      |
| Draft state and vocabulary-bound error          | [Draft runner](../runtime/speculative/draft_runner.py)                                                                                                                                                                                                                              |
| Acceptance and target logit alignment           | [Target verifier](../runtime/speculative/target_step.py)                                                                                                                                                                                                                            |
| Streaming and timing definitions                | [Demo server](server.py), [speculative loop](decoding.py)                                                                                                                                                                                                                           |
| Prefill, decode, split/combine, and RoPE        | [Target attention](../runtime/production_kernels/target/attention/kernel.cu)                                                                                                                                                                                                        |
| Embedding, normalization, residuals, activation | [Embedding](../runtime/production_kernels/target/embedding/kernel.cu), [RMSNorm](../runtime/production_kernels/target/rmsnorm/kernel.cu), [residuals](../runtime/production_kernels/target/residual_ops/kernel.cu), [SwiGLU](../runtime/production_kernels/target/swiglu/kernel.cu) |
| Recent measured optimizations and rollback      | [Paired results](../benchmarking/optimizations/attention_v1/REPORT.md)                                                                                                                                                                                                              |
| Roofline methodology and vLLM comparison        | [Benchmarking guide](../benchmarking/README.md), [historical results](../benchmarking/results/REPORT.md)                                                                                                                                                                            |
