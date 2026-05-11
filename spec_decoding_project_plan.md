# Vanilla Speculative Decoding on Stanford ICME Quadro RTX 6000: CME 213 Final Project Plan

## TL;DR

- **Model pair: Qwen2.5-7B-Instruct (target, GPU 1) + Qwen2.5-0.5B-Instruct (draft, GPU 0), both FP16.**
  This pair shares a 151,936-token BPE tokenizer (required for vanilla SpecDec correctness), has a ~14x
  parameter ratio giving c ~= 0.07 and empirical alpha ~= 0.55-0.75, and both models fit individually on a
  single 24 GB Quadro RTX 6000 with ample KV-cache headroom. No quantization, no tensor or pipeline
  parallelism.

- **Multi-GPU layout: one MPI rank per GPU, whole models only.** Rank 0 runs the entire draft model;
  rank 1 runs the entire target model. Inter-rank communication is point-to-point MPI carrying only token
  IDs and per-token probability vectors -- O(gamma x V x 2 bytes) per round, which at V=152K and gamma=5 is
  ~1.5 MB, costing ~250 us over PCIe SYS. This is well under 2% of total round latency and avoids the
  fatal flaw of tensor parallelism on a SYS-topology cluster: TP=2 would require 56 AllReduces of
  hidden-state activations per verify call, adding 5-14 ms of PCIe latency and likely making TP=2
  *slower* than single-GPU. Pipeline parallelism (PP) is equally inappropriate for batch-1 inference:
  with PP=3 at most one GPU is active at any time, achieving no latency reduction while tripling
  resource consumption. The SYS topology (confirmed via nvidia-smi topo -m: all GPU pairs traverse
  PCIe + NUMA interconnect, no NVLink bridges) makes both TP and PP non-starters for this project.

- **The one CUDA kernel to write from scratch is the fused speculative-sampling verify-and-accept
  kernel.** It reads target and draft probability distributions for gamma draft positions, runs the
  Leviathan et al. Algorithm 1 acceptance test in a single launch, samples a replacement token on
  rejection from the renormalized residual distribution, and emits a bonus token on full acceptance.
  This is the only speculative-decoding-specific kernel beyond a standard transformer forward pass.
  It exercises every rubric item (memory hierarchy, thread organization, coalescing), and the fused
  vs. naive multi-launch comparison is the "non-trivial algorithmic optimization with quantified
  improvement" the course requires.

- **Headline result: end-to-end tokens/sec speedup of speculative vs. autoregressive decoding, paired
  with a closed-form analytical model.** With Quadro RTX 6000 FP16-tensor peak ~= 130.5 TFLOPS and
  memory bandwidth = 672 GB/s, the ridge point is ~= 194 FLOPs/byte. Batch-1 decode sits at ~1
  FLOPs/byte (memory-bound by ~200x). Verifying gamma=5 tokens lifts arithmetic intensity ~5x, still
  deeply memory-bound, which is exactly why speculative decoding helps: in the memory-bound regime,
  verifying gamma tokens costs only marginally more than verifying 1. Plugging alpha ~= 0.7, c ~= 0.07,
  gamma = 5 into Leviathan's formula S = (1 - alpha^(gamma+1)) / ((1 - alpha)(gamma*c + 1)) gives
  S ~= 2.65x. The report can overlay this analytical curve against measured tokens/sec across
  gamma in {1, ..., 8}.

---

## Key Findings

### 1. What Kernels Does Vanilla SpecDec Actually Need?

Vanilla speculative decoding requires exactly one kernel beyond a standard transformer forward pass.

**Draft generation** is a standard autoregressive loop: run the draft model for gamma steps (batch=1,
one token at a time), caching KV at each step. No new kernels needed beyond a normal transformer forward.

**Target verification** is a single transformer forward pass over gamma+1 positions (the last committed
token plus gamma draft tokens), with a standard causal mask. Algorithmically identical to a prefill of
gamma+1 tokens. No new kernels needed.

**The accept/reject step** is the only speculative-decoding-specific computation:

1. For each draft position i = 0, ..., gamma-1:
   - Compute ratio r = p_target[i, draft_tokens[i]] / p_draft[i, draft_tokens[i]]
   - Sample u ~ Uniform(0, 1)
   - If u < min(1, r): accept draft_tokens[i], continue
   - Else: sample replacement from norm(max(0, p_target[i] - p_draft[i])), stop
2. If all gamma tokens accepted: sample bonus token from p_target[gamma]

**KV-cache rollback** is not a kernel. It is a single host-side integer assignment retracting the
per-layer write pointer. On rejection at position i, the next call simply overwrites cache slots
i+1 onward. No memcpy, no device-side operation needed.

vLLM's production reference for this step is a Triton kernel in
`vllm/v1/sample/rejection_sampler.py`. Study the structure; do not copy the code.

### 2. Model Pair Recommendation

**Use Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct, FP16.**

| Property | Qwen2.5-7B (target) | Qwen2.5-0.5B (draft) |
|---|---|---|
| Parameters | 7.6 B | 0.49 B |
| FP16 weight size | ~14 GB | ~1 GB |
| Tokenizer | Shared 151,936-token BPE | Identical |
| Architecture | 28 layers, GQA (28Q/4KV), hidden 3584, intermediate 18944 | 24 layers, GQA (14Q/2KV), hidden 896 |
| Head dimension | 128 | 128 |
| Empirical alpha (MT-Bench, T=0.7) | -- | 0.55-0.75 |
| Cost ratio c (memory-bound) | -- | ~= 0.5/7 ~= 0.07 |

**Memory budget per GPU (FP16, 4K context, one model per GPU):**

- GPU 0 (draft): weights ~1 GB + KV cache ~96 MB + activations ~0.5 GB ~= **~2 GB used, ~22 GB free**
- GPU 1 (target): weights ~14 GB + KV cache ~231 MB + activations ~1 GB ~= **~16 GB used, ~8 GB free**

Both models fit easily with no quantization or sharding needed.

**Backup: Llama-3.1-8B + Llama-3.2-1B.** Same tokenizer (128K vocab), NVIDIA's TensorRT-LLM uses this
pair as its canonical demo. The ~8:1 parameter ratio gives c ~= 0.13 (slightly worse). Use if Qwen2.5
alpha < 0.45 or if you hit a tokenizer mismatch bug.

**Do not use Pythia or TinyLlama:** older architectures, weaker empirical alpha, no GQA, poor
tokenizer alignment.

### 3. Multi-GPU Architecture (Draft-Target Process Separation)

**One MPI rank per GPU. Rank 0 runs the complete draft model; rank 1 runs the complete target model.**

```
GPU 0 (Rank 0 -- Draft)           GPU 1 (Rank 1 -- Target)
+------------------------+         +------------------------+
|  Qwen2.5-0.5B, FP16   |         |  Qwen2.5-7B, FP16      |
|  Full model, ~1 GB     |         |  Full model, ~14 GB     |
|  Own KV cache          |         |  Own KV cache           |
|  672 GB/s HBM          |         |  672 GB/s HBM           |
+-----------|------------+         +-----------|------------+
            |  MPI_Send(draft_tokens)  ------> |
            |  MPI_Send(draft_probs)   ------> |
            | <------  MPI_Send(n_accepted)    |
            | <------  MPI_Send(bonus_token)   |
```

**Communication volume per round (gamma=5, V=152K, FP16):**

| Message | Direction | Size |
|---|---|---|
| Draft token IDs | Rank 0 -> Rank 1 | 5 x 4 bytes = 20 bytes |
| Draft probabilities | Rank 0 -> Rank 1 | 5 x 152K x 2 bytes ~= 1.5 MB |
| n_accepted + bonus_token | Rank 1 -> Rank 0 | 8 bytes |

At ~6 GB/s effective PCIe SYS bandwidth: 1.5 MB / 6 GB/s ~= 250 us per round. A target verify
call takes ~20-25 ms. Communication is ~1% of round latency.

Note: for initial correctness testing, implement greedy acceptance first (accept iff
argmax(p_target) == draft_token). This eliminates the need to send draft_probs entirely
(only 20-byte token IDs needed) and makes losslessness trivially verifiable.

**MPI bootstrap (no NCCL needed):**

```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
cudaSetDevice(rank);  // rank 0 -> GPU 0, rank 1 -> GPU 1
// All communication is MPI point-to-point token IDs and probs.
// No hidden-state transfers, no NCCL, no AllReduce.
```

**SLURM invocation:**

```bash
srun --partition=gpu-turing --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 \
     python spec_decode_main.py --gamma 5
```

**Why not tensor parallelism?** TP=2 requires 56 AllReduces of hidden-state activations per verify
call (2 per layer x 28 layers). Each ~43 KB AllReduce over PCIe SYS costs ~100-250 us in NCCL
latency. Total: 5.6-14 ms of communication overhead per verify call, against ~20-25 ms of compute.
TP=2 over PCIe SYS is likely slower than single-GPU for batch-1 inference.

**Why not pipeline parallelism?** With PP=3, exactly one GPU is active at any moment. Total verify
latency ~= single-GPU latency. PP provides zero benefit for batch-1 inference when the model fits
on one GPU, and uses 3x the resources to achieve it.

### 4. The CUDA Kernel to Write From Scratch

**Name:** `fused_speculative_verify_and_sample`

**What it does:** Implements Leviathan et al. Algorithm 1 in a single GPU kernel launch. The naive
alternative iterates through draft positions in a Python loop -- one kernel launch per position for
the ratio check, one for conditional sampling, one for renormalization. The fused version does all
of this in a single launch.

**Inputs (device pointers):**
- `target_probs[gamma, V]` -- post-softmax target probabilities for positions 0...gamma-1
- `target_bonus_probs[V]` -- target probs at position gamma (for the bonus token)
- `draft_probs[gamma, V]` -- post-softmax draft probabilities
- `draft_tokens[gamma]` -- token IDs generated by the draft model
- `uniform_rand[gamma+1]` -- pre-sampled U[0,1] values
- `V` -- vocabulary size (152K for Qwen2.5)

**Outputs:**
- `output_tokens[gamma+1]` -- accepted + replacement/bonus tokens
- `n_accepted` -- scalar in {0, ..., gamma+1}

**Algorithm (one CUDA block per draft position, V threads per block):**
1. Thread `draft_tokens[i]` loads p_t = target_probs[i, draft_tokens[i]] and
   p_d = draft_probs[i, draft_tokens[i]].
2. Thread 0 checks uniform_rand[i] < min(1, p_t / p_d) (coordinating via shared memory).
   On acceptance: write draft_tokens[i] to output, increment n_accepted, continue.
3. On rejection: all V threads cooperatively compute adjusted[v] = max(0, p_target[v] - p_draft[v]),
   run a block-level parallel prefix sum (Blelloch scan in shared memory) to build a CDF, then use
   inverse-CDF sampling with uniform_rand[i]. Write replacement token, stop.
4. If all gamma accepted: run the same inverse-CDF path on target_bonus_probs with uniform_rand[gamma].

**CUDA engineering points to highlight in the report:**
- **Coalesced loads:** V=152K, each probability row is 304 KB in FP32. Load via __half2 vectorized
  loads (2 elements per instruction) for 128-byte transaction alignment.
- **Block-level reduction:** Normalization sum uses shared-memory warp-level reduce (__shfl_xor_sync)
  then warp-leader accumulation.
- **Parallel prefix sum (Blelloch scan):** CDF construction for inverse-CDF sampling is an in-place
  parallel scan in shared memory. This is the non-trivial component and the main source of the
  fused kernel's speedup.
- **Single launch:** Eliminates per-position kernel-launch overhead. Snowflake's Arctic Inference
  reported a comparable fusion reducing rejection-sampling latency from 1.34 ms to 0.38 ms (3.5x).
- **Numerical stability:** Compute p_t/p_d in FP32 with denominator clipped to max(p_d, 1e-6).

**Rubric coverage:**
- Hand-written CUDA kernel: yes
- Memory hierarchy (HBM streaming + SMEM reduction + registers): yes
- Thread organization (block-per-position, V cooperative threads): yes
- Coalesced access (vocab dimension is contiguous): yes
- Non-trivial optimization with quantified improvement (fused vs. multi-launch latency): yes

### 5. Roofline and Analytical Speedup Model

**Quadro RTX 6000 (Turing TU102, SM 7.5) specs:**

| Metric | Value |
|---|---|
| FP16 tensor-core peak | 130.5 TFLOPS |
| FP16 non-tensor peak | 32.6 TFLOPS |
| HBM bandwidth | 672 GB/s (24 GB GDDR6) |
| FP16-tensor ridge point | 130.5e12 / 672e9 ~= 194 FLOPs/byte |
| No native BF16 | (SM 7.5 constraint) |

**Arithmetic intensity of decode operations:**

For a dense weight matrix W of size (H_out x H_in) and T input tokens in FP16:
- FLOPs: 2 x T x H_in x H_out
- Bytes (weight-dominated at T=1): H_out x H_in x 2
- Arithmetic intensity: T FLOPs/byte

So: one autoregressive decode token -> AI ~= 1 FLOPs/byte (memory-bound by ~194x).
Verifying gamma tokens in one forward pass -> AI ~= gamma FLOPs/byte.
At gamma=5: AI ~= 5 FLOPs/byte, still memory-bound by ~40x, but achieving ~5x the useful work
per byte of memory bandwidth. This is the mechanism of speculative decoding's speedup.

The compute-bound transition would require gamma ~= 194, far beyond what acceptance rates sustain.
Speculative decoding operates entirely in the memory-bound regime on this hardware.

**Leviathan et al. speedup formula:**

  S(gamma, alpha, c) = (1 - alpha^(gamma+1)) / ((1 - alpha) x (gamma*c + 1))

Where alpha = per-token acceptance probability, gamma = draft length per round, c = cost ratio of
one draft step to one target step. In the memory-bound regime, c ~= params_draft / params_target
~= 0.5 / 7 ~= 0.07 (each step costs proportionally to the number of parameters loaded from HBM).

| alpha | gamma=2 | gamma=4 | gamma=5 | gamma=6 | gamma=8 |
|---|---|---|---|---|---|
| 0.55 | 1.35x | 1.74x | 1.84x | 1.90x | 1.96x |
| 0.65 | 1.49x | 2.09x | 2.26x | 2.36x | 2.48x |
| 0.70 | 1.57x | 2.30x | 2.52x | 2.65x | 2.82x |
| 0.75 | 1.65x | 2.54x | 2.82x | 3.00x | 3.24x |

**Async pipeline gain (second analytical contribution):**

With non-blocking MPI (MPI_Isend/MPI_Irecv), rank 0 begins drafting round N+1 while rank 1
is still running the verify pass for round N. Theoretical pipeline efficiency:

  T_sequential = T_draft x gamma + T_verify
  T_pipelined  = max(T_draft x gamma, T_verify)
  Pipeline gain = T_sequential / T_pipelined

With T_verify / T_draft ~= 14 (parameter ratio, memory-bound), T_draft x gamma at gamma=5 is
5 units, T_verify is ~14 units. Pipeline critical path = max(5, 14) = 14 units vs. sequential
19 units: theoretical pipeline gain = 19/14 ~= 1.36x on top of the SpecDec speedup.

This is analytically derivable and empirically verifiable from Nsight Systems timelines showing
GPU 0 and GPU 1 activity side by side. Reference: Spector & Re, "Staged Speculative Decoding"
(arXiv 2308.04623).

### 6. Pitfalls and Correctness Requirements

1. **Acceptance rule:** `u < min(1, p_t/p_d)`, not `p_t > p_d`. The latter is not lossless.
2. **Rejection sample:** from `norm(max(0, p_target - p_draft))`, not `p_target`. Sampling from
   p_target on rejection also breaks distributional equivalence.
3. **Start with greedy:** for temperature=0, acceptance reduces to `argmax(p_target) == draft_token`.
   Outputs must match autoregressive greedy token-for-token. Validate before implementing stochastic.
4. **Matching distributions:** p_draft must be computed at the same temperature and top-k/top-p as
   p_target. Any mismatch breaks the distributional guarantee.
5. **Shared tokenizer:** Qwen2.5 draft and target use identical tokenizers. Verify this explicitly
   (check `tokenizer.vocab_size` matches) before running any experiments.
6. **KV-cache rollback is one integer:** decrement the sequence-length counter by
   `gamma - n_accepted` positions. The next step overwrites the stale slots.
7. **Numerics:** compute p_t/p_d in FP32; denominator = max(p_d, 1e-6) to avoid inf/nan.
8. **Nomenclature:** "Speculative decoding" (Leviathan ICML 2023) and "speculative sampling"
   (Chen et al. DeepMind 2023, arXiv 2302.01318) are the same algorithm discovered concurrently.
   Both are lossless. Cite both. Do not confuse with "self-speculative decoding" (LayerSkip),
   which is a different algorithm.

### 7. Recommended Stack

| Component | Choice | Rationale |
|---|---|---|
| Target model | Qwen2.5-7B-Instruct, FP16 | Shared tokenizer, single GPU, good alpha |
| Draft model | Qwen2.5-0.5B-Instruct, FP16 | 14x ratio, c~=0.07, well-studied pair |
| Precision | FP16 everywhere | Native on Turing SM 7.5; no BF16; avoids quantization kernel issues |
| GEMMs | cuBLAS cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_16F | Tensor-core path on Turing |
| Attention | torch.nn.functional.scaled_dot_product_attention | Memory-Efficient Attention on SM 7.5 |
| Multi-GPU launch | srun --gres=gpu:2 --ntasks=2 | One MPI rank per whole model |
| Communication | MPI_Send/MPI_Recv (blocking) then MPI_Isend/MPI_Irecv (async pipeline) | Token IDs + probs only; no NCCL needed |
| Accept/reject kernel | Hand-written CUDA | The headline original kernel |
| Profiling | Nsight Systems (timeline) + Nsight Compute (roofline) | Per-GPU tracing + kernel metrics |

**FlashAttention on Turing:** Official FA-2 dropped SM 7.5 support. torch SDPA (xformers
Memory-Efficient Attention backend on Turing) works out of the box. Community fork
`ssiu/flash-attention-turing` achieves ~63% compute throughput and is a stretch-goal optimization.

**Quantization:** AWQ and Marlin require SM >= 8.0. Stay in FP16.

### 8. 4-Week Schedule

**Week 1 -- Single-process baseline + correctness (May 11-17)**

- Day 1: cluster setup. Confirm SYS topology (done). Install PyTorch 2.x + CUDA 12.x.
  Load Qwen2.5-7B and 0.5B in FP16 using HuggingFace Transformers. Confirm both generate text.
- Days 2-3: single-GPU autoregressive baseline with KV cache. Measure tokens/sec on 100 MT-Bench
  prompts at 200-token output length. This is the control number for the entire project.
- Days 4-5: implement vanilla SpecDec in PyTorch, single process, both models on GPU 0 (sequential).
  Use greedy acceptance first. Validate outputs are token-for-token identical to autoregressive
  greedy on all test prompts.
- Days 6-7: measure empirical alpha, c, and tokens/sec for gamma in {1, 2, 4, 6, 8} on single GPU.
  Build the Leviathan formula prediction table and compare against measured S(gamma).
  **Deliverable: baseline tokens/sec, alpha table, formula vs. measurement comparison.**

**Week 2 -- Two-process MPI layout + async pipeline (May 18-24)**

- Days 1-2: implement MPI two-process layout. Rank 0 runs draft; rank 1 runs target. Blocking
  MPI_Send/MPI_Recv. Greedy mode first. Match single-GPU SpecDec outputs exactly (same seed).
- Day 3: upgrade to stochastic acceptance. Validate via chi-squared test (1000 prompts,
  output distribution must match autoregressive sampling within statistical tolerance).
- Days 4-5: implement async pipeline with MPI_Isend/MPI_Irecv. Rank 0 begins drafting round N+1
  immediately after sending round N's tokens, without blocking on rank 1's response. Measure
  pipeline efficiency: T_sequential / T_pipelined should approach ~1.36x for large gamma.
  This satisfies "overlapping communication with computation" from the course rubric.
- Days 6-7: profile with Nsight Systems. Produce GPU 0 / GPU 1 side-by-side timeline showing
  draft and verify overlap.
  **Deliverable: working two-GPU async SpecDec, Nsight timeline showing pipeline overlap.**

**Week 3 -- Custom CUDA kernel + roofline (May 25-31)**

- Days 1-3: write fused_speculative_verify_and_sample.cu as a PyTorch C++ extension. Build
  two versions: (a) multi-launch baseline (~5-6 launches per round); (b) fused single-launch
  with in-block Blelloch CDF scan.
- Day 4: correctness validation. Greedy mode: outputs must be identical to PyTorch reference.
  Stochastic mode: chi-squared test passes (1000 prompts, 128-token outputs).
- Day 5: roofline analysis on paper. Compute arithmetic intensity for draft step, target verify
  step as function of gamma, and acceptance kernel. Plot on RTX 6000 FP16 roofline (ridge point
  194 FLOPs/byte). Run Nsight Compute with --set full on the custom kernel and overlay measured
  performance on the roofline.
- Days 6-7: gamma sweep. Measure tokens/sec for gamma in {1, 2, 4, 6, 8} on two prompt domains
  (MT-Bench dialog, HumanEval code). Measure fused vs. multi-launch kernel latency directly.
  **Deliverable: roofline figure, kernel speedup table, gamma-sweep results.**

**Week 4 -- Ablations, polish, report (June 1-7)**

- Days 1-2: build the ablation table:
  1. Single-GPU, both models, sequential (week 1 baseline)
  2. Two-GPU, blocking MPI (week 2 baseline)
  3. Two-GPU, async pipeline (week 2 optimization)
  4. Two-GPU, async + fused kernel (week 3 final)
  Each row: tokens/sec, cumulative speedup, source of gain.
- Day 3: losslessness sweep: 1000 prompts x 128 output tokens, greedy SpecDec vs. autoregressive.
  Assert zero mismatches.
- Days 4-5: write the 6-page report. Suggested structure:
  1. Algorithm + correctness proof sketch (Leviathan et al.)
  2. System architecture (draft-target separation, MPI layout, async pipeline)
  3. Custom CUDA kernel design and optimization
  4. Roofline analysis + analytical speedup model (Leviathan formula + pipeline gain derivation)
  5. Empirical results: alpha, c, S(gamma), kernel ablation, domain comparison, pipeline efficiency
  6. Comparison to vLLM / HuggingFace / TensorRT-LLM
- Days 6-7: buffer, poster prep if required.

### 9. Comparison to Existing Implementations

- **HuggingFace assisted_generation** (transformers/generation/candidate_generator.py): pure Python,
  single-GPU, easy to read. Use as the correctness oracle for greedy mode. Cite Joao Gante's
  "Assisted Generation" blog post (HuggingFace, 2023).

- **vLLM v1 rejection sampler** (vllm/v1/sample/rejection_sampler.py): Triton-based fused kernel,
  structurally similar to your CUDA kernel. Study for algorithmic reference; do not copy.
  Reference: docs.vllm.ai/en/latest/features/speculative_decoding/

- **TensorRT-LLM**: closed-source kernels. NVIDIA's published demo uses Llama-3.3-70B + Llama-3.2-1B
  on H200, achieving ~3x throughput. Cite as the production-scale SOTA.
  Reference: "Boost Llama 3.3 70B Inference Throughput 3x with NVIDIA TensorRT-LLM Speculative
  Decoding" (NVIDIA Developer Blog).

- **Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"** ICML 2023,
  arXiv 2211.17192. Canonical citation for Algorithm 1 and the speedup formula. Reported 2.3-3.4x
  on T5-XXL with alpha in [0.53, 0.75].

- **Chen et al. (DeepMind), "Accelerating Large Language Model Decoding with Speculative Sampling"**
  arXiv 2302.01318. Concurrent independent discovery. Same algorithm, 2-2.5x on Chinchilla 70B.
  Cite alongside Leviathan.

- **SpecInfer** (Miao et al., arXiv 2305.09781) and **Medusa** (Cai et al., arXiv 2401.10774):
  tree-based and multi-head extensions beyond vanilla SpecDec. Mention as future work.

- **Spector & Re, "Staged Speculative Decoding"** (arXiv 2308.04623): analyzes the pipelined
  draft-target execution model. Cite for the pipeline efficiency derivation.

### 10. Evaluation Methodology

- **Primary metric:** output tokens/sec (decoded text), measured over 100 prompts from MT-Bench
  and HumanEval. Report mean +/- std across prompts.
- **Secondary metrics:** time-to-first-token (TTFT, prefill latency -- SpecDec does not affect
  this; report separately), acceptance rate alpha per domain and per gamma, measured c
  (draft latency / target verify latency from microbenchmark).
- **Correctness (greedy):** 1000 prompts x 128 output tokens. Every SpecDec output token must
  equal the autoregressive output token. Zero failures expected.
- **Correctness (stochastic):** 1000 prompts. Output-token frequency histograms at each position.
  Chi-squared goodness-of-fit against autoregressive baseline, p > 0.01 per position.
- **Spec-Bench** (Xia et al. 2024, sites.google.com/view/spec-bench): use MT-Bench or
  summarization subsets for a methodologically citable benchmark reference.

### 11. Profiling Recipes

```bash
# Nsight Systems: per-GPU timeline with MPI and CUDA, one output file per rank
srun --partition=gpu-turing --gres=gpu:2 --ntasks=2 \
  nsys profile \
    -t cuda,nvtx,mpi \
    -o spec_decode_rank%q{PMI_RANK} \
    --capture-range=cudaProfilerApi \
    python spec_decode_main.py --gamma 5

# Nsight Compute: kernel roofline metrics for the custom kernel
srun --partition=gpu-turing --gres=gpu:1 --ntasks=1 \
  ncu \
    --set full \
    --kernel-name fused_speculative_verify_and_sample \
    --launch-skip 10 --launch-count 1 \
    -o kernel_profile \
    python spec_decode_main.py --gamma 5 --profile-kernel
```

Add NVTX ranges around each phase (draft generation, MPI_Send, target verify, accept/reject)
so the Nsight Systems timeline clearly shows each component per rank. The draft/target overlap
in the async pipeline should be visible as GPU 0 compute activity overlapping with GPU 1 compute.

### 12. Caveats

1. **Alpha is domain-dependent.** Coding (HumanEval) gives higher alpha (~0.7) than open-ended
   chat (~0.55) than math reasoning (~0.5). Report per-domain.

2. **The 1.36x pipeline gain is a theoretical ceiling.** Real overlap depends on CUDA stream
   scheduling and MPI progress. Measure it; if smaller than theoretical, discuss why.

3. **Fair baseline:** compare SpecDec-with-KV-cache against autoregressive-with-KV-cache, both
   on the same two-GPU layout. Do not compare against a naive single-GPU Python loop.

4. **FlashAttention-2 does not support SM 7.5 officially.** Use torch SDPA as baseline.
   ssiu/flash-attention-turing is a stretch goal.

5. **FP16 non-determinism:** fix random seeds and use greedy decoding for the losslessness test
   to get reproducible outputs across runs.

6. **Sending draft_probs (1.5 MB) is the largest message.** At ~250 us for gamma=5 it is
   negligible. If you implement greedy-only acceptance, you can skip this message entirely
   (only 20-byte token IDs needed), which simplifies the initial implementation.

7. **Quantization on Turing:** AWQ and Marlin require SM >= 8.0. Stay in FP16.
