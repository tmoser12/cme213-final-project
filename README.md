# Speculative Decoding — CME 213 Final Project
Tobias Moser · Eli Wandless

Custom CUDA kernels for **vanilla speculative decoding** with Qwen2.5-7B-Instruct (target)
+ Qwen2.5-0.5B-Instruct (draft), FP16, on Stanford ICME's Quadro RTX 6000 (SM 7.5, Turing).

The eventual headline kernel is a fused speculative verify-and-sample (Leviathan et al.
Algorithm 1). The current focus is building the per-op CUDA kernels for the Qwen2 forward
pass and integrating them into a native C++/CUDA host-side runtime. **`tmp.md` is the
authoritative plan** (Qwen2 Native Forward-Pass Execution Plan); read it before changing the
architecture, model pair, or runtime layout.

## Kernel status

Each kernel lives under `src/kernels/<name>/` and is developed/validated in isolation by
monkey-patching it into the HuggingFace model (see `skills/kernel_monkey_patching_plan.md`).

| Kernel | Status | Notes |
|---|---|---|
| `embedding` | ✅ implemented | pure indexed-gather, bit-exact vs `F.embedding` |
| `rmsnorm`   | ✅ implemented | float4/half2 vectorized, block reduction |
| `attention` | ✅ implemented | qkv_proj · rope · kv_write · fused_attn · o_proj; **prefill-optimized** |
| `swiglu`    | 🚧 scaffold | wiring + compilable stub launcher; `swiglu_act_forward` body TODO |

**Roadmap (not yet built):**
- **SwiGLU** kernel body — the `silu(gate) * up` elementwise fusion (scaffold is in place).
- **Residual-add** kernel — the two `+ residual` writes per decoder layer.
- **Decode-optimized attention** — a variant for token-by-token decode (S=1, growing KV cache);
  the current kernel targets prefill.
- **0.5B draft-model kernel suite** — current launch configs bake in 7B dims; the 0.5B draft has
  different dims and needs separately-tuned kernels.
- **Native C++/CUDA runtime** — host-side weight manager, memory/KV-cache planner, and decoder
  executor that stitch the kernels into an end-to-end forward pass (milestones M1–M5 in `tmp.md`).
- **Speculative decoding + MPI multi-GPU** — the end goal layered on top of the runtime.

## Environment setup

Every fresh shell on the cluster (see `skills/system.md` for the full build-issue field notes):

```bash
module load gnu12/12.3.0          # cuda/12.2 + course nvhpc module are loaded by default
conda activate cme213             # python 3.11, torch 2.4.0, transformers 4.45.0
```

Sanity-check toolchain resolution:
```bash
which nvcc        # expect /opt/ohpc/.../cuda/12.2/bin/nvcc  (NOT under nvidia-hpc-sdk)
which g++         # expect /opt/ohpc/.../gnu12/.../bin/g++   (12.3.0)
```

One-time model download (~15 GB, ~10–20 min) and a load sanity check:
```bash
bash scripts/download_models.sh
srun --partition=gpu-turing --gres=gpu:1 --pty \
     bash -c "source activate cme213 && python scripts/verify_env.py"
```
Both models must report `Vocab size: 152064` — confirms the shared tokenizer that vanilla
speculative decoding requires.

## Developing / benchmarking a kernel

Always run CUDA on a GPU node — the login node sees no devices. Each kernel dir has a
`run_benchmark.sh` that clears the JIT cache and `srun`s the benchmark (which checks correctness
against the HF reference, then times custom vs eager vs `torch.compile`):

```bash
bash slurm/interactive.sh                            # interactive 1-GPU shell (optional)

bash src/kernels/rmsnorm/run_benchmark.sh            # correctness + micro-benchmark
bash src/kernels/attention/run_benchmark.sh
bash src/kernels/swiglu/run_benchmark.sh             # prints "scaffold-only" until kernel.cu is filled in
bash src/kernels/rmsnorm/run_benchmark.sh --profile  # + nsys timeline + ncu metrics
```

To add a new kernel, copy `src/kernels/rmsnorm/` (single custom op) or `src/kernels/attention/`
(cuBLAS GEMMs + multiple custom sub-ops). `skills/kernel_development_workflow.md` is the
step-by-step SOP.

## Repo structure

```
cme213-final-project/
├── CLAUDE.md                          # local agent guidance (gitignored)
├── README.md
├── tmp.md                             # authoritative plan: native forward-pass runtime
├── setup.sh                           # env setup, sourced by run_benchmark.sh (gitignored)
├── requirements.txt
│
├── src/
│   ├── kernels/                       # one dir per custom kernel
│   │   ├── embedding/                 # ✅ kernel.cu bindings.cpp jit.py wrapper.py benchmark.py
│   │   ├── rmsnorm/                   # ✅ + run_benchmark.sh + kernel_walkthrough.md
│   │   ├── attention/                 # ✅ + benchmark_scripts/ (per-sub-op) + attention.md
│   │   └── swiglu/                    # 🚧 scaffold
│   └── models/
│       ├── modeling_qwen2.py          # verbatim HF reference (NOT imported; kernel patch-target map)
│       └── loading.py                 # model/weight loading helpers
│
├── scripts/
│   ├── download_models.sh             # download Qwen2.5 weights
│   └── verify_env.py                  # sanity check CUDA + both models load
│
├── skills/
│   ├── system.md                      # cluster build-issue field notes (read first)
│   ├── gpu_spec.md                    # RTX 6000 / SM 7.5 hardware notes
│   ├── kernel_development_workflow.md # step-by-step SOP for a new kernel
│   └── kernel_monkey_patching_plan.md # how kernels splice into the HF model
│
├── slurm/
│   └── interactive.sh                 # srun --pty bash wrapper (arg = #GPUs)
│
├── benchmarks/prompts/                # prompt sets for end-to-end benchmarking
├── models/                            # downloaded weights (gitignored)
├── results/                           # profiles + reports (gitignored)
└── logs/                              # SLURM stdout/stderr (gitignored)
```
