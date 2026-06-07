# Qwen2 Inference Runtime

Config-driven inference engine for Qwen2.5 (**7B target** + **0.5B draft**) on Turing GPUs.
Python owns orchestration; ahead-of-time-compiled CUDA kernels own the math. Built for a research
project (CUDA graphs + speculative decoding), not production.

## Layout

```
runtime/
├── core/                       # config, shapes, memory, weights
│   ├── configs/                # qwen2.5-7b.yaml (kernel_set=target), qwen2.5-0.5b.yaml (draft)
│   ├── config.py shapes.py memory.py weights.py
├── buffers.py                  # pre-allocated device buffers (KV cache, rope tables, static graph scratch)
├── executor.py                 # Qwen2Executor — eager prefill / decode / verify_gamma / greedy_extend
├── executor_graph.py           # GraphExecutorMixin — CUDA-graph capture/replay (decode + verify)
├── production_kernels/
│   ├── target/<op>/            # 7B kernels (head_dim=128): rmsnorm, embedding, attention, swiglu, residual_ops
│   └── draft/<op>/             # 0.5B kernels (head_dim=64), same layout
├── speculative/                # draft_runner, target_step, sampler, spec_decode, mpi_coordinator, mpi_protocol
├── benchmarks/                 # baseline + graph + spec-decode benchmarks
├── tests/                      # unit + GPU parity tests
└── plan.md                     # phase roadmap / status
```

Each YAML is the **single source of truth** for a model (dims, dtype, paths, layer order,
`kernel_set`). Code derives `head_dim`, weight shapes, and buffer sizes from the config.

## Quick start

```python
from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)   # or CONFIG_05B (draft)
weights, _ = load_weights_on_gpu(cfg, batch=1, device="cuda")
buffers = allocate_buffers(cfg, batch=1, max_seq_len=512, device="cuda")
ex = Qwen2Executor(cfg, weights, buffers, use_cuda_graph=True)        # kernel_set from cfg

logits = ex.prefill(input_ids)                    # [B, S, vocab]
seq = ex.greedy_extend(input_ids, n_new_tokens=64, use_cuda_graph=True)
```

## Build kernels

```bash
bash scripts/build_kernels.sh                 # all roles, all ops
bash scripts/build_kernels.sh draft attention # one role/op  (attention OOMs on the login node — use srun)
```
`.so` files land beside each `ops.py`; inference imports them (no runtime JIT).

## CUDA graphs

The per-token forward (~150 kernel launches) is capturable behind `use_cuda_graph`:
`decode_step_graph` (S=1) and `verify_gamma_graph` (S=γ+1). Positions live in device scalars and
RoPE is gathered in-graph, so one graph replays across all steps/prompts. Bit-exact vs eager.

- **7B target: ~1.00×** — memory-bandwidth-bound (reading ~15 GB of weights/token dominates).
- **0.5B draft: ~1.64×** — genuinely launch-bound; this is where graphs pay off.

See `documentation/cuda_graphs_explained.md` (concepts), `cuda_graph_issues_and_concepts.md`
(issues journal), and the `*_graph_benchmarks.md` docs.

## Speculative decoding

7B target verifies γ drafts proposed by the 0.5B draft (host-side stochastic accept/reject;
`documentation/speculative_decoding.md`).

```python
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.spec_decode import speculative_generate
res = speculative_generate(target_ex, DraftRunner(draft_ex, seed=0), prompt, n_new=64, gamma=4)
```

- **Single-process** (both models, one GPU — they fit: ~15 GB): `speculative_generate`. With greedy
  standardization it reproduces the target's greedy sequence exactly.
- **2-GPU MPI** (target on cuda:0, draft on cuda:1): `bash slurm/run_speculative.sh --steps 32 --gamma 4`.

## Tests

```bash
# CPU (login node): config / memory / structure / wire protocol
python -m unittest runtime.tests.test_config runtime.tests.test_mpi_protocol
# GPU (one targeted module per job — avoids the 30-min cap; one model copy per job)
bash slurm/run_tests_gpu.sh runtime.tests.test_decode_graph
bash slurm/run_tests_gpu.sh runtime.tests.test_draft_executor
```

## Docs

`documentation/runtime_architecture_and_execution.md` (forward pass), `runtime_kernel_system.md`
(kernels/build), `cuda_graphs_explained.md` + `cuda_graph_issues_and_concepts.md` (graphs),
`target_graph_benchmarks.md` + `draft_graph_benchmarks.md` (numbers). Plans: `graph_plan.md`,
`draft_integration_plan.md`, `runtime/plan.md`.
