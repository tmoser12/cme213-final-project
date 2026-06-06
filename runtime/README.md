# Qwen2 Inference Runtime

Minimal, config-driven runtime for custom CUDA kernel integration. Not production code — built for fast iteration on a research project.

## Layout

```
runtime/
├── core/                    # config, shapes, memory, weights
│   ├── configs/
│   │   ├── qwen2.5-7b.yaml  # 7B model constants + policy
│   │   └── qwen2.5-0.5b.yaml
│   ├── config.py            # load YAML → RuntimeConfig
│   ├── shapes.py            # shape/byte helpers (derived from config)
│   ├── memory.py            # plan_memory() estimates
│   └── weights.py           # load_weights() from safetensors
├── production_kernels/      # CUDA kernels (host-callable ops)
│   └── rmsnorm/             # ops.py: init / workspace_bytes / forward
├── tests/                   # unit tests (config, shapes, weights)
└── plan.md                  # implementation roadmap
```

Each YAML is the **single source of truth** for a model: architecture dims, dtype policy, paths, KV layout, and decoder layer order. Code derives `head_dim`, weight shapes, and buffer sizes from the loaded config.

## Usage

```python
from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.memory import plan_memory
from runtime.core import shapes

cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root="/path/to/project")
cfg.validate()

print(cfg.head_dim)                          # 128
print(shapes.q_states(1, 128, cfg))          # (1, 28, 128, 128)
print(plan_memory(cfg, batch=1, max_seq_len=512)["weight_mib"])

# Production: load 7B only (one model per GPU — never both on same device)
cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
weights, budget = load_weights_on_gpu(cfg, batch=1, reserve_mib=512)
print(f"free VRAM after 7B weights: {budget['gpu']['free_mib']:.0f} MiB")
print(f"max seq len for buffers: {budget['max_seq_len']}")
```

**Single-model assumption:** `vram_budget()` measures free HBM after exactly one model's weights are loaded. Use `CONFIG_7B` for production buffer sizing on 24GB Turing nodes. Never load 7B and 0.5B on the same GPU.

## 7B VRAM budget (Quadro RTX 6000, batch=1)

Measured on `gpu-turing` after loading **only** the 7B weights in FP16:

| | Value |
|---|---|
| GPU total | 22,682 MiB |
| 7B weights | 14,526 MiB |
| **Free after weights** | **7,984 MiB** |
| Buffer budget (free − 512 MiB reserve) | 7,472 MiB |
| Cost per seq position | ~413 KiB (activations + KV cache) |
| **Max seq len @ batch=1** | **18,525 tokens** |

The YAML default `max_seq_len: 2048` is conservative — ~18.5k tokens fits at batch size 1 on this GPU. The 512 MiB reserve is headroom for kernel scratch and fragmentation.

Re-measure on a GPU node:

```bash
PYTHONNOUSERSITE=1 srun --partition=gpu-turing --gres=gpu:1 \
  bash -lc 'cd $PROJECT_ROOT && conda run -n cme213 python -m runtime.tests.print_7b_vram'
```

GPU tests: `bash slurm/run_tests_gpu.sh` (defaults to 7B-only tests)

Pass either bundled config path or your own YAML — same code works for 7B and 0.5B.

## Tests

Setup tests (config, memory plan, engine-ready configuration) — run on CPU:

```bash
source setup.sh
python -m unittest runtime.tests.test_config runtime.tests.test_memory runtime.tests.test_engine_setup -v
```

GPU kernel / weight tests:

```bash
bash slurm/run_tests_gpu.sh runtime.tests.test_weights.TestGpuLoad7B
```

## Production kernels (Phase 3)

Kernels live under `runtime/production_kernels/target/<op>/`:

```
kernel.cu, bindings.cpp, ops.py, target_<op>_ops*.so   # built via scripts/build_kernels.sh
```

Build once, then run GPU parity tests in `runtime/tests/`:

```bash
bash scripts/build_kernels.sh              # all ops, or: bash scripts/build_kernels.sh rmsnorm
bash slurm/run_tests_gpu.sh runtime.tests.test_rmsnorm
```

```python
from runtime.production_kernels.target.rmsnorm import forward

out = forward(input, weight, eps)  # AOT extension — no runtime compile
```

## Next steps

See `plan.md` — remaining kernels + full decoder loop; allocate buffers via `plan_memory()`.
