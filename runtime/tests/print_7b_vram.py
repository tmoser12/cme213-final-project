#!/usr/bin/env python3
"""Print 7B VRAM budget on GPU. Run via: bash slurm/run_tests_gpu.sh runtime.tests.print_7b_vram"""
import os
from pathlib import Path

from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights_on_gpu

root = os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=root)
_, b = load_weights_on_gpu(cfg, batch=1, reserve_mib=512)

gpu = b["gpu"]
w = b["weights"]
print("=== 7B on GPU (batch=1) ===")
print(f"GPU total:      {gpu['total_mib']:.0f} MiB")
print(f"Weights:        {w['total_mib']:.0f} MiB")
print(f"Reserved VRAM:  {gpu['reserved_mib']:.0f} MiB  (weights + CUDA context)")
print(f"Free VRAM:      {gpu['free_mib']:.0f} MiB  (available for buffers)")
print(f"Buffer budget:  {b['buffer_budget_mib']:.0f} MiB  (free minus 512 MiB reserve)")
print(f"Bytes/seq:      {b['runtime_bytes_per_seq'] / 1024:.1f} KiB")
print(f"Max seq len:    {b['max_seq_len']:,}  (activations + KV cache)")
print(f"Runtime at max: {b['runtime_mib_at_max_seq_len']:.0f} MiB")
print(f"Headroom left:  {b['headroom_mib_at_max_seq_len']:.0f} MiB")
