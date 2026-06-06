"""Scrappy Qwen2 inference runtime — config-driven, minimal surface area."""

from runtime.core import (
    CONFIG_05B,
    CONFIG_7B,
    RuntimeConfig,
    gpu_memory_snapshot,
    load_weights,
    load_weights_on_gpu,
    max_seq_len_after_weights,
    max_seq_len_for_budget,
    memory_report,
    plan_memory,
    startup_report,
    vram_budget,
)
from runtime.core import shapes
from runtime.buffers import RuntimeBuffers, allocate_buffers
from runtime.executor import Qwen2Executor

__all__ = [
    "Qwen2Executor",
    "RuntimeBuffers",
    "allocate_buffers",
    "CONFIG_05B",
    "CONFIG_7B",
    "RuntimeConfig",
    "gpu_memory_snapshot",
    "load_weights",
    "load_weights_on_gpu",
    "max_seq_len_after_weights",
    "max_seq_len_for_budget",
    "memory_report",
    "plan_memory",
    "shapes",
    "startup_report",
    "vram_budget",
]
