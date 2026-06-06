"""Shared paths and constants for runtime/tests."""

from __future__ import annotations

import os
from pathlib import Path

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig

TESTS_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = TESTS_DIR.parent
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(RUNTIME_DIR.parent))
)

LAYER_ORDER = (
    "input_rmsnorm",
    "attention",
    "residual_add",
    "post_attn_rmsnorm",
    "swiglu_mlp",
    "residual_add",
)


def load_7b() -> RuntimeConfig:
    cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    cfg.validate()
    return cfg


def load_05b() -> RuntimeConfig:
    cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    cfg.validate()
    return cfg
