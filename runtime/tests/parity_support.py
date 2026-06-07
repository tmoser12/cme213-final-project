"""Shared helpers for staged HF parity tests (Phase 6)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from runtime.buffers import RuntimeBuffers, allocate_buffers
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import expected_keys
from runtime.executor import Qwen2Executor
from runtime.tests._support import PROJECT_ROOT

LOGITS_ATOL = 0.5
LOGITS_RTOL = 0.05
HIDDEN_ATOL = 0.5
HIDDEN_RTOL = 0.05

HAS_7B_WEIGHTS = os.path.isfile(
    Path(PROJECT_ROOT) / "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors"
)
HAS_05B_WEIGHTS = os.path.isfile(
    Path(PROJECT_ROOT) / "models/Qwen2.5-0.5B-Instruct/model.safetensors"
)

REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


def weights_from_hf_model(
    model: AutoModelForCausalLM,
    cfg: RuntimeConfig,
) -> dict[str, torch.Tensor]:
    """One GPU weight copy — do not also call load_weights() on the same device."""
    state = model.state_dict()
    return {name: state[name] for name in expected_keys(cfg)}


def load_hf_and_executor(
    cfg: RuntimeConfig,
    *,
    max_seq_len: int = 128,
    batch: int = 1,
) -> tuple[AutoModelForCausalLM, Qwen2Executor]:
    hf = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        .eval()
    )
    weights = weights_from_hf_model(hf, cfg)
    buffers = allocate_buffers(cfg, batch=batch, max_seq_len=max_seq_len, device="cuda")
    executor = Qwen2Executor(cfg, weights, buffers)
    return hf, executor


def capture_decoder_layer(
    hf_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hidden_in, hidden_out) for one HF decoder layer via forward hook."""
    storage: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output) -> None:
        storage["input"] = inputs[0].detach()
        storage["output"] = output[0].detach()

    handle = hf_model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            hf_model(input_ids)
    finally:
        handle.remove()

    return storage["input"], storage["output"]


def reset_layer_kv(executor: Qwen2Executor, layer_idx: int) -> None:
    """Zero one layer's KV slice and reset sequence cursor for isolated layer tests."""
    executor.buffers.kv_cache_k_layer(layer_idx).zero_()
    executor.buffers.kv_cache_v_layer(layer_idx).zero_()
    executor.buffers.reset_cache_position(0)
    executor._cache_pos = 0


def logits_allclose(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return torch.allclose(
        expected.float(),
        actual.float(),
        atol=LOGITS_ATOL,
        rtol=LOGITS_RTOL,
    )


def hidden_allclose(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return torch.allclose(
        expected.float(),
        actual.float(),
        atol=HIDDEN_ATOL,
        rtol=HIDDEN_RTOL,
    )


def greedy_decode_hf(
    hf_model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> list[int]:
    with torch.no_grad():
        out = hf_model.generate(
            prompt_ids,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            pad_token_id=hf_model.config.eos_token_id,
        )
    return out[0].tolist()


def greedy_decode_executor(
    executor: Qwen2Executor,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> list[int]:
    """
    Greedy token trajectory matching HF ``generate(..., do_sample=False)``.

    The first new token comes from prefill logits; only subsequent tokens use
    ``decode_step`` (prefill already consumed the prompt in the KV cache).
    """
    with torch.no_grad():
        full = executor.greedy_extend(prompt_ids, n_new_tokens)
    return full[0].tolist()


def default_7b_cfg() -> RuntimeConfig:
    return RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)


def default_05b_cfg() -> RuntimeConfig:
    return RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
