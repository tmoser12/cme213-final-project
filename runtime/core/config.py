"""Load runtime config from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DTYPE_BYTES = {"fp16": 2, "fp32": 4, "int64": 8}

# Bundled configs shipped with the runtime
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_7B = CONFIGS_DIR / "qwen2.5-7b.yaml"
CONFIG_05B = CONFIGS_DIR / "qwen2.5-0.5b.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything the runtime needs, loaded from one YAML file."""

    name: str
    model_path: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    hidden_act: str
    dtype: str
    accum_dtype: str
    cuda_arch: str
    max_batch: int
    max_seq_len: int
    kv_cache_layout: str
    layer_order: tuple[str, ...]
    kernel_set: str = "target"  # which production_kernels/<role>/ tree the executor uses

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> RuntimeConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data, project_root=project_root)

    @classmethod
    def from_dict(cls, data: dict[str, Any], project_root: str | Path | None = None) -> RuntimeConfig:
        model_path = data["model_path"]
        if project_root is not None and not Path(model_path).is_absolute():
            model_path = str(Path(project_root) / model_path)

        return cls(
            name=data["name"],
            model_path=model_path,
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            vocab_size=int(data["vocab_size"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            tie_word_embeddings=bool(data["tie_word_embeddings"]),
            hidden_act=str(data["hidden_act"]),
            dtype=str(data["dtype"]),
            accum_dtype=str(data["accum_dtype"]),
            cuda_arch=str(data["cuda_arch"]),
            max_batch=int(data["max_batch"]),
            max_seq_len=int(data["max_seq_len"]),
            kv_cache_layout=str(data["kv_cache_layout"]),
            layer_order=tuple(data["layer_order"]),
            kernel_set=str(data.get("kernel_set", "target")),
        )

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act: {self.hidden_act}")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def dtype_bytes(self) -> int:
        return _DTYPE_BYTES[self.dtype]
