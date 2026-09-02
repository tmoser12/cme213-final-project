"""Python host ABI for target RMSNorm (AOT extension in this directory)."""

from __future__ import annotations

_EXT_MODULE = "runtime.production_kernels.target.rmsnorm.target_rmsnorm_ops"
_BUILD_HINT = "Run: bash scripts/build_kernels.sh rmsnorm"


def _load_ext():
    try:
        from . import target_rmsnorm_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/target/rmsnorm" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under target/rmsnorm/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def init() -> None:
    """Verify the prebuilt extension imports (no runtime compile)."""
    _load_ext()


def workspace_bytes(*, batch: int, seq_len: int, hidden_size: int) -> int:
    del batch, seq_len, hidden_size
    return 0


def forward(input, weight, eps: float):
    return _load_ext().forward(input, weight, eps)


EXTENSION_MODULE = _EXT_MODULE
