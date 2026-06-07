"""Python host ABI for draft residual_ops (AOT extension in this directory)."""

from __future__ import annotations

_EXT_MODULE = "runtime.production_kernels.draft.residual_ops.draft_residual_ops"
_BUILD_HINT = "Run: bash scripts/build_kernels.sh draft residual_ops"


def _load_ext():
    try:
        from . import draft_residual_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/draft/residual_ops" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under draft/residual_ops/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def residual_add_forward(a, b):
    return _load_ext().residual_add_forward(a, b)


def lm_head_forward(hidden, weight):
    ext = _load_ext()
    if hidden.dim() == 3:
        batch, seq, hidden_size = hidden.shape
        flat = hidden.reshape(batch * seq, hidden_size)
        logits = ext.lm_head_forward(flat, weight)
        return logits.reshape(batch, seq, weight.shape[0])
    return ext.lm_head_forward(hidden, weight)


EXTENSION_MODULE = _EXT_MODULE
