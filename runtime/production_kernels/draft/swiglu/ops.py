"""Python host ABI for draft SwiGLU (AOT extension in this directory)."""

from __future__ import annotations

_EXT_MODULE = "runtime.production_kernels.draft.swiglu.draft_swiglu_ops"
_BUILD_HINT = "Run: BUILD_ROLE=draft BUILD_KERNEL=swiglu python setup.py build_ext --inplace"


# WALKTHROUGH [DRAFT_SWIGLU_WRAPPER] — see demo/WALKTHROUGH.md.
# Import the prebuilt draft module; Python caches successful imports.
# This boundary passes CUDA tensors by reference and never JIT-compiles a kernel.
def _load_ext():
    try:
        from . import draft_swiglu_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/draft/swiglu" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under draft/swiglu/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def swiglu_forward(x, w_gate, w_up, w_down):
    ext = _load_ext()
    if x.dim() == 3:
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)
        out = ext.swiglu_forward(flat, w_gate, w_up, w_down)
        return out.reshape(batch, seq, hidden)
    return ext.swiglu_forward(x, w_gate, w_up, w_down)


EXTENSION_MODULE = _EXT_MODULE
