"""Python host ABI for draft embedding (AOT extension in this directory)."""

from __future__ import annotations

_EXT_MODULE = "runtime.production_kernels.draft.embedding.draft_embedding_ops"
_BUILD_HINT = "Run: BUILD_ROLE=draft BUILD_KERNEL=embedding python setup.py build_ext --inplace"


# WALKTHROUGH [DRAFT_EMBEDDING_WRAPPER] — see demo/WALKTHROUGH.md.
# Import the prebuilt draft module; Python caches successful imports.
# This boundary passes CUDA tensors by reference and never JIT-compiles a kernel.
def _load_ext():
    try:
        from . import draft_embedding_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/draft/embedding" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under draft/embedding/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def embedding_forward(input_ids, weight):
    return _load_ext().embedding_forward(input_ids, weight)


EXTENSION_MODULE = _EXT_MODULE
