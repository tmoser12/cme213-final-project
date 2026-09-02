"""Python host ABI for target embedding (AOT extension in this directory)."""

from __future__ import annotations

_EXT_MODULE = "runtime.production_kernels.target.embedding.target_embedding_ops"
_BUILD_HINT = "Run: bash scripts/build_kernels.sh embedding"


def _load_ext():
    try:
        from . import target_embedding_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/target/embedding" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under target/embedding/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def embedding_forward(input_ids, weight):
    return _load_ext().embedding_forward(input_ids, weight)


EXTENSION_MODULE = _EXT_MODULE
