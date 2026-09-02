"""Shared profile/report path resolution for the kernel benchmarks.

Every benchmark lives at ``.../src/<model>/kernels/<kernel>/...`` where
``<model>`` is ``draft`` or ``target``. Profiles (nsys/ncu) and benchmark
reports both land under::

    results/profiles/<model>/<group>/<label>_<stamp>.<ext>

with ``<group>`` derived from the kernel name (all attention sub-ops fold into
``attention``; rmsnorm/embedding/residual_ops fold into ``misc``). The model is
encoded by the directory, so filenames drop the model prefix.

The ``run_benchmark.sh`` scripts compute the same ``<model>/<group>`` dir in
shell for nsys/ncu output; this module is the Python side, used by the
``benchmark.py`` modules to place their ``benchmark_report.txt`` equivalent.
"""

from datetime import datetime
from pathlib import Path

# kernel dir name -> profile group. Anything unlisted falls back to "misc".
GROUPS = {
    "swiglu": "swiglu",
    "attention": "attention",
    "rmsnorm": "misc",
    "embedding": "misc",
    "residual_ops": "misc",
}


def _resolve(source_file):
    """Return (project_root, model, group) from a benchmark's ``__file__``."""
    parts = Path(source_file).resolve().parts
    i = parts.index("kernel_dev")
    model = parts[i + 1]                            # "draft" | "target"
    kernel = parts[parts.index("kernels") + 1]      # "attention" even for sub-ops
    group = GROUPS.get(kernel, "misc")
    project_root = Path(*parts[:i])                 # parent of kernel_dev/
    return project_root, model, group


def profile_dir(source_file):
    """``results/profiles/<model>/<group>/`` for the caller, created on demand."""
    project_root, model, group = _resolve(source_file)
    out = project_root / "results" / "profiles" / model / group
    out.mkdir(parents=True, exist_ok=True)
    return out


def stamp():
    """Timestamp matching the shell scripts' ``date +%Y%m%d_%H%M%S``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def report_file(source_file, label):
    """Timestamped report path ``<profile_dir>/<label>_<stamp>.txt`` (never overwrites)."""
    return profile_dir(source_file) / f"{label}_{stamp()}.txt"
