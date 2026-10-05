"""Shared measurement and provenance helpers; run modules from the repository root."""
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')

def metadata():
    props = torch.cuda.get_device_properties(0)
    files = sorted((ROOT / 'runtime').rglob('*.py')) + sorted((ROOT / 'runtime/production_kernels').rglob('*.cu'))
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=props.name, sm_count=props.multi_processor_count, total_memory=props.total_memory,
                nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

def summary(values):
    values = sorted(values)
    return dict(median=statistics.median(values), min=values[0], max=values[-1],
                p10=values[int(.1*(len(values)-1))], p90=values[int(.9*(len(values)-1))], samples=values)

def graph_time(fn, repeats=20, inner=50, flush=None):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    # Event nodes live inside the graph: Python launch latency is excluded.
    # A cold sample flushes 128 MiB before its START event, outside timed work.
    with torch.cuda.graph(graph, stream=stream):
        if flush is not None: flush.zero_()
        start.record()
        for _ in range(inner): output = fn()
        end.record()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end)*1000/inner)
    graph._timing_events = (start, end)  # Keep event handles alive for later profiler replay.
    return summary(samples), graph, output
