"""Profile the production target layer 0 without loading a second whole model.

Run from the repository root with: python -m benchmarks.profile_decode_layer
The real layer weights and token embeddings come from the pinned checkpoint.
NVTX labels wrap existing runtime calls; no production kernel is replaced.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors import safe_open

from runtime.buffers import allocate_buffers
from runtime.core.config import CONFIG_7B, RuntimeConfig
from runtime.core.weights import expected_keys
from runtime.executor import Qwen2Executor
from runtime.speculative.types import ForwardMode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotated(name, fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with torch.cuda.nvtx.range(name):
            return fn(*args, **kwargs)
    return call


class AnnotatedAttention:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        fn = getattr(self.original, name)
        return annotated(name, fn) if callable(fn) else fn


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=512, help='Cached tokens before the decode token')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen2.5-7B-Instruct')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'benchmarks/decode_layer_metadata.json')
    args = parser.parse_args()
    if not 1 <= args.context < 1742 or args.iterations < 1 or args.warmup < 1:
        parser.error('Require context 1..1741 and positive iteration/warmup counts')
    torch.cuda.set_device(0)
    cfg = replace(RuntimeConfig.from_yaml(CONFIG_7B, project_root=ROOT),
                  model_path=str(args.model), num_hidden_layers=1, max_seq_len=2048)
    prompt_path = ROOT / 'benchmarking/prompts.json'
    cases = json.loads(prompt_path.read_text())
    tokens = next(c['token_ids'] for c in cases if len(c['token_ids']) > args.context)[:args.context + 1]
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    names = [n for n in expected_keys(cfg) if n.startswith('model.layers.0.')]
    weights = {}
    # Read only this layer and needed embedding rows: no duplicate full-model residency.
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(str(args.model / name), framework='pt', device='cpu'))
                   for name in {index[n] for n in names + ['model.embed_tokens.weight']}}
        for name in names:
            tensor = handles[index[name]].get_tensor(name)
            assert tuple(tensor.shape) == expected_keys(cfg)[name]
            weights[name] = tensor.to(device='cuda', dtype=torch.float16)
        table = handles[index['model.embed_tokens.weight']].get_slice('model.embed_tokens.weight')
        rows = {token: table[token:token + 1].to(dtype=torch.float16) for token in set(tokens)}
        hidden = torch.cat([rows[token] for token in tokens]).unsqueeze(0).to('cuda')

    ex = Qwen2Executor(cfg, weights, allocate_buffers(cfg), use_cuda_graph=False)
    prefix = hidden[:, :args.context].contiguous()
    query = hidden[:, args.context:].contiguous()
    # First-layer inputs are embeddings, so this produces the real prefix KV state
    # without evaluating later layers or allocating the embedding/LM-head matrices.
    ex._run_decoder_layer(prefix, 0, args.context, mode=ForwardMode.PREFILL)
    ex._advance_cache_pos(args.context)

    def forward():
        # Intentionally repeat the SAME decode position: one newly written KV row,
        # identical valid context and algorithm dispatch on every profiled iteration.
        return ex._run_decoder_layer(query, 0, 1, mode=ForwardMode.DECODE)

    expected = forward().clone()
    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()
    ex._ops = {name: AnnotatedAttention(fn) if name == 'attn' else annotated(name, fn)
               for name, fn in ex._ops.items()}
    ex._run_attention = annotated('attention_block', ex._run_attention)
    torch.testing.assert_close(forward(), expected, atol=0, rtol=0)
    torch.cuda.synchronize()

    files = [Path(__file__), ROOT / 'runtime/executor.py', ROOT / 'runtime/buffers.py']
    files += sorted((ROOT / 'runtime/production_kernels/target').rglob('*.cu'))
    binaries = sorted((ROOT / 'runtime/production_kernels/target').rglob('*.so'))
    info = dict(model='Qwen2.5-7B-Instruct', layer_index=0, dtype='float16', batch=1,
                query_tokens=1, prefix_tokens=args.context, attended_tokens=args.context + 1,
                capacity=2048, iterations=args.iterations, warmup=args.warmup, execution='eager',
                input='Real checkpoint embeddings; next token is a fixed continuation from benchmark prompt, not sampled',
                isolation='Production layer 0 only; no embedding kernel, final norm, LM head, or other layers inside capture',
                repetition='Same input and cache position repeated; no per-operation synchronization',
                annotation_bitwise_equal=True, prompt_sha256=sha(prompt_path), token_ids=tokens,
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
                binary_sha256={str(p.relative_to(ROOT)): sha(p) for p in binaries},
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(info, indent=2) + '\n')
    print(f'Profiling target layer 0: prefix={args.context}, query=1, iterations={args.iterations}', flush=True)
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range('decoder_layer_decode'):
            for iteration in range(args.iterations):
                with torch.cuda.nvtx.range(f'decode_iteration_{iteration:03d}'):
                    output = forward()
            torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert torch.isfinite(output).all()
    print('Profile region completed; outputs match unannotated production layer bitwise.', flush=True)


if __name__ == '__main__':
    main()
