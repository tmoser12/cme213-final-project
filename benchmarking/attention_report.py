"""Summarize the paired attention experiments without requiring CUDA."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarking/optimizations/attention_v1'


def main():
    kernels = json.loads((OUT / 'ab.json').read_text())
    model = json.loads((OUT / 'model_ab.json').read_text())
    assert len(kernels['rows']) == 63 and all(r['bitwise_equal'] for r in kernels['rows'])
    assert len(model['rows']) == 30
    for data in [kernels, model]:
        for name, digest in data['metadata']['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for case in {r['case'] for r in model['rows']}:
        for repeat in range(5):
            pair = [r for r in model['rows'] if r['case'] == case and r['repeat'] == repeat]
            assert len(pair) == 2 and pair[0]['token_ids'] == pair[1]['token_ids']
    rows = []
    for case in kernels['rows']:
        row = {key: case[key] for key in ['role', 'op', 'batch', 'query', 'context', 'bitwise_equal']}
        for condition in ['warm', 'cold']:
            for variant in ['baseline', 'optimized']:
                row[f'{variant}_{condition}_us'] = median(
                    run['median'] for run in case['measurements'][variant][condition])
            row[f'{condition}_speedup'] = row[f'baseline_{condition}_us'] / row[f'optimized_{condition}_us']
        rows.append(row)
    with (OUT / 'kernel_summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Attention optimization results', '',
        'Both changes are retained in the target and draft production attention extensions. '
        'Measurements below compare the saved original binaries with rebuilt optimized binaries on the same RTX 4090.', '',
        '## Changes', '',
        '1. Pad the scalar decode/verification shared K tile from `[KV_TILE][D]` to `[KV_TILE][D + 2]`. '
        'The added two FP16 elements make the row stride odd in 32-bit bank words, removing the bank-address aliasing '
        'of column reads. Applied to both ordinary and split attention. Arithmetic and reduction order are unchanged.',
        '2. Give each RoPE/KV-write block at most 16 sequence rows. At batch one and 1,742 target prompt tokens, '
        'this increases the launch from 4 blocks to 436. Short sequences still use one block per KV head. '
        'Host and device cache positions, partial tiles, and empty writes remain supported.', '',
        'The WMMA prefill attention implementation is unchanged. The operation-level comparisons isolate the changes: '
        'RoPE measurements exercise sequence tiling; scalar attention measurements exercise shared-memory padding. '
        'No hardware bank-conflict counters were collected, so the timing improvement is measured while the '
        'bank-conflict explanation comes from inspecting the access pattern.', '',
        '## Kernel results', '',
        'Batch one; microseconds per operation. Each value is the median of three round medians. '
        'Warm timing reuses resident inputs; cold timing evicts cache before the operation. '
        'Speedup is original time divided by optimized time.', '',
        '| Role / operation | Query / context | Warm original → optimized | Speedup | Cold original → optimized | Speedup |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['batch'] != 1 or (row['op'] == 'rope' and row['query'] not in [1, 9, 128, 1742]):
            continue
        lines.append(f"| {row['role']} {row['op']} | {row['query']} / {row['context']} | "
            f"{row['baseline_warm_us']:.2f} → {row['optimized_warm_us']:.2f} | {row['warm_speedup']:.2f}× | "
            f"{row['baseline_cold_us']:.2f} → {row['optimized_cold_us']:.2f} | {row['cold_speedup']:.2f}× |")
    lines += ['', 'All 63 cases, including batches 1, 2, and 4, are in [kernel_summary.csv](kernel_summary.csv). '
        'Raw event samples and source/binary hashes are in [ab.json](ab.json). '
        'Tiny short-sequence differences should not be interpreted as reliable wins; event timing is coarse at a few microseconds.', '',
        '## Full target-only model', '',
        'Same loaded Qwen2.5-7B-Instruct weights and executor, swapping only the attention extension. '
        'Eager execution, batch one, 128 generated tokens, two warmups per variant and prompt, '
        'five paired repetitions with randomized variant order. Decode throughput covers the 127 steps after the first token. '
        'TTFT includes prefill and the first token. Values below are medians.', '',
        '| Prompt tokens | Original tokens/s | Optimized tokens/s | Throughput change | TTFT original → optimized (ms) |',
        '|---:|---:|---:|---:|---:|']
    for length in sorted({r['prompt_tokens'] for r in model['rows']}):
        grouped = {v: [r for r in model['rows'] if r['prompt_tokens'] == length and r['variant'] == v]
                   for v in ['baseline', 'optimized']}
        b, o = [median(r['decode_tps'] for r in grouped[v]) for v in ['baseline', 'optimized']]
        bt, ot = [median(r['ttft_ms'] for r in grouped[v]) for v in ['baseline', 'optimized']]
        lines.append(f'| {length} | {b:.2f} | {o:.2f} | {100*(o/b-1):+.2f}% | {bt:.2f} → {ot:.2f} |')
    lines += ['', 'Generated token IDs matched for every original/optimized pair. '
        'The end-to-end gains are much smaller than the isolated kernel gains because these operations occupy only part '
        'of each model step. Nine-token verification is relevant to speculative decoding; its speedup is not a measurement '
        'of complete speculative generation. Full-model batching and optimized graph-mode generation were not benchmarked here. '
        'The historical vLLM comparison remains under `benchmarking/results/`; it is not a fresh concurrent comparison '
        'against these modified kernels.', '',
        '## Validation and measurement limits', '',
        '- 63 paired operation/shape comparisons required bitwise-identical outputs before timing.',
        '- 45 runtime tests and 16 demo tests passed on the GPU; see [runtime-tests.log](runtime-tests.log) and [demo-tests.log](demo-tests.log).',
        '- Added tests cover both model dimensions, batches 1 and 2, empty/partial/exact RoPE tiles, untouched cache regions, '
        'nonzero positions, graph replay with changing device positions, query sizes 1–9, and contexts around the split boundary.',
        '- CUDA events are graph nodes around the operation, excluding Python launch overhead. Each of three paired rounds '
        'uses 20 samples; warm samples average 50 calls, cold samples time one call following a 128 MiB eviction outside the timed region.',
        '- This is one GPU and one session, not a confidence interval across machines. Cold-cache eviction is an approximation '
        'of a large working set, not a simulation of the entire inference memory hierarchy.', '',
        '## Saved originals and rollback', '',
        '`baseline/` contains the original target/draft attention CUDA sources, bindings, wrappers, associated documentation, '
        'and original compiled extensions. SHA-256 manifests protect the source and binary snapshots. '
        'Copies are saved both locally and on the GPU. The binary backups are ignored by Git and should be retained '
        'alongside the source snapshots if moving this directory.', '',
        'Preview without changes:', '', '```sh',
        'python3 benchmarking/optimizations/attention_v1/rollback.py', '```', '',
        'Restore and rebuild on the GPU, from the repository root:', '', '```sh',
        '.venv/bin/python benchmarking/optimizations/attention_v1/rollback.py --apply --rebuild', '```', '',
        'The restore first saves the displaced optimized sources and binaries to a timestamped directory. '
        'Restart any model process after rebuilding. The restore was exercised in a disposable copy, including exact '
        'comparison of restored originals and preserved displaced files; production remains optimized.', '',
        '## Reproduce', '', '```sh',
        'PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 BUILD_KERNEL=attention .venv/bin/python setup.py build_ext --inplace',
        '.venv/bin/python -m benchmarking.attention_ab',
        '.venv/bin/python -m benchmarking.attention_model_ab',
        '.venv/bin/python -m benchmarking.attention_report', '```', '',
        'Run GPU measurements sequentially on an idle device. The original binaries require the saved Python/PyTorch/CUDA ABI; '
        'source backups are the portable rollback path.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
