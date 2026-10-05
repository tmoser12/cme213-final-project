"""Regenerate CSVs, figures, and the defense notes from measured JSON only."""
import csv
import json
from pathlib import Path
import statistics
ROOT = Path(__file__).resolve().parents[1]

OUT=ROOT/'benchmarking/results'

def kernel_cost(row,name):
    """Optimistic logical traffic; NOT a claim of measured DRAM transactions."""
    b,f=row['bytes'],row['flops']
    if name=='decode_attn_split_kernel':
        h,s,d,n=row['hq'],row['s'],row['d'],row['splits']
        scratch=4*h*n*s*(d+2)
        # Replace final output with partial fp32 tensors; Q/rope are reused in cache.
        b=b-2*h*s*d+scratch
        f=f+3*h*s*d*(n-1)
    elif name=='decode_attn_combine_kernel':
        h,s,d,n=row['hq'],row['s'],row['d'],row['splits']
        b=4*h*n*s*(d+2)+2*h*s*d
        f=h*s*(3*n*d+4*n+d)
    return b,f

def main():
    data=json.loads((OUT/'kernels.json').read_text())
    rows=data['rows']; flat=[]; individual=[]
    for r in rows:
        flat.append(dict(role=r['role'],operation=r['op'],query_tokens=r['s'],context=r['t'],
                         logical_bytes=r['bytes'],useful_flops=r['flops'],ideal_dram_us=r['ideal_dram_us'],
                         warm_us=r['warm_us']['median'],cold_us=r['cold_us']['median'],
                         cold_p10_us=r['cold_us']['p10'],cold_p90_us=r['cold_us']['p90'],
                         cold_roofline_pct=100*r['roofline_efficiency_cold'],correct=r['correct']))
        for name,timing in r['kernels'].items():
            b,f=kernel_cost(r,name)
            peak=165.2 if name=='flash_attention_kernel' else 82.6
            ideal=max(b/1008000,f/(peak*1e6))
            cold=r['cold_kernels'][name]['median']
            individual.append(dict(role=r['role'],operation=r['op'],query_tokens=r['s'],context=r['t'],kernel=name,
                                   logical_bytes=b,useful_flops=f,arithmetic_intensity=f/b,
                                   theoretical_us=ideal,warm_profile_us=timing['median'],cold_profile_us=cold,
                                   dram_reference_pct=100*ideal/cold,
                                   cache_note='scratch is produced by preceding split kernel' if 'combine' in name else '128 MiB flush precedes operation'))
    for name,values in [('operations.csv',flat),('individual_kernels.csv',individual)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    engines={}
    for name in ['custom','vllm','custom-graph','vllm-eager']:
        path=OUT/(name+'.json')
        if path.exists(): engines[name]=json.loads(path.read_text())
    engine_rows=[]
    for name,data_e in engines.items():
        for t in sorted({r['prompt_tokens'] for r in data_e['rows']}):
            subset=[r for r in data_e['rows'] if r['prompt_tokens']==t]
            engine_rows.append(dict(engine=name,prompt_tokens=t,repeats=len(subset),
                                    tpot_ms=statistics.median(r['tpot_ms'] for r in subset),
                                    decode_tps=statistics.median(r['decode_tps'] for r in subset),
                                    decode_min_tps=min(r['decode_tps'] for r in subset),
                                    decode_max_tps=max(r['decode_tps'] for r in subset),
                                    ttft_ms=statistics.median(r['ttft_ms'] for r in subset)))
    if engine_rows:
        with (OUT/'engines.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(engine_rows[0])); writer.writeheader(); writer.writerows(engine_rows)
    lines=['# Measured results — RTX 4090, October 5, 2026','', '**Historical baseline:** These measurements precede the attention optimizations. See [the paired before/after report](../optimizations/attention_v1/REPORT.md) for the active kernels.', '',
           f"{len(rows)} operation/shape cases passed numerical checks. These results describe the original production kernels before the saved attention optimizations.",'',
           f"Measured 128 MiB copy bandwidth: **{data['measured_copy_GBs']:.1f} GB/s**, versus 1008 GB/s specified peak. GPU: {data['metadata']['gpu']}, {data['metadata']['sm_count']} SMs; torch {data['metadata']['torch']}.",'',
           '## Target-only engine comparison','',
           'Batch 1, FP16, Qwen2.5-7B-Instruct, 128 generated tokens, EOS ignored, identical prompt IDs, context capacity 2048. Two warmups per prompt length and five measured requests. TPOT = time from first to last output / 127; tokens/s = 127 / that interval. Tokenization, loading, compilation, HTTP, browser rendering, and detokenization are excluded. CPU scheduling, input copies, greedy selection, and GPU synchronization needed to return IDs are included. TTFT includes prefill and cache reset in the custom path. The draft model is not loaded.','',
           '| Engine | Prompt | Decode tokens/s (median; min–max) | TPOT ms | TTFT ms |','|---|---:|---:|---:|---:|']
    for r in engine_rows:
        lines.append(f"| {r['engine']} | {r['prompt_tokens']} | {r['decode_tps']:.2f} ({r['decode_min_tps']:.2f}–{r['decode_max_tps']:.2f}) | {r['tpot_ms']:.3f} | {r['ttft_ms']:.1f} |")
    if 'custom' in engines and 'vllm' in engines:
        assert engines['custom']['prompts_sha256']==engines['vllm']['prompts_sha256']
        lines+=['','The main comparison is **custom** (the demo’s eager target path) versus **vllm** (vLLM V1 with its default compilation/graph optimizations). `custom-graph` and `vllm-eager`, if present, are diagnostic configurations, not the demo default.','',
                f"vLLM version: {engines['vllm']['vllm_version']}. Both environments use torch {engines['vllm']['metadata']['torch']}; model revision is `{engines['vllm']['revision']}`.",'',
                '| Prompt | vLLM / custom decode throughput | Matching output prefix | Equal token IDs |','|---|---:|---:|---|']
        for t in [128,512,1742]:
            c=next(r for r in engine_rows if r['engine']=='custom' and r['prompt_tokens']==t)
            v=next(r for r in engine_rows if r['engine']=='vllm' and r['prompt_tokens']==t)
            custom_tokens=[r['token_ids'] for r in engines['custom']['rows'] if r['prompt_tokens']==t]
            vllm_tokens=[r['token_ids'] for r in engines['vllm']['rows'] if r['prompt_tokens']==t]
            prefix=min(next((i for i,(a,b) in enumerate(zip(ct,vt)) if a!=b),len(ct)) for ct in custom_tokens for vt in vllm_tokens)
            all_equal=all(ct==vt for ct in custom_tokens for vt in vllm_tokens)
            lines.append(f"| {t} | {v['decode_tps']/c['decode_tps']:.2f}× | {prefix}/{len(custom_tokens[0])} | {all_equal} (all repeats) |")
        lines+=['','Token agreement is checked, not assumed. FP16 kernel and reduction differences can change an argmax; divergent sequences still have equal lengths and computational shapes, but they are not evidence of exact model parity. All repeats and complete generated IDs are retained in JSON.']
    # Weight-streaming reference for the actual 7B config, excluding unused embedding rows.
    h,i,k,d,v,l=3584,18944,4,128,152064,28
    weight_bytes=l*(2*(h+2*k*d)*h+2*(h+2*k*d)+2*h*h+6*h*i+4*h)+4*h+2*v*h
    weight_ms=weight_bytes/1008e9*1000
    weight_tps=1008e9/weight_bytes
    if 'custom-graph' in engines:
        agreement=all(r['token_ids']==next(c['token_ids'] for c in engines['custom']['rows'] if c['case']==r['case']) for r in engines['custom-graph']['rows'])
        lines+=['',f'Existing target graph mode also matches the eager target tokens across all measured requests: **{agreement}**. It is an optional diagnostic, not a change to the demo default.']
    lines+=['','## Whole-target bandwidth reference','',
            f'The target reads approximately **{weight_bytes/1e9:.3f} GB of FP16 weights per decode step**: packed QKV, attention output, three MLP projections, normalization scales, LM head, and one embedding row. At 1008 GB/s, streaming those weights alone takes **{weight_ms:.2f} ms**, or **{weight_tps:.2f} tokens/s**. This excludes KV traffic, activations, scratch, compute, and CPU/device scheduling. It assumes weights are streamed once; cache residency can change actual DRAM bytes. It is a model-level bandwidth reference, not a measured counter or a promise of attainable latency.','']
    for r in engine_rows:
        if r['engine']=='custom':
            lines.append(f"- Custom at context {r['prompt_tokens']}: {r['decode_tps']:.2f} tokens/s, or {100*r['decode_tps']/weight_tps:.1f}% of this weight-streaming throughput reference.")
    lines+=['','## Representative custom-kernel measurements','',
            'Times below are CUDA graph event medians with Python dispatch excluded. Cold means a 128 MiB write precedes the operation, outside the timed region. Warm replays reuse inputs and may be served by L2. “Roofline %” is an optimistic logical-DRAM lower bound divided by cold measured time; it is not a measured hardware utilization counter. Values near or above 100% require cache/traffic analysis, not a claim of exceeding peak hardware throughput.','',
            '| Role | Operation | Query / context | Warm µs | Cold µs | DRAM lower bound µs | Roofline % |','|---|---|---:|---:|---:|---:|---:|']
    for r in flat:
        if (r['query_tokens'] in [1,1742] and r['context'] in [1,1742]) or (r['query_tokens']==9 and r['context']==1742):
            lines.append(f"| {r['role']} | {r['operation']} | {r['query_tokens']} / {r['context']} | {r['warm_us']:.3f} | {r['cold_us']:.3f} | {r['ideal_dram_us']:.4f} | {r['cold_roofline_pct']:.1f}% |")
    lines+=['','## Claims the data supports','',
            '- Cite measured latency at an explicit role, query length, and context. Avoid calling every kernel “near roofline.” The full table includes unfavorable cases.',
            '- Small kernels can be limited by launch/device scheduling, insufficient blocks, reductions, and special functions well before the full-chip memory roofline. The target RoPE/cache-write kernel launches only four blocks; draft launches two, versus 128 SMs on this GPU.',
            '- The split-KV heuristic targets 144 blocks based on an older 72-SM assumption. It is unchanged here. Split and combine are recorded individually in `individual_kernels.csv`, including their FP32 scratch traffic.',
            '- Prefill attention uses FP16 tensor-core products with FP32 accumulation; decode and verification use scalar arithmetic. Comparing scalar decode to advertised sparse tensor TOPS would be invalid.',
            '- Fusing RoPE with the cache write and attention avoids separate passes, and vectorized elementwise kernels reduce instruction count. This experiment measures their latency; it does not isolate the speedup of fusion versus an unfused implementation.',
            '- End-to-end engine performance also depends on cuBLAS projections, launch orchestration, cache layout, and sampling. Fast individual custom kernels do not establish superiority to vLLM.',
            '', '## Artifacts','',
            '- `operations.csv`: every tested wrapper/shape, warm/cold latency, numerical result, logical bytes/FLOPs and roofline.',
            '- `individual_kernels.csv`: every observed custom CUDA kernel, including split and combine, with profiler timing and per-kernel cost model.',
            '- `kernels.json`: all event samples, profiler duration samples, hardware metadata and runtime source hashes.',
            '- Engine JSON files: every request, each decode step, all output IDs, exact prompt hash and environment.',
            '- See `../README.md` for cost equations, scope, limitations and reproduction commands.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    try: plots(flat,engine_rows)
    except ImportError as exc: print('Plots unavailable:',exc)

def plots(rows,engine_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={name:plt.get_cmap('tab10')(i) for i,name in enumerate(['embedding','rmsnorm','residual','swiglu_activation','rope_kv_write','prefill_attention','decode_attention','verify_attention'])}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharex=True,sharey=True)
    for ax,role in zip(axes,['target','draft']):
        ops=sorted({r['operation'] for r in rows if r['role']==role})
        for op in ops:
            points=[r for r in rows if r['role']==role and r['operation']==op]
            ax.scatter([r['ideal_dram_us'] for r in points],[r['cold_us'] for r in points],s=22,label=op,color=colors[op.removesuffix('_dev')])
        ax.plot([.001,300],[.001,300],color='black',linestyle='--',label='ideal = measured')
        ax.set(xscale='log',yscale='log',xlabel='Optimistic DRAM/compute bound (µs)',ylabel='Measured cold GPU time (µs)',title=role.capitalize())
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Custom kernels: distance from the theoretical lower bound\nBatch 1, FP16, RTX 4090 — lower is better')
    fig.savefig(OUT/'kernel_bounds.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharex=True,sharey=True)
    for ax,role in zip(axes,['target','draft']):
        for op in sorted({r['operation'] for r in rows if r['role']==role}):
            points=[r for r in rows if r['role']==role and r['operation']==op and r['useful_flops']>0]
            if points:
                ax.scatter([r['useful_flops']/r['logical_bytes'] for r in points],
                           [r['useful_flops']/r['cold_us']/1e6 for r in points],s=22,label=op,color=colors[op.removesuffix('_dev')])
        xs=[10**(-2+j/30) for j in range(181)]
        ax.plot(xs,[min(1.008*x,82.6) for x in xs],color='black',label='Scalar roof')
        ax.plot(xs,[min(1.008*x,165.2) for x in xs],color='black',linestyle='--',label='Prefill tensor roof')
        ax.set(xscale='log',yscale='log',xlabel='Useful FLOPs / logical byte',ylabel='Useful TFLOP/s',title=role.capitalize())
        ax.grid(alpha=.2); ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Optimistic DRAM roofline — cold inputs, batch 1, RTX 4090\nEmbedding has no floating-point work and is omitted')
    fig.savefig(OUT/'roofline.png',dpi=180); plt.close(fig)
    if engine_rows:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        names=list(dict.fromkeys(r['engine'] for r in engine_rows))
        for i,name in enumerate(names):
            subset=[r for r in engine_rows if r['engine']==name]
            xs=[j+(i-(len(names)-1)/2)*.18 for j in range(len(subset))]
            ys=[r['decode_tps'] for r in subset]
            ax.bar(xs,ys,width=.17,label={'custom':'Demo target (eager)','vllm':'vLLM 0.10.2 (V1 graphs)','custom-graph':'Target graph diagnostic','vllm-eager':'vLLM 0.10.2 (eager)'}[name],yerr=[[r['decode_tps']-r['decode_min_tps'] for r in subset],[r['decode_max_tps']-r['decode_tps'] for r in subset]],capsize=3)
        ax.set(xticks=[0,1,2],xticklabels=['128','512','1742'],xlabel='Prompt tokens',ylabel='Decode tokens/s (127 steps)',title='Qwen2.5-7B-Instruct: one request, FP16, RTX 4090')
        ax.legend(); ax.grid(axis='y',alpha=.2)
        fig.savefig(OUT/'engine_decode.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
