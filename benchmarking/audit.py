"""CPU-only integrity checks for saved benchmark evidence (no GPU required)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarking/results'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--require-engines',action='store_true'); parser.add_argument('--source-snapshot',type=Path,help='Use saved source files where present when auditing a historical run'); args=parser.parse_args()
    data=json.loads((OUT/'kernels.json').read_text())
    for path,digest in data['metadata']['source_sha256'].items():
        source=ROOT/path
        if args.source_snapshot is not None and (args.source_snapshot/path).is_file():
            source=args.source_snapshot/path
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest, 'Runtime changed: '+path
    expected={'embedding_kernel','rmsnorm_forward_kernel_vectorized','residual_add_kernel','swiglu_act_kernel',
              'rope_kv_write_kernel','flash_attention_kernel','decode_attn_split_kernel','decode_attn_combine_kernel'}
    seen={}
    for r in data['rows']:
        assert r['correct'] is True
        assert r['bytes']>0 and r['flops']>=0
        for label in ['warm_us','cold_us']:
            values=r[label]['samples']
            assert len(values)==20 and all(math.isfinite(x) and x>0 for x in values)
            assert statistics.median(values)==r[label]['median']
        assert r['kernels'].keys()==r['cold_kernels'].keys()
        assert r['kernels'],'No CUDA kernel events found'
        seen.setdefault(r['role'],set()).update(r['kernels'])
    assert seen['target']==expected|{'decode_attn_kernel'},seen
    assert seen['draft']==expected,seen
    prompts_path=ROOT/'benchmarking/prompts.json'
    prompts=json.loads(prompts_path.read_text())
    prompt_hash=hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    for name in ['custom','vllm','custom-graph','vllm-eager']:
        path=OUT/(name+'.json')
        if not path.exists():
            assert not (args.require_engines and name in ['custom','vllm']), 'Missing '+name
            continue
        engine=json.loads(path.read_text())
        assert engine['prompts_sha256']==prompt_hash
        assert engine['dtype']=='float16' and engine['batch']==1
        n=engine['output_tokens']
        assert len(engine['rows'])==15,(name,len(engine['rows']))
        for row in engine['rows']:
            assert len(row['token_ids'])==n and len(row['step_ms'])==n-1
            assert all(math.isfinite(t) and t>0 for t in row['step_ms'])
            assert math.isclose(row['decode_tps'],(n-1)*1000/row['decode_ms'])
            assert math.isclose(row['tpot_ms'],row['decode_ms']/(n-1))
            assert any(len(p['token_ids'])==row['prompt_tokens'] for p in prompts)
            assert all(0<=t<152064 for t in row['token_ids'])
        print(name, '15 requests validated')
    print(len(data['rows']),'kernel cases; both roles covered; source hashes and sample arithmetic verified')
if __name__=='__main__': main()
