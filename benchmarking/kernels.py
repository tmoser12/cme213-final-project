"""Production custom-kernel rooflines, correctness, graph timing, and kernel traces."""
import argparse
import gc
import importlib
import json
import math
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarking.common import graph_time, metadata, summary, write_json
from runtime.core.config import RuntimeConfig, CONFIG_7B, CONFIG_05B

PEAKS = dict(dram_GBs=1008., scalar_TFLOPs=82.6, tensor_fp16_fp32_TFLOPs=165.2)
NAMES = ['embedding_kernel', 'rmsnorm_forward_kernel_vectorized', 'residual_add_kernel',
         'swiglu_act_kernel', 'rope_kv_write_kernel', 'flash_attention_kernel',
         'decode_attn_kernel', 'decode_attn_split_kernel', 'decode_attn_combine_kernel']

def rand(*shape): return torch.randn(shape, device='cuda', dtype=torch.float16)*.2

def profile(graph):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5): graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        prof.export_chrome_trace(tmp.name)
        events = json.loads(Path(tmp.name).read_text())['traceEvents']
    kernels = {}
    for ev in events:
        if ev.get('cat') == 'kernel':
            for name in NAMES:
                if name+'(' in ev['name'] or name+'<' in ev['name']:
                    kernels.setdefault(name, []).append(ev['dur'])
    return {name: summary(times) for name,times in kernels.items()}

def cases(role):
    cfg = RuntimeConfig.from_yaml(CONFIG_7B if role=='target' else CONFIG_05B)
    h,i,hq,hk,d = cfg.hidden_size,cfg.intermediate_size,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim
    base = 'runtime.production_kernels.'+role
    emb,norm,res,sw,attn = [importlib.import_module(base+'.'+part+'.ops') for part in ['embedding','rmsnorm','residual_ops','swiglu','attention']]
    for s in [1,2,4,5,8,9,128,512,1742]:
        # Same CUDA launch as production; embedding lookup uses the full vocabulary.
        x,y,w = rand(1,s,h),rand(1,s,h),rand(h)
        ids = torch.arange(s,device='cuda').reshape(1,s)
        ew = rand(cfg.vocab_size,h)
        yield dict(op='embedding',s=s,t=s,bytes=4*s*h+8*s,flops=0), lambda: emb.embedding_forward(ids,ew), lambda out: torch.testing.assert_close(out,ew[ids])
        del ew
        yield dict(op='rmsnorm',s=s,t=s,bytes=4*s*h+2*h,flops=4*s*h+2*s), lambda: norm.forward(x,w,1e-6), lambda out: torch.testing.assert_close(out,(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).half(),atol=.001,rtol=.005)
        yield dict(op='residual',s=s,t=s,bytes=6*s*h,flops=s*h), lambda: res.residual_add_forward(x,y), lambda out: torch.testing.assert_close(out,x+y)
        gate,up=rand(s,i),rand(s,i)
        yield dict(op='swiglu_activation',s=s,t=s,bytes=6*s*i,flops=4*s*i), lambda: sw._load_ext().swiglu_act_forward(gate,up), lambda out: torch.testing.assert_close(out,(F.silu(gate.float())*up.float()).half(),atol=.001,rtol=.005)
        del gate,up
        k,v=rand(1,hk,s,d),rand(1,hk,s,d)
        ck,cv=rand(1,hk,2048,d),rand(1,hk,2048,d)
        cos,sin=torch.ones((1,s,d),device='cuda',dtype=torch.float16),torch.zeros((1,s,d),device='cuda',dtype=torch.float16)
        # Identity rotation allows exact cache-write verification; attention below uses nonzero angles.
        dev=(role=='draft' and s==1)
        pos=torch.tensor(0,device='cuda') if dev else 0
        kvfn=attn.rope_kv_write_forward_dev if dev else attn.rope_kv_write_forward
        def run_kv():
            kvfn(k,v,ck,cv,pos,cos,sin)
            return ck[:,:,:s],cv[:,:,:s]
        def check_kv(out):
            torch.testing.assert_close(out[0],k); torch.testing.assert_close(out[1],v)
        yield dict(op='rope_kv_write_dev' if dev else 'rope_kv_write',s=s,t=s,bytes=8*hk*s*d+2*s*d,flops=3*hk*s*d),run_kv,check_kv
        del k,v,ck,cv,x,y,w
    for t in [128,512,1742]:
        for s in ([1,t] if role=='draft' else [1,2,4,5,8,9,t]):
            q,k,v=rand(1,hq,s,d),rand(1,hk,2048,d),rand(1,hk,2048,d)
            angles=torch.rand(1,s,d//2,device='cuda').repeat(1,1,2)
            cos,sin=angles.cos().half(),angles.sin().half()
            prefill=s==t
            dev=role=='draft' and s==1
            op='prefill_attention' if prefill else ('decode_attention_dev' if dev else ('decode_attention' if s==1 else 'verify_attention'))
            fn=attn.fused_attn_forward if prefill else (attn.decode_attn_forward_dev if dev else (attn.decode_attn_forward if s==1 else attn.small_q_attn_forward))
            length=torch.tensor(t,device='cuda') if dev else t
            def run(): return fn(q,k,v,length,1/math.sqrt(d),cos,sin)
            def check(out):
                rotated=(q.float()*cos[:,None].float()+torch.cat([-q[...,d//2:],q[...,:d//2]],-1).float()*sin[:,None].float()).half()
                keys=k[:,:,:t].repeat_interleave(hq//hk,dim=1)
                vals=v[:,:,:t].repeat_interleave(hq//hk,dim=1)
                mask=torch.arange(t,device='cuda')[None,:] <= (t-s+torch.arange(s,device='cuda')[:,None])
                ref=F.scaled_dot_product_attention(rotated.float(),keys.float(),vals.float(),attn_mask=mask).half()
                torch.testing.assert_close(out,ref,atol=.004,rtol=.04)
            pairs=s*t-s*(s-1)//2
            splits=min(math.ceil(144/hq),math.ceil((2048 if dev else t)/256),32) if not prefill else 1
            yield dict(op=op,s=s,t=t,bytes=4*hq*s*d+4*hk*t*d+2*s*d,
                       flops=4*hq*d*pairs+3*hq*s*d,tensor=prefill,splits=splits,hq=hq,hk=hk,d=d),run,check
            del q,k,v,cos,sin

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='benchmarking/results/kernels.json'); args=parser.parse_args()
    torch.manual_seed(213)
    meta=metadata()
    if '4090' not in meta['gpu']: raise RuntimeError('Set documented hardware peaks before running on another GPU')
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    copy_a=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8); copy_b=torch.empty_like(copy_a)
    timing,graph,_=graph_time(lambda: copy_b.copy_(copy_a))
    measured_bw=2*copy_a.numel()/timing['median']/1000
    del graph,copy_a,copy_b
    result=dict(metadata=meta,peaks=PEAKS,measured_copy_GBs=measured_bw,rows=[])
    for role in ['target','draft']:
        for spec,fn,check in cases(role):
            out=fn(); check(out); del out
            warm,g,out=graph_time(fn)
            cold,cg,co=graph_time(fn,inner=1,flush=flush)
            gpu_kernels=profile(g)
            cold_kernels=profile(cg)
            peak=PEAKS['tensor_fp16_fp32_TFLOPs'] if spec.get('tensor') else PEAKS['scalar_TFLOPs']
            ideal=max(spec['bytes']/(PEAKS['dram_GBs']*1000),spec['flops']/(peak*1e6))
            row=dict(role=role,**spec,warm_us=warm,cold_us=cold,kernels=gpu_kernels,cold_kernels=cold_kernels,
                     ideal_dram_us=ideal,roofline_efficiency_cold=ideal/cold['median'],
                     effective_GBs_cold=spec['bytes']/cold['median']/1000,correct=True)
            result['rows'].append(row)
            write_json(args.output,result)
            print(role,spec['op'],spec['s'],spec['t'],round(warm['median'],3),round(cold['median'],3),flush=True)
            del g,cg,out,co,fn,check
            gc.collect()
    print('Saved',args.output)
if __name__=='__main__': main()
