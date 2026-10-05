"""Paired before/after timing using saved and current AOT attention extensions."""
import argparse
import gc
import hashlib
import importlib
import importlib.util
from pathlib import Path
import random
import torch
from benchmarking.common import ROOT, graph_time, metadata, write_json

BASE=ROOT/'benchmarking/optimizations/attention_v1/baseline'


def load_pair(role):
    path=next((BASE/'binaries'/role).glob('*.so'))
    spec=importlib.util.spec_from_file_location(f'baseline_{role}.{role}_attention_ops',path)
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    current=importlib.import_module(f'runtime.production_kernels.{role}.attention.ops')._load_ext()
    return original,current


def rand(*shape):
    return torch.randn(shape,device='cuda',dtype=torch.float16)*.3


def cases(pair, role):
    heads,kh,d=(28,4,128) if role=='target' else (14,2,64)
    for batch in [1,2,4]:
        for s in [1,9,17,128,512,1742]:
            k,v=rand(batch,kh,s,d),rand(batch,kh,s,d)
            cks=[rand(batch,kh,2048,d) for _ in pair]
            cvs=[rand(batch,kh,2048,d) for _ in pair]
            angles=rand(batch,s,d//2).repeat(1,1,2)
            cos,sin=angles.cos(),angles.sin()
            pos=3
            dev=role=='draft' and s==1
            wp=torch.tensor(pos,device='cuda') if dev else pos
            def make(ext,ck,cv):
                fn=ext.rope_kv_write_forward_dev if dev else ext.rope_kv_write_forward
                def run():
                    fn(k,v,ck,cv,wp,cos,sin)
                    return ck[:,:,pos:pos+s],cv[:,:,pos:pos+s]
                return run
            yield dict(role=role,op='rope',batch=batch,query=s,context=s),[make(ext,ck,cv) for ext,ck,cv in zip(pair,cks,cvs)]
    for batch in [1,2,4]:
        for t in [128,512,1742]:
            for s in ([1,9] if role=='target' else [1]):
                q,k,v=rand(batch,heads,s,d),rand(batch,kh,2048,d),rand(batch,kh,2048,d)
                angles=rand(batch,s,d//2).repeat(1,1,2)
                cos,sin=angles.cos(),angles.sin()
                dev=role=='draft'
                length=torch.tensor(t,device='cuda') if dev else t
                def make(ext):
                    fn=ext.small_q_attn_forward_dev if dev else ext.small_q_attn_forward
                    return lambda:fn(q,k,v,length,d**-.5,cos,sin)
                yield dict(role=role,op='decode' if s==1 else 'verify',batch=batch,query=s,context=t),[make(ext) for ext in pair]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmarking/optimizations/attention_v1/ab.json');a=p.parse_args()
    torch.manual_seed(216); random.seed(216)
    result={'metadata':metadata(),'rows':[],'method':'3 paired rounds, randomized baseline/optimized order; 20 CUDA graph event samples per round'}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for role in ['target','draft']:
        pair=load_pair(role)
        result[role+'_binary_sha256']={name:hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest() for name,ext in zip(['baseline','optimized'],pair)}
        for info,fns in cases(pair,role):
            outs=[fn() for fn in fns]
            torch.testing.assert_close(outs[0],outs[1],atol=0,rtol=0)
            del outs
            samples={name:{'warm':[],'cold':[]} for name in ['baseline','optimized']}
            for repeat in range(3):
                order=[0,1];random.shuffle(order)
                for i in order:
                    name=['baseline','optimized'][i]
                    for condition,inner,evict in [('warm',50,None),('cold',1,flush)]:
                        timing,g,out=graph_time(fns[i],inner=inner,flush=evict)
                        samples[name][condition].append(timing)
                        del g,out
            row={**info,'bitwise_equal':True,'measurements':samples}
            result['rows'].append(row);write_json(a.output,result)
            b=sum(x['median'] for x in samples['baseline']['warm'])/3
            o=sum(x['median'] for x in samples['optimized']['warm'])/3
            print(role,info['op'],info['batch'],info['query'],info['context'],round(b,3),round(o,3),round(b/o,2),flush=True)
            del fns;gc.collect()
if __name__=='__main__':main()
