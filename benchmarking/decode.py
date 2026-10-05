"""Batch-one in-process engine timing: prefill/first token and N-1 decode steps."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from benchmarking.common import ROOT, metadata, write_json

REVISION='a09a35458c702b33eeacc393d103063234e8bc28'

def prompts(model_path):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_path)
    text=('Explain how GPU memory bandwidth, arithmetic intensity, kernel launch overhead, '
          'and caching affect transformer inference. Give concrete examples and a careful comparison. ')*500
    ids=tok.encode(text,add_special_tokens=False)
    return [dict(name=f'context_{n}',token_ids=ids[:n]) for n in [128,512,1742]]

class Custom:
    def __init__(self,path,graph):
        from runtime.buffers import allocate_buffers
        from runtime.core.config import RuntimeConfig, CONFIG_7B
        from runtime.core.weights import load_weights
        from runtime.executor import Qwen2Executor
        cfg=replace(RuntimeConfig.from_yaml(CONFIG_7B),model_path=path)
        self.executor=Qwen2Executor(cfg,load_weights(cfg),allocate_buffers(cfg),use_cuda_graph=graph)
        self.graph=graph
    @torch.inference_mode()
    def run(self,ids,n):
        ex=self.executor
        prompt=torch.tensor([ids],device='cuda',dtype=torch.int64)
        torch.cuda.synchronize()
        t0=time.perf_counter()
        logits=ex.prefill(prompt)
        first=int(logits[0,-1].argmax().item())
        t1=time.perf_counter()
        times=[]; tokens=[first]
        decode=ex.decode_step_graph if self.graph else ex.decode_step
        for _ in range(n-1):
            t=time.perf_counter()
            inp=torch.tensor([tokens[-1]],device='cuda',dtype=torch.int64)
            logits=decode(inp)
            tokens.append(int(logits[0,-1].argmax().item()))
            times.append((time.perf_counter()-t)*1000)
        t2=time.perf_counter()
        return dict(ttft_ms=(t1-t0)*1000,decode_ms=(t2-t1)*1000,total_ms=(t2-t0)*1000,
                    step_ms=times,token_ids=tokens)

class Vllm:
    def __init__(self,path,eager):
        from vllm import LLM, SamplingParams
        self.SamplingParams=SamplingParams
        self.llm=LLM(model=path,tokenizer=path,dtype='float16',tensor_parallel_size=1,
                     max_model_len=2048,max_num_seqs=1,gpu_memory_utilization=.85,
                     enforce_eager=eager,enable_prefix_caching=False,
                     seed=213,disable_log_stats=True)
        self.engine=self.llm.llm_engine
        self.counter=0
    def run(self,ids,n):
        self.counter+=1
        params=self.SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,detokenize=False)
        t0=time.perf_counter()
        self.engine.add_request(str(self.counter),{'prompt_token_ids':ids},params)
        times=[]; prev=t0; first_time=None; final=None; emissions=[]
        while self.engine.has_unfinished_requests():
            outputs=self.engine.step()
            for output in outputs:
                if not output.outputs: continue
                now=time.perf_counter()
                final=output.outputs[0].token_ids
                emissions.append(dict(time_ms=(now-t0)*1000,count=len(final)))
                if first_time is None: first_time=now
                else: times.append((now-prev)*1000)
                prev=now
        assert final is not None and len(final)==n,(len(final) if final else None,n)
        assert emissions[0]['count']==1,'First emission is not one token; cannot isolate decode'
        assert [e['count'] for e in emissions]==list(range(1,n+1)),'Engine buffered output; per-token timing invalid'
        return dict(ttft_ms=(first_time-t0)*1000,decode_ms=(prev-first_time)*1000,
                    total_ms=(prev-t0)*1000,step_ms=times,token_ids=list(final),emissions=emissions)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['custom','custom-graph','vllm','vllm-eager'],required=True)
    p.add_argument('--model',default=str(ROOT/'models/Qwen2.5-7B-Instruct'))
    p.add_argument('--prompts',default='benchmarking/prompts.json'); p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=5); p.add_argument('--tokens',type=int,default=128)
    a=p.parse_args()
    if not Path(a.prompts).exists(): write_json(a.prompts,prompts(a.model))
    cases=json.loads(Path(a.prompts).read_text())
    meta=metadata()
    result=dict(metadata=meta,engine=a.engine,model=a.model,revision=REVISION,dtype='float16',batch=1,
                output_tokens=a.tokens,ignore_eos=True,prefix_cache=False,rows=[],
                prompts_sha256=hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest())
    if a.engine.startswith('vllm'): result['vllm_version']=importlib.metadata.version('vllm')
    engine=Vllm(a.model,a.engine=='vllm-eager') if a.engine.startswith('vllm') else Custom(a.model,a.engine=='custom-graph')
    if a.engine.startswith('vllm'):
        cfg=engine.engine.vllm_config
        result['effective_settings']={
            'engine_class':type(engine.engine).__module__,
            'enable_chunked_prefill':cfg.scheduler_config.enable_chunked_prefill,
            'max_num_batched_tokens':cfg.scheduler_config.max_num_batched_tokens,
            'enable_prefix_caching':cfg.cache_config.enable_prefix_caching,
            'enforce_eager':cfg.model_config.enforce_eager,
            'kv_cache_dtype':cfg.cache_config.cache_dtype,
        }
    for case in cases:
        assert len(case['token_ids'])+a.tokens<=2048
        for _ in range(2): engine.run(case['token_ids'],a.tokens)
        for repeat in range(a.repeats):
            row=engine.run(case['token_ids'],a.tokens)
            row.update(case=case['name'],prompt_tokens=len(case['token_ids']),repeat=repeat,
                       decode_tps=(a.tokens-1)*1000/row['decode_ms'],tpot_ms=row['decode_ms']/(a.tokens-1))
            result['rows'].append(row); write_json(a.output,result)
            print(a.engine,case['name'],repeat,round(row['decode_tps'],2),flush=True)
    result['after']=metadata()
    write_json(a.output,result)
if __name__=='__main__': main()
