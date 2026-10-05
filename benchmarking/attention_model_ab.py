"""Paired target-only engine comparison, changing only the attention extension."""
import argparse
import json
from pathlib import Path
import random
import torch
from benchmarking.attention_ab import load_pair
from benchmarking.common import ROOT, metadata, write_json
from benchmarking.decode import Custom


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmarking/optimizations/attention_v1/model_ab.json');a=p.parse_args()
    random.seed(217)
    cases=json.loads((ROOT/'benchmarking/prompts.json').read_text())
    engine=Custom(str(ROOT/'models/Qwen2.5-7B-Instruct'),False)
    baseline,optimized=load_pair('target')
    result={'metadata':metadata(),'output_tokens':128,'rows':[],
            'method':'Same loaded target and weights; swap only attention extension; two warmups per variant and shape; five randomized paired repetitions'}
    for case in cases:
        for ext in [baseline,optimized]:
            engine.executor._ops['attn']=ext
            for _ in range(2): engine.run(case['token_ids'],128)
        for repeat in range(5):
            order=[('baseline',baseline),('optimized',optimized)];random.shuffle(order)
            results={}
            for name,ext in order:
                engine.executor._ops['attn']=ext
                row=engine.run(case['token_ids'],128)
                row.update(variant=name,case=case['name'],prompt_tokens=len(case['token_ids']),repeat=repeat,
                           decode_tps=127000/row['decode_ms'],tpot_ms=row['decode_ms']/127)
                results[name]=row;result['rows'].append(row)
                print(case['name'],repeat,name,round(row['decode_tps'],3),round(row['ttft_ms'],3),flush=True)
            assert results['baseline']['token_ids']==results['optimized']['token_ids'],'Generated tokens changed'
            write_json(a.output,result)
if __name__=='__main__':main()
