#!/usr/bin/env python3
"""
scripts/verify_env.py
Run on a GPU node to confirm the environment, CUDA, and both models load correctly.

  srun --partition=gpu-turing --gres=gpu:1 --pty python scripts/verify_env.py
"""

import torch
import sys


def check_cuda():
    print("=== CUDA ===")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Are you on a GPU node?")
        sys.exit(1)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  Device          : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute cap     : SM {props.major}.{props.minor}")
    total_gb = props.total_memory / 1e9
    print(f"  VRAM            : {total_gb:.1f} GB")
    print()


def check_model(model_id: str, device: str = "cuda:0"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time

    print(f"=== Loading {model_id} ===")
    mem_before = torch.cuda.memory_allocated(device) / 1e9

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    load_time = time.time() - t0

    mem_after = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Load time   : {load_time:.1f}s")
    print(f"  Weight mem  : {mem_after - mem_before:.2f} GB")
    print(f"  Vocab size  : {tokenizer.vocab_size}")

    # Minimal forward pass to confirm no errors
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Sample output: '{result.strip()}'")
    print()

    # Free memory before loading next model
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    check_cuda()
    check_model("Qwen/Qwen2.5-7B-Instruct")
    check_model("Qwen/Qwen2.5-0.5B-Instruct")
    print("All checks passed.")
