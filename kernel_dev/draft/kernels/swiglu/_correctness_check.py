"""Fast correctness-only check: CustomQwenMLP vs HF Qwen2MLP (no timing).

Run on a GPU node:
    bash setup.sh   # or: module load gnu12/12.3.0 && conda activate cme213
    srun --partition=gpu-turing --gres=gpu:1 python -m kernel_dev.draft.kernels.swiglu._correctness_check
"""

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from kernel_dev.draft.kernels.swiglu.wrapper import CustomQwenMLP

cfg = Qwen2Config(hidden_size=896, intermediate_size=4864, hidden_act="silu")
hf = Qwen2MLP(cfg).cuda().half()
custom = CustomQwenMLP(hf).cuda()

torch.manual_seed(0)
ok = True
for (B, S) in [(1, 1), (1, 128), (2, 128), (8, 512), (16, 1024)]:
    x = torch.randn(B, S, 896, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        ref = hf(x)
        out = custom(x)
    max_abs = (ref - out).abs().max().item()
    close = torch.allclose(ref, out, atol=1e-2, rtol=1e-2)
    print(f"B={B:2d} S={S:5d}  shape={tuple(out.shape)}  "
          f"max_abs_diff={max_abs:.4e}  allclose(1e-2)={close}")
    ok = ok and close
print("RESULT:", "PASS" if ok else "FAIL")
