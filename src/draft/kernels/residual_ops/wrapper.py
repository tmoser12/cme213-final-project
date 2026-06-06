import torch.nn as nn

from src.draft.kernels.residual_ops.jit import load_residual_ops

custom_ops = load_residual_ops()


def residual_add(a, b):
    """out = a + b (elementwise fp16).

    The decoder layer's two "+ residual" writes. There's no HF submodule to
    monkey-patch (the add is an inline `+` inside Qwen2DecoderLayer.forward), so
    this stays a plain function the native runtime / parity tests call directly.
    a and b must be contiguous fp16 CUDA tensors of identical shape.
    """
    return custom_ops.residual_add_forward(a, b)


class CustomQwenLMHead(nn.Module):
    """Drop-in replacement for Qwen2ForCausalLM's lm_head (nn.Linear, bias=False).

    Runs the vocab projection logits = hidden @ weight^T as a cuBLAS GEMM in the
    host launcher. Adopts lm_head.weight by reference (no extra GPU memory).
    Qwen2.5-7B does NOT tie embeddings, so this weight is distinct from
    embed_tokens. Forward takes [..., H] and returns [..., vocab] fp16.
    """

    def __init__(self, original_lm_head):
        super().__init__()
        self.weight = original_lm_head.weight   # [vocab, H]

    def forward(self, x):
        *lead, H = x.shape
        M = 1
        for d in lead:
            M *= d
        logits = custom_ops.lm_head_forward(x.reshape(M, H), self.weight)  # [M, vocab]
        return logits.view(*lead, self.weight.size(0))


def patch_model(model):
    """Replace the model's lm_head with CustomQwenLMHead.

    (residual_add has no module to patch -- it's wired in by the native runtime,
    not the monkey-patcher.)
    """
    model.lm_head = CustomQwenLMHead(model.lm_head)
    print("✅ Patched lm_head with CustomQwenLMHead.")
