import torch.nn as nn

from src.draft.kernels.swiglu.jit import load_swiglu_ops

custom_ops = load_swiglu_ops()


class CustomQwenMLP(nn.Module):
    """Drop-in replacement for Qwen2MLP.

    The whole MLP runs in the custom extension: swiglu_forward does the gate /
    up / down projections as cuBLAS GEMMs in the host launcher plus the custom
    silu(gate) * up activation kernel, returning the down_proj output [M, H]
    that the decoder layer adds straight into the residual stream. We only adopt
    the three projection weights (by reference; no extra GPU memory). Forward
    signature mirrors Qwen2MLP so patch_model() can splice us into HF's
    DecoderLayer.
    """

    def __init__(self, original_mlp):
        super().__init__()
        # Adopt the projection weights by reference (stay on the GPU, no copy).
        self.W_gate = original_mlp.gate_proj.weight   # [I, H]
        self.W_up = original_mlp.up_proj.weight       # [I, H]
        self.W_down = original_mlp.down_proj.weight   # [H, I]

    def forward(self, x):
        B, S, H = x.shape
        out = custom_ops.swiglu_forward(
            x.reshape(B * S, H), self.W_gate, self.W_up, self.W_down)  # [M, H]
        return out.view(B, S, H)


def patch_model(model):
    """Replace every layer.mlp with CustomQwenMLP."""
    n = 0
    for layer in model.model.layers:
        layer.mlp = CustomQwenMLP(layer.mlp)
        n += 1
    print(f"✅ Patched {n} MLP layers with CustomQwenMLP.")
