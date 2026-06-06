import torch
import torch.nn as nn
from src.draft.kernels.rmsnorm.jit import get_ops

custom_ops = get_ops()

class CustomQwenRMSNorm(nn.Module):
    def __init__(self, original_rmsnorm):
        super().__init__()
        # We steal the exact weight reference from the HF model, 
        # meaning it stays on the GPU and requires no extra memory.
        self.weight = original_rmsnorm.weight
        self.variance_epsilon = original_rmsnorm.variance_epsilon

    def forward(self, hidden_states):
        # Dispatch directly to our compiled CUDA kernel
        return custom_ops.forward(hidden_states, self.weight, self.variance_epsilon)

def patch_model(model):
    """Entry point to replace all RMSNorm layers in the model."""
    num_patched = 0
    
    # Replace input_layernorm and post_attention_layernorm in every layer
    for layer in model.model.layers:
        layer.input_layernorm = CustomQwenRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = CustomQwenRMSNorm(layer.post_attention_layernorm)
        num_patched += 2
        
    # Replace the final norm block
    if hasattr(model.model, 'norm'):
        model.model.norm = CustomQwenRMSNorm(model.model.norm)
        num_patched += 1
        
    print(f"✅ Patched {num_patched} RMSNorm layers with CustomQwenRMSNorm.")
