import torch
import torch.nn as nn

def pytorch_rmsnorm_baseline(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    The exact unoptimized PyTorch eager implementation of Qwen2RMSNorm.
    Extracted from transformers.models.qwen2.modeling_qwen2.
    
    Args:
        hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size)
        weight (torch.Tensor): Learned scaling weight of shape (hidden_size,)
        eps (float): Variance epsilon for numerical stability
        
    Returns:
        torch.Tensor: Normalized tensor of same shape and dtype as input
    """
    input_dtype = hidden_states.dtype
    # RMSNorm is typically computed in float32 for numerical stability
    hidden_states = hidden_states.to(torch.float32)
    
    # Compute variance: mean of squares along the hidden dimension
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    
    # Normalize
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    
    # Multiply by weight and cast back to original dtype
    return weight * hidden_states.to(input_dtype)

def test_baseline():
    """A quick function to ensure the baseline runs without errors."""
    print("Testing PyTorch RMSNorm baseline...")
    
    # Set up dummy variables matching Qwen 7B dimensions
    # batch=2, seq_len=128, hidden_size=3584
    batch_size = 2
    seq_len = 128
    hidden_size = 3584
    
    # Create random tensors in FP16 (which is what the model uses)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
    weight = torch.ones(hidden_size, dtype=torch.float16, device="cuda")
    
    # Run the function
    try:
        out = pytorch_rmsnorm_baseline(x, weight)
        print(f"✅ Success! Output shape: {out.shape}, Output dtype: {out.dtype}")
        
        # Verify it did something (variance should be ~1.0)
        # Note: Because the input is N(0,1), the variance is already ~1,
        # but scaling ensures it's exact.
        out_fp32 = out.to(torch.float32)
        measured_variance = out_fp32.pow(2).mean(-1)
        print(f"✅ Measured variance of output (should be very close to 1.0): {measured_variance[0, 0].item():.4f}")
        
    except Exception as e:
        print(f"❌ Error running baseline: {e}")

if __name__ == "__main__":
    test_baseline()
