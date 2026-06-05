import torch
import torch.nn as nn

from jit import load_embedding_ops

custom_ops = load_embedding_ops()


class CustomEmbedding(nn.Module):
    def __init__(self, original_embedding: nn.Embedding):
        super().__init__()
        # Steal the weight reference from the HF model so it stays on the GPU
        # and requires no extra memory.
        self.weight = original_embedding.weight

    def forward(self, input_ids):
        return custom_ops.embedding_forward(input_ids, self.weight)


def patch_model(model):
    """Entry point to replace the input embedding layer in the model."""
    model.model.embed_tokens = CustomEmbedding(model.model.embed_tokens)
    print("✅ Patched 1 Embedding layer with CustomEmbedding.")
