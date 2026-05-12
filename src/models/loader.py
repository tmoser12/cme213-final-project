"""
src/models/loader.py
Loads a Qwen2.5 model and tokenizer in FP16 onto a specific CUDA device.
Both target and draft share the same loader; they differ only in model_id and device.
"""

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModelBundle:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device


def load_model(model_id: str, device: str) -> ModelBundle:
    """
    Load model + tokenizer in FP16.

    Args:
        model_id: Model ID or local path, e.g. "./models/Qwen2.5-7B-Instruct"
        device:   CUDA device string, e.g. "cuda:0" or "cuda:1"

    Returns:
        ModelBundle with model in eval mode on the specified device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    return ModelBundle(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )
