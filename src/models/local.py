"""src/models/local.py - Local inference via transformers"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class LocalModel(BaseModel):
    """Load model locally via transformers"""

    def __init__(self, model_name: str, model_path: str,
                 device: str = "auto", **kwargs):
        super().__init__(model_name)
        self.device = _resolve_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"[LocalModel] Loading {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: list[dict], temperature: float = 0.7,
                 max_tokens: int = 1024) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
