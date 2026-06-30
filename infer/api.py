"""infer/api.py - Inference via OpenAI-compatible API (vLLM server)"""
from __future__ import annotations

import requests
import json


class APIModelInfer:
   

    def __init__(self, api_base: str, model: str = "default", **kwargs):
        self.api_base = api_base.rstrip("/")
        self.model = model
        print(f"[Infer API] {self.model} @ {self.api_base}")

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate_messages(self, messages: list[dict], temperature: float = 0.7,
                          max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def close(self):
        pass
