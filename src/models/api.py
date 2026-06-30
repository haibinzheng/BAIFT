"""src/models/api.py - OpenAI-compatible API inference (vLLM, etc.)"""
from __future__ import annotations

import requests
import json
from typing import Optional
from .base import BaseModel


class APIModel(BaseModel):
    """Call an OpenAI-compatible API endpoint"""

    def __init__(self, model_name: str, api_base: str,
                 api_key: str = "EMPTY", **kwargs):
        super().__init__(model_name)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        print(f"[APIModel] {model_name} @ {api_base}")

    def generate(self, messages: list[dict], temperature: float = 0.7,
                 max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate_stream(self, messages: list[dict], temperature: float = 0.7,
                        max_tokens: int = 1024):
        """Streaming generation"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with requests.post(f"{self.api_base}/chat/completions",
                           headers=self.headers,
                           json=payload, stream=True, timeout=120) as resp:
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        yield json.loads(data_str)["choices"][0]["delta"].get("content", "")
