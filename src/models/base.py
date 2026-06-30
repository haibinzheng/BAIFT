"""src/models/base.py"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Abstract model interface for LLM inference"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.7,
                 max_tokens: int = 1024) -> str:
        """Generate a response for a list of messages"""
        ...