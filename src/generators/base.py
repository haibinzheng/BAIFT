"""src/generators/base.py"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseGenerator(ABC):
    """Abstract base for test case generation."""

    def __init__(self, method: str, model, **kwargs):
        self.method = method
        self.model = model  # LLM for generation

    @abstractmethod
    def generate(self, **kwargs) -> list[dict]:
        """Generate test cases. Each case must have:
        - case_id: unique identifier
        - left: {"prompt": str, "response": Optional[str]}
        - right: {"prompt": str, "response": Optional[str]}
        - mode: bias type (e.g., "race|religion")
        - meta: additional info (intent, etc.)
        """
        ...

    @abstractmethod
    def infer(self, case: dict, **kwargs) -> dict:
        """Run inference on a test case, filling in left/right responses"""
        ...
