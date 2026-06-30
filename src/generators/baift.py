"""src/generators/baift.py - BAIFT formal dataset loader + inference"""
from __future__ import annotations

from .base import BaseGenerator
from .formal_datasets import load_formal_dataset_cases


class BAIFTGenerator(BaseGenerator):
    def __init__(self, model=None, **kwargs):
        super().__init__("BAIFT", model, **kwargs)

    def generate(self, **kwargs) -> list[dict]:
        return load_formal_dataset_cases("baift")

    def infer(self, case: dict, **kwargs) -> dict:
        if self.model is None:
            return case
        for side in ["left", "right"]:
            msgs = [{"role": "user", "content": case[side]["prompt"]}]
            case[side]["response"] = self.model.generate(msgs)
        return case
