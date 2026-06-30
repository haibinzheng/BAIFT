"""src/generators/metal.py - METAL formal dataset loader + inference"""
from __future__ import annotations

from .base import BaseGenerator
from .formal_datasets import load_formal_dataset_cases


class METALGenerator(BaseGenerator):
    def __init__(self, model=None, **kwargs):
        super().__init__("METAL", model, **kwargs)

    def generate(self, **kwargs) -> list[dict]:
        return load_formal_dataset_cases("metal")

    def infer(self, case: dict, **kwargs) -> dict:
        if self.model is None:
            return case
        for side in ["left", "right"]:
            msgs = [{"role": "user", "content": case[side]["prompt"]}]
            case[side]["response"] = self.model.generate(msgs)
        return case
