from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.schemas import DIMENSIONS


class LearnedScorer:
    """Score behavior with a fine-tuned Hugging Face sequence classifier.

    The checkpoint must expose exactly the five BAIFT dimensions through its
    ``id2label`` configuration. Multi-label checkpoints should set
    ``problem_type=multi_label_classification``; single-label logits are
    converted with softmax.
    """

    def __init__(self, model_name: str | None = None, max_length: int = 512):
        self.model_name = model_name or os.getenv("BAIFT_LEARNED_MODEL")
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None
        self._label_indices: dict[str, int] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_name:
            raise RuntimeError(
                "A fine-tuned learned scorer is required. Pass --learned-model "
                "or set BAIFT_LEARNED_MODEL to a BERT/RoBERTa checkpoint with "
                f"these labels: {', '.join(DIMENSIONS)}"
            )

        

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        id2label = self._model.config.id2label
        normalized = {str(label).strip().lower(): int(index) for index, label in id2label.items()}
        missing = [dimension for dimension in DIMENSIONS if dimension not in normalized]
        if missing or self._model.config.num_labels != len(DIMENSIONS):
            raise ValueError(
                f"Checkpoint {self.model_name!r} is not a five-dimension BAIFT scorer; "
                f"missing labels: {missing or 'none'}, num_labels={self._model.config.num_labels}"
            )
        self._label_indices = {dimension: normalized[dimension] for dimension in DIMENSIONS}

    def score(self, text: str) -> dict[str, float]:
        self._load()
        

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        with torch.inference_mode():
            logits = self._model(**encoded).logits[0]

        if self._model.config.problem_type == "multi_label_classification":
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)
        return {
            dimension: float(probabilities[index].item())
            for dimension, index in self._label_indices.items()
        }
