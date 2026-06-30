from __future__ import annotations

import re
from collections import Counter
from math import sqrt


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_similarity_bow(left: str, right: str) -> float:
    left_counts = Counter(_tokenize(left))
    right_counts = Counter(_tokenize(right))
    keys = set(left_counts) | set(right_counts)
    dot = sum(left_counts[key] * right_counts[key] for key in keys)
    left_norm = sqrt(sum(value * value for value in left_counts.values()))
    right_norm = sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
