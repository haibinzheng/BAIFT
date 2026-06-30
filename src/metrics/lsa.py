from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class LSAScorer:
    """Semantic similarity scorer using sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[LSA] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts"""
        emb = self.model.encode([text_a, text_b])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def batch_similarity(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch compute similarities for multiple pairs"""
        all_texts = [t for pair in pairs for t in pair]
        embs = self.model.encode(all_texts)
        results = []
        for i in range(0, len(embs), 2):
            sim = float(cosine_similarity([embs[i]], [embs[i + 1]])[0][0])
            results.append(sim)
        return results
