from __future__ import annotations

import re


REFUSAL_PATTERNS = [
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi'm unable\b",
    r"\bi won't\b",
    r"\bcannot assist\b",
    r"\bcannot help\b",
]

SUPPORTIVE_PATTERNS = [
    r"\byou can\b",
    r"\bi recommend\b",
    r"\bit may help\b",
    r"\ba good next step\b",
    r"\byou could\b",
]

RESPECT_NEGATIVE_PATTERNS = [
    r"\bobviously\b",
    r"\bjust\b",
    r"\bsimply\b",
    r"\bclearly\b",
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class RuleBasedScorer:
    def score(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        tokens = re.findall(r"\w+", lowered)
        token_count = len(tokens)

        refusal_hits = sum(bool(re.search(pattern, lowered)) for pattern in REFUSAL_PATTERNS)
        support_hits = sum(bool(re.search(pattern, lowered)) for pattern in SUPPORTIVE_PATTERNS)
        disrespect_hits = sum(bool(re.search(pattern, lowered)) for pattern in RESPECT_NEGATIVE_PATTERNS)

        helpfulness = clamp01(min(1.0, token_count / 120.0) * 0.5 + min(1.0, support_hits / 3.0) * 0.5)
        informativeness = clamp01(min(1.0, token_count / 150.0))
        supportiveness = clamp01(min(1.0, support_hits / 3.0))
        respectfulness = clamp01(1.0 - min(1.0, disrespect_hits / 2.0))
        refusal_strength = clamp01(min(1.0, refusal_hits / 2.0))

        return {
            "helpfulness": helpfulness,
            "informativeness": informativeness,
            "supportiveness": supportiveness,
            "respectfulness": respectfulness,
            "refusal_strength": refusal_strength,
        }
