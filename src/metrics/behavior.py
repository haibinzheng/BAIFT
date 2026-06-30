"""src/metrics/behavior.py - Three-scorer behavior scoring"""
from __future__ import annotations

from src.scoring.aggregator import HybridBehaviorScorer
from src.core.schemas import LLMResponseRecord
from src.core.constants import DEFAULT_DIMENSION_WEIGHTS


class BehaviorScorer:
    

    def __init__(self, judge_model=None, learned_model: str | None = None):
        self.scorer = HybridBehaviorScorer(
            judge_model=judge_model,
            learned_model=learned_model,
        )

    def score(self, prompt: str, response: str) -> dict[str, float]:
        
        record = LLMResponseRecord(
            response_id="tmp",
            prompt_id="tmp",
            model_name="default",
            raw_text=response,
        )
        result = self.scorer.score_response(prompt_text=prompt, response=record)
        return {
            "helpfulness": result.helpfulness,
            "informativeness": result.informativeness,
            "supportiveness": result.supportiveness,
            "respectfulness": result.respectfulness,
            "refusal_strength": result.refusal_strength,
        }

    def disparity(self, left_prompt: str, left_response: str,
                  right_prompt: str, right_response: str) -> dict[str, float]:
        
        left = self.score(left_prompt, left_response)
        right = self.score(right_prompt, right_response)
        return {dim: abs(left[dim] - right[dim]) for dim in DEFAULT_DIMENSION_WEIGHTS}

    def fv(self, left_prompt: str, left_response: str,
           right_prompt: str, right_response: str) -> float:
        
        disp = self.disparity(left_prompt, left_response, right_prompt, right_response)
        return sum(DEFAULT_DIMENSION_WEIGHTS[d] * disp[d] for d in DEFAULT_DIMENSION_WEIGHTS)
