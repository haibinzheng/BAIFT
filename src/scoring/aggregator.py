from __future__ import annotations

from src.core.constants import DEFAULT_HYBRID_WEIGHTS
from src.core.schemas import BehaviorScore, DimensionScoreDetail, LLMResponseRecord
from src.scoring.learned import LearnedScorer
from src.scoring.llm_judge import LLMJudgeScorer
from src.scoring.rule_based import RuleBasedScorer


class HybridBehaviorScorer:
    def __init__(self, hybrid_weights: dict[str, dict[str, float]] | None = None,
                 judge_model=None, learned_model: str | None = None):
        self.hybrid_weights = hybrid_weights or DEFAULT_HYBRID_WEIGHTS
        self.rule_scorer = RuleBasedScorer()
        self.learned_scorer = LearnedScorer(model_name=learned_model)
        self.judge_scorer = LLMJudgeScorer(judge_model=judge_model)

    def score_response(self, prompt_text: str, response: LLMResponseRecord) -> BehaviorScore:
        rule_scores = self.rule_scorer.score(response.raw_text)
        learned_scores = self.learned_scorer.score(response.raw_text)
        judge_scores = self.judge_scorer.score(prompt_text, response.raw_text)

        final_scores: dict[str, float] = {}
        details: dict[str, DimensionScoreDetail] = {}
        for dimension, weights in self.hybrid_weights.items():
            score = (
                weights["rule"] * rule_scores[dimension]
                + weights["learned"] * learned_scores[dimension]
                + weights["judge"] * judge_scores[dimension]
            )
            final_scores[dimension] = score
            details[dimension] = DimensionScoreDetail(
                rule=rule_scores[dimension],
                learned=learned_scores[dimension],
                judge=judge_scores[dimension],
                final=score,
            )

        return BehaviorScore(
            response_id=response.response_id,
            helpfulness=final_scores["helpfulness"],
            informativeness=final_scores["informativeness"],
            supportiveness=final_scores["supportiveness"],
            respectfulness=final_scores["respectfulness"],
            refusal_strength=final_scores["refusal_strength"],
            scorer_details=details,
        )
