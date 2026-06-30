from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DIMENSIONS = [
    "helpfulness",
    "informativeness",
    "supportiveness",
    "respectfulness",
    "refusal_strength",
]


@dataclass
class TaskAnchor:
    anchor_id: str
    task_intent: str
    context: str
    response_constraint: str
    sensitive_slots: list[str]
    template: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Group:
    assignments: dict[str, str]

    def stable_name(self) -> str:
        return "|".join(f"{name}={self.assignments[name]}" for name in sorted(self.assignments))


@dataclass
class RenderedPrompt:
    prompt_id: str
    anchor_id: str
    group: Group
    attribute_subset: list[str]
    prompt_text: str


@dataclass
class LLMBackendConfig:
    base_url: str
    api_key_env: str
    api_key: str
    model: str
    request_path: str
    timeout_seconds: int
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    mock_on_failure: bool = True


@dataclass
class DatasetSourceConfig:
    name: str
    path: str
    role: str
    kind: str
    enabled: bool = True
    description: str = ""


@dataclass
class ExperimentModelConfig:
    name: str
    family: str
    release_date: str
    size: str
    open_source: bool
    architecture: str
    modality: str
    deployment: str
    vendor: str
    role: str = "target"
    endpoint: str = ""
    local_path: str = ""
    enabled: bool = True


@dataclass
class BaselineConfig:
    name: str
    paper: str
    rendering_model: str = ""
    scoring_model: str = ""
    finetune_model: str = ""
    enabled: bool = True


@dataclass
class EvaluationProtocolConfig:
    intents: list[str] = field(default_factory=list)
    metrics: dict[str, list[str]] = field(default_factory=dict)
    selected_attribute_pairs: list[list[str]] = field(default_factory=list)
    query_budget_per_model: int = 100000
    finetune_model: str = ""
    annotation_cases: int = 1600
    annotators: int = 40
    labels_per_case: int = 3
    pair_screening_count: int = 55
    pilot_models: list[str] = field(default_factory=list)
    calibration_rule: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class SearchConfig:
    max_order: int = 2
    enable_mutation: bool = True
    max_mutations_per_anchor: int = 3
    query_budget: int = 300
    top_k_per_iteration: int = 50
    selected_attribute_pairs: list[list[str]] = field(default_factory=list)
    lambda_fairness: float = 0.4
    lambda_amplification: float = 0.2
    lambda_coverage: float = 0.2
    lambda_novelty: float = 0.15
    lambda_penalty: float = 0.05

    def lambda_dict(self) -> dict[str, float]:
        return {
            "fairness": self.lambda_fairness,
            "amplification": self.lambda_amplification,
            "coverage": self.lambda_coverage,
            "novelty": self.lambda_novelty,
            "penalty": self.lambda_penalty,
        }


@dataclass
class CalibrationConfig:
    fairness_sigma_multiplier: float = 3.0
    semantic_similarity_threshold: float = 0.9
    calibration_sample_size: int = 200


@dataclass
class BAIFTConfig:
    anchors_path: str
    output_dir: str
    sensitive_values: dict[str, list[str]]
    dimension_weights: dict[str, float]
    hybrid_weights: dict[str, dict[str, float]]
    search: SearchConfig
    calibration: CalibrationConfig
    llm: LLMBackendConfig
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    experiment_models: list[ExperimentModelConfig] = field(default_factory=list)
    baselines: list[BaselineConfig] = field(default_factory=list)
    protocol: EvaluationProtocolConfig = field(default_factory=EvaluationProtocolConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BAIFTConfig:
        return cls(
            anchors_path=str(payload["anchors_path"]),
            output_dir=str(payload["output_dir"]),
            sensitive_values=dict(payload["sensitive_values"]),
            dimension_weights=dict(payload["dimension_weights"]),
            hybrid_weights=dict(payload["hybrid_weights"]),
            search=SearchConfig(**payload["search"]),
            calibration=CalibrationConfig(**payload["calibration"]),
            llm=LLMBackendConfig(**payload["llm"]),
            datasets=[DatasetSourceConfig(**item) for item in payload.get("datasets", [])],
            experiment_models=[ExperimentModelConfig(**item) for item in payload.get("experiment_models", [])],
            baselines=[BaselineConfig(**item) for item in payload.get("baselines", [])],
            protocol=EvaluationProtocolConfig(**payload.get("protocol", {})),
        )


@dataclass
class LLMResponseRecord:
    response_id: str
    prompt_id: str
    model_name: str
    raw_text: str
    latency_ms: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    success: bool = True
    error_msg: str = ""


@dataclass
class DimensionScoreDetail:
    rule: float
    learned: float
    judge: float
    final: float


@dataclass
class BehaviorScore:
    response_id: str
    helpfulness: float
    informativeness: float
    supportiveness: float
    respectfulness: float
    refusal_strength: float
    scorer_details: dict[str, DimensionScoreDetail] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "helpfulness": self.helpfulness,
            "informativeness": self.informativeness,
            "supportiveness": self.supportiveness,
            "respectfulness": self.respectfulness,
            "refusal_strength": self.refusal_strength,
        }


@dataclass
class FairnessCaseResult:
    anchor_id: str
    attribute_subset: list[str]
    groups: list[Group]
    fairness_score: float
    semantic_similarity: float
    discrimination_type: str
    amplification_score: float | None
    disparity_vector: dict[str, float]
    pairwise_disparities: dict[str, dict[str, float]]
    diagnosis_labels: list[str]
    behavior_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    prompt_texts: list[str] = field(default_factory=list)
    response_texts: list[str] = field(default_factory=list)
    coverage_gain: float = 0.0
    novelty_score: float = 0.0
    invalidity_penalty: float = 0.0
    objective_score: float = 0.0
