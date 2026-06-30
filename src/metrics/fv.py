
from __future__ import annotations

import random
import math
from statistics import mean, stdev
from typing import Optional

from src.metrics.behavior import BehaviorScorer


def compute_fv(scorer: BehaviorScorer,
               left_prompt: str, left_response: str,
               right_prompt: str, right_response: str) -> float:
    
    return scorer.fv(left_prompt, left_response, right_prompt, right_response)


def calibrate_tau(fv_values: list[float], sigma_mult: float = 3.0) -> float:
    
    mu = mean(fv_values)
    sigma = stdev(fv_values)
    tau = mu + sigma_mult * sigma
    return tau


def calibrate_tau_mixed(all_fv_dict: dict[str, list[float]],
                        sigma_mult: float = 3.0, seed: int = 42) -> float:
    
    rng = random.Random(seed)
    min_n = min(len(v) for v in all_fv_dict.values())
    mixed = []
    for method, fvs in all_fv_dict.items():
        mixed.extend(rng.sample(fvs, min_n))
    mu = mean(mixed)
    sigma = stdev(mixed)
    return mu + sigma_mult * sigma
