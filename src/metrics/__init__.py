"""src/metrics/__init__.py"""
from .behavior import BehaviorScorer
from .fv import compute_fv, calibrate_tau, calibrate_tau_mixed
from .disparity import (
    classify_pair, count_disparities, paper_metrics, per_category_metrics
)
from .iou_auc import (
    fv_set_iou, iou_curve, auc, detection_rate_curve,
    detection_rate_auc, method_pairwise_iou
)
from .lsa import LSAScorer


def create_lsa_scorer(model_name: str = "all-MiniLM-L6-v2"):
    """Create LSA scorer."""
    return LSAScorer(model_name)
