"""src/metrics/iou_auc.py - IoU between methods and AUC computation"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


def fv_set_iou(fvs_a: list[float], fvs_b: list[float],
               tau: float, precision: float = 0.01) -> float:
    """
    IoU(A, B; tau) = |C_A ∩ C_B| / |C_A ∪ C_B|
    where C_X(tau) = {round(fv, 2) | fv ∈ method_X, fv > tau}
    """
    set_a = set(round(fv / precision) * precision for fv in fvs_a if fv > tau)
    set_b = set(round(fv / precision) * precision for fv in fvs_b if fv > tau)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(1, union)


def iou_curve(fvs_a: list[float], fvs_b: list[float],
              taus: list[float], precision: float = 0.01) -> list[float]:
    """IoU at each tau value"""
    return [fv_set_iou(fvs_a, fvs_b, tau, precision) for tau in taus]


def auc(ys: list[float], xs: list[float]) -> float:
    """Trapezoidal AUC"""
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += (ys[i] + ys[i - 1]) * dx / 2
    return area


def detection_rate_curve(fvs: list[float], taus: list[float]) -> list[float]:
    """Rate(tau) = |{fv > tau}| / total"""
    total = len(fvs)
    return [sum(1 for fv in fvs if fv > tau) / total for tau in taus]


def detection_rate_auc(fvs: list[float], taus: list[float]) -> float:
    """AUC of detection rate curve"""
    rates = detection_rate_curve(fvs, taus)
    return auc(rates, taus)


def method_pairwise_iou(all_fvs: dict[str, list[float]],
                        taus: list[float],
                        precision: float = 0.01) -> dict:
    """Compute IoU for all method pairs at all taus + AUC"""
    methods = list(all_fvs.keys())
    results = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            ious = iou_curve(all_fvs[a], all_fvs[b], taus, precision)
            results[f"{a}_vs_{b}"] = {
                "ious": ious,
                "auc": auc(ious, taus),
            }
    return results
