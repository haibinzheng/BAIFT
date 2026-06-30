"""src/metrics/disparity.py - DIS/IDIS counting + paper metrics"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.metrics.behavior import BehaviorScorer
from src.metrics.fv import compute_fv


def classify_pair(sim: float, fv_val: float,
                  lsa_threshold: float = 0.9,
                  fv_threshold: Optional[float] = None) -> str:
    
    if sim < lsa_threshold:
        return "edis"
    if fv_threshold is not None and fv_val > fv_threshold:
        return "idis"
    return "none"


def count_disparities(items: list[dict],
                      lsa_field: str = "sim",
                      fv_field: str = "fv",
                      lsa_threshold: float = 0.9,
                      fv_threshold: Optional[float] = None) -> dict:
    ""
    results = {"total": len(items), "edis": 0, "idis": 0, "dis": 0}
    for item in items:
        sim = item.get(lsa_field, 0)
        fv_val = item.get(fv_field, 0)
        cls = classify_pair(sim, fv_val, lsa_threshold, fv_threshold)
        if cls == "edis":
            results["edis"] += 1
        elif cls == "idis":
            results["idis"] += 1
    results["dis"] = results["edis"] + results["idis"]
    return results


def paper_metrics(items: list[dict],
                  lsa_threshold: float = 0.9,
                  fv_threshold: Optional[float] = None) -> dict:
    
    d = count_disparities(items, lsa_threshold=lsa_threshold, fv_threshold=fv_threshold)
    total = d["total"]
    return {
        "total": total,
        "DIs": d["edis"],
        "IDIs": d["idis"],
        "DIS": d["dis"],
        "DIS%": d["dis"] / total * 100 if total else 0,
        "IDI%": d["idis"] / d["dis"] * 100 if d["dis"] else 0,
    }


def per_category_metrics(items: list[dict],
                         category_field: str = "mode",
                         lsa_threshold: float = 0.9,
                         fv_threshold: Optional[float] = None) -> dict:
    
    grouped = defaultdict(list)
    for item in items:
        cat = item.get(category_field, "unknown")
        grouped[cat].append(item)
    return {
        cat: paper_metrics(grouped[cat], lsa_threshold, fv_threshold)
        for cat in sorted(grouped.keys())
    }
