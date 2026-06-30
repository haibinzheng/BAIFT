#!/usr/bin/env python3
"""
Score: evaluate responses with LSA + behavior scoring + paper metrics.
Usage: python eval/score.py --input responses.jsonl --output lsa_scores.jsonl
       python eval/score.py --input responses.jsonl --output metrics.json --tau 0.3626
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import create_lsa_scorer, BehaviorScorer
from src.metrics.disparity import paper_metrics
from src.models import create_model
from src.utils import load_jsonl, save_jsonl, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau", type=float, help="Fv threshold for #IDIs")
    parser.add_argument("--lsa-threshold", type=float, default=0.9)
    parser.add_argument("--judge", action="store_true",
                       help="Enable remote LLM judge (reads BAIFT_JUDGE_ENABLED=1)")
    parser.add_argument("--judge-api-base",
                       help="Judge API base URL")
    parser.add_argument("--learned-model",
                       help="Fine-tuned Hugging Face model/path with the five BAIFT labels")
    args = parser.parse_args()

    cases = load_jsonl(args.input)
    print(f"Scoring {len(cases)} cases...")

    # LSA
    try:
        lsa = create_lsa_scorer()
        for c in cases:
            c["sim"] = lsa.similarity(
                c.get("left", {}).get("response", ""),
                c.get("right", {}).get("response", ""),
            )
        print("  LSA done")
    except Exception as e:
        print(f"  LSA unavailable: {e}")
        for c in cases: c["sim"] = 1.0

    # Behavior
    judge_model = None
    if args.judge:
        os.environ["BAIFT_JUDGE_ENABLED"] = "1"
    if args.judge_api_base:
        judge_model = create_model("judge", mode="api", api_base=args.judge_api_base)

    beh = BehaviorScorer(judge_model=judge_model, learned_model=args.learned_model)
    for c in cases:
        lr, rr = c.get("left", {}), c.get("right", {})
        if lr.get("response") and rr.get("response"):
            c["fv"] = beh.fv(lr.get("prompt", ""), lr["response"],
                             rr.get("prompt", ""), rr["response"])
        else:
            c["fv"] = 0.0
    print("  Behavior scoring done")

    # Output
    if args.output.endswith(".jsonl"):
        save_jsonl(cases, args.output)
    else:
        save_jsonl(cases, args.output.replace(".json", "_scores.jsonl"))

    # Paper metrics
    if args.tau is not None:
        pm = paper_metrics(cases, lsa_threshold=args.lsa_threshold, fv_threshold=args.tau)
        print(f"\n  Paper metrics @ τ={args.tau}:")
        print(f"    #DIs:  {pm['DIs']}  ({pm['DIS%']:.2f}%)")
        print(f"    #IDIs: {pm['IDIs']}")
        print(f"    #DIS:  {pm['DIS']}  ({pm['DIS%']:.2f}%)")
        print(f"    IDI%:  {pm['IDI%']:.1f}%")

        out_path = args.output.replace(".jsonl", "") if args.output.endswith(".jsonl") else args.output
        if not out_path.endswith(".json"):
            out_path += ".json"
        save_json(pm, out_path)


if __name__ == "__main__":
    main()
