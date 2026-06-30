#!/usr/bin/env python3
"""
BAIFT — Generate bias test cases


"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer import create_infer
from src.generators import create_generator, GENERATORS
from src.utils import save_jsonl


def main():
    parser = argparse.ArgumentParser(description="BAIFT — Generate bias test cases")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", help="Generate bias test cases")
    p.add_argument("--method", required=True, choices=list(GENERATORS.keys()),
                   help="Method: baift / caffe / metal / llm_roleplay")
    p.add_argument("--num", type=int, default=100,
                   help="Number of pairs to generate for baseline methods")
    p.add_argument("--output", default="./data/cases.jsonl",
                   help="Output JSONL path")
    p.add_argument("--api-base",
                   help="API base URL (optional: use LLM for prompt generation)")
    p.add_argument("--model", default="default",
                   help="Model name for LLM generation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for case generation")

    args = parser.parse_args()

    model = None
    if args.api_base:
        model = create_infer(mode="api", api_base=args.api_base, model=args.model)

    gen = create_generator(args.method, model=model)
    cases = gen.generate(num_pairs=args.num, seed=args.seed)
    save_jsonl(cases, args.output)
    print(f"Generated {len(cases)} cases → {args.output}")


if __name__ == "__main__":
    main()
