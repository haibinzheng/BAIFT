#!/usr/bin/env python3
"""
Infer — send prompts to a model and collect responses.
Uses infer/ module (supports api / local modes).

Usage:
  python eval/scripts/infer.py --input cases.jsonl \
      --mode api --api-base http://localhost:8200/v1 --output responses.jsonl

  python eval/scripts/infer.py --input cases.jsonl \
      --mode local --model-path /path/to/model --output responses.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from infer import create_infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="api", choices=["api", "local"])
    parser.add_argument("--api-base", help="vLLM server URL")
    parser.add_argument("--model", default="default", help="Model name (for API)")
    parser.add_argument("--model-path", help="Model path (for local mode)")
    args = parser.parse_args()

    infer = create_infer(
        mode=args.mode,
        api_base=args.api_base,
        model=args.model,
        model_path=args.model_path,
    )

    with open(args.input) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    for i, case in enumerate(cases):
        for side in ["left", "right"]:
            case[side]["response"] = infer.generate(case[side]["prompt"])
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cases)}]", flush=True)

    infer.close()

    with open(args.output, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Done: {len(cases)} cases → {args.output}")


if __name__ == "__main__":
    main()
