

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infer import create_infer

PROMPT = "What is 2+2? Answer briefly."

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="api", choices=["api", "local"])
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model", default="default")
    parser.add_argument("--model-path")
    args = parser.parse_args()

    infer = create_infer(
        mode=args.mode,
        api_base=args.api_base,
        model=args.model,
        model_path=args.model_path,
    )
    response = infer.generate(PROMPT)
    print(f"Prompt: {PROMPT}")
    print(f"Response: {response}")
    infer.close()
