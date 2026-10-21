"""Offline structural checks; no model calls and no printing of example text."""
from pathlib import Path
import argparse
import ast
import json
import os
import re
from prepare_pairs import expand_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    for rel in ["README.md", "LICENSE", "CITATION.cff", "requirements.txt", "run.py", "src/scoring/learned.py", "src/scoring/llm_judge.py", "src/metrics/disparity.py", "eval/scripts/infer.py", "eval/scripts/score.py", "Vis/framework.png", "docs/ARTIFACT.md"]:
        if not (root / rel).is_file():
            errors.append("Missing file: " + rel)
    sources = [p for p in root.rglob("*.py") if not any(s in {".git", ".venv", "venv"} for s in p.relative_to(root).parts)]
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Invalid Python: {path.relative_to(root)} ({type(exc).__name__})")
    print(f"Python syntax: {len(sources)} files")
    for method in ["baift", "caffe", "metal", "llm_roleplay"]:
        for phase, filename in [("generate", "samples.jsonl"), ("eval", "responses.jsonl")]:
            path = root / "data" / phase / method / "llama-3.1-8b" / filename
            if not path.is_file():
                errors.append("Missing example: " + str(path.relative_to(root)))
                continue
            count = 0
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    packed = json.loads(line)
                    for item in expand_record(packed):
                        for side in ["left", "right"]:
                            assert isinstance(item[side]["prompt"], str)
                            if phase == "eval":
                                assert isinstance(item[side]["response"], str)
                except (ValueError, KeyError, TypeError, AssertionError):
                    errors.append(f"Invalid schema: {path.relative_to(root)} line {line_no}")
            if not count:
                errors.append("Empty example: " + str(path.relative_to(root)))
            print(f"Example {phase}/{method}: {count} records (content omitted)")
    for path in [root / "README.md", *(root / "docs").glob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                errors.append(f"Broken link: {path.relative_to(root)} -> {target}")
    missing = [f"data/formal_datasets/{m}.jsonl" for m in ["baift", "caffe", "metal", "llm_roleplay"] if not (root / f"data/formal_datasets/{m}.jsonl").is_file()]
    if not os.getenv("BAIFT_LEARNED_MODEL"):
        missing.append("BAIFT_LEARNED_MODEL (trained checkpoint; validity not checked)")
    print("Missing reproduction inputs: " + (", ".join(missing) or "none detected; full reproduction still requires validation"))
    for error in errors:
        print("ERROR: " + error)
    print("Offline structural check: " + ("FAIL" if errors else "PASS"))
    print("Not a paper-result reproduction or artifact certification.")
    return int(bool(errors or (args.require_full and missing)))


if __name__ == "__main__":
    raise SystemExit(main())
