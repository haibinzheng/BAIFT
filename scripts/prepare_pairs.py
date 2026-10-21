"""Convert supplied packed examples to the flat paired inference/scoring schema."""
from pathlib import Path
import argparse
import json


def expand_record(record):
    if "left" in record and "right" in record:
        for side in ("left", "right"):
            if not isinstance(record[side].get("prompt"), str):
                raise ValueError("Flat case requires string prompts")
        return [record]
    prompts = record["prompts"]
    result = []
    for pair in record["pairs"]:
        sides = []
        for key in ("left", "right"):
            index = pair[key]
            if type(index) is not int or not 0 <= index < len(prompts):
                raise ValueError("Pair index is outside the prompt list")
            source = prompts[index]
            if not isinstance(source["text"], str):
                raise ValueError("Packed prompt text must be a string")
            side = {"prompt": source["text"]}
            for field in ("response", "group"):
                if field in source:
                    side[field] = source[field]
            sides.append(side)
        result.append({"case_id": pair.get("id", record.get("id")), "method": record.get("method"), "left": sides[0], "right": sides[1], "mode": pair.get("mode"), "meta": record.get("meta", {})})
    if not result:
        raise ValueError("Packed example contains no pairs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output path; input and existing outputs are preserved")
    output = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.extend(expand_record(json.loads(line)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in output), encoding="utf-8")
    print(f"Prepared {len(output)} pairs; text content omitted")


if __name__ == "__main__":
    main()
