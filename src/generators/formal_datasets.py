from __future__ import annotations

from pathlib import Path

from src.utils import load_jsonl


FORMAL_DATASETS = {
    "baift": Path("data/formal_datasets/baift.jsonl"),
    "caffe": Path("data/formal_datasets/caffe.jsonl"),
    "metal": Path("data/formal_datasets/metal.jsonl"),
    "llm_roleplay": Path("data/formal_datasets/llm_roleplay.jsonl"),
}


def load_formal_dataset_cases(method: str) -> list[dict]:
    path = FORMAL_DATASETS[method]
    rows = load_jsonl(path)
    if method == "metal":
        return _load_metal(rows)
    return _load_pair_records(rows)


def _load_pair_records(rows: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for row in rows:
        prompts = row.get("prompts", [])
        pairs = row.get("pairs", [])
        if not pairs and len(prompts) >= 2:
            pairs = [{"id": row.get("id"), "left": 0, "right": 1, "mode": row.get("mode", "pair")}]
        for pair in pairs:
            left = prompts[pair["left"]]
            right = prompts[pair["right"]]
            case_id = pair.get("id") or row.get("id")
            cases.append(
                {
                    "case_id": case_id,
                    "left": {"prompt": left["text"]},
                    "right": {"prompt": right["text"]},
                    "mode": pair.get("mode", left.get("mode", "default")),
                    "meta": dict(row.get("meta", {})),
                }
            )
    return cases


def _load_metal(rows: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for row in rows:
        for block in row.get("intents", []):
            pair = block["pair"]
            cases.append(
                {
                    "case_id": f"{row['id']}::{block['intent']}",
                    "left": {"prompt": pair[0]["prompt"]},
                    "right": {"prompt": pair[1]["prompt"]},
                    "mode": row.get("perturbation", "identity_prefix"),
                    "meta": {
                        "bias_type": row.get("bias_type"),
                        "topic": row.get("topic"),
                        "intent": block.get("intent"),
                        "groups": row.get("groups", []),
                        "triple": row.get("triple"),
                    },
                }
            )
    return cases
