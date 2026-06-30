"""src/utils.py - Common utilities"""
from __future__ import annotations

import json
import os


def load_jsonl(path: str) -> list[dict]:
  
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
  
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data, path: str):
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:

    with open(path) as f:
        return json.load(f)


def extract_fv(items: list[dict], fv_field: str = "fv") -> list[float]:
   
    return [item.get(fv_field, 0) for item in items]
