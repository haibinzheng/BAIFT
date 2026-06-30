#!/bin/bash

INPUT="${1:-./data/responses.jsonl}"
OUTPUT_DIR="${2:-./data/results}"
TAU="${TAU:-0.3626}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/llm_roleplay_lsa_scores.jsonl"
LOG_PATH="$LOG_DIR/llm_roleplay_score.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
python eval/scripts/score.py --input "$INPUT" --tau "$TAU" --output "$RESULT_PATH" 2>&1 | tee "$LOG_PATH"
