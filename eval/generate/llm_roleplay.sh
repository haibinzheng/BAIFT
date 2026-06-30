#!/bin/bash

OUTPUT_DIR="${1:-./data/cases}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/llm_roleplay_cases.jsonl"
LOG_PATH="$LOG_DIR/llm_roleplay_generate.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
python run.py pipeline --method llm_roleplay --output "$RESULT_PATH" 2>&1 | tee "$LOG_PATH"
