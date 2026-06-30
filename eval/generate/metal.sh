#!/bin/bash

OUTPUT_DIR="${1:-./data/cases}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/metal_cases.jsonl"
LOG_PATH="$LOG_DIR/metal_generate.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
python run.py pipeline --method metal --output "$RESULT_PATH" 2>&1 | tee "$LOG_PATH"
