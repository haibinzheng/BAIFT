#!/bin/bash

MODEL="deepseek-chat"
API_BASE="${API_BASE:-}"
INPUT="${1:-./data/responses.jsonl}"
OUTPUT_DIR="${2:-./data/responses}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/deepseek-v3.2_responses.jsonl"
LOG_PATH="$LOG_DIR/deepseek-v3.2_infer.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
python eval/scripts/infer.py --input "$INPUT" --mode api --api-base "$API_BASE" --model "$MODEL" --output "$RESULT_PATH" 2>&1 | tee "$LOG_PATH"
