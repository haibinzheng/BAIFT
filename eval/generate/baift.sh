#!/bin/bash

OUTPUT_DIR="${1:-./data/cases}"
API_BASE="${API_BASE:-}"
MODEL="${MODEL:-default}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/baift_cases.jsonl"
LOG_PATH="$LOG_DIR/baift_generate.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ARGS="--method baift --output $RESULT_PATH"
[ -n "$API_BASE" ] && ARGS="$ARGS --api-base $API_BASE --model $MODEL"

python run.py pipeline $ARGS 2>&1 | tee "$LOG_PATH"
