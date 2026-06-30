#!/bin/bash

OUTPUT_DIR="${1:-./data/cases}"
API_BASE="${API_BASE:-}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RESULT_PATH="$OUTPUT_DIR/caffe_cases.jsonl"
LOG_PATH="$LOG_DIR/caffe_generate.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ARGS="--method caffe --output $RESULT_PATH"
[ -n "$API_BASE" ] && ARGS="$ARGS --api-base $API_BASE --model default"

python run.py pipeline $ARGS 2>&1 | tee "$LOG_PATH"
