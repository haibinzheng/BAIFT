#!/bin/bash

set -e

MODEL="${1:?Usage: bash eval/run_vllm.sh MODEL_PATH [PORT] [GPU] [MEM]}"
PORT="${2:-8200}"
GPU="${3:-0}"
MEM="${4:-40}"
MODEL_NAME=$(basename "$MODEL")
LOG_DIR="${LOG_DIR:-./logs}"
LOG_PATH="$LOG_DIR/vllm_${MODEL_NAME}_port${PORT}.log"

mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    2>&1 | tee "$LOG_PATH"
