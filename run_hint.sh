#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
DATA_FILE="${DATA_FILE:-./data/dapo-selected-10k.jsonl}"
SAVE_PATH="${SAVE_PATH:-./outputs/hint-qwen2.5-7b}"
REF_PORT="${REF_PORT:-59888}"
REF_DEVICE="${REF_DEVICE:-0}"
GEN_DEVICES="${GEN_DEVICES:-1}"

CUDA_VISIBLE_DEVICES="${REF_DEVICE}" python ref_server.py \
  --model-path "${MODEL_PATH}" \
  --port "${REF_PORT}" &
REF_PID=$!

cleanup() {
  kill "${REF_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${GEN_DEVICES}" python rl_hint.py train \
  --model-path "${MODEL_PATH}" \
  --data-file "${DATA_FILE}" \
  --ref-url "http://127.0.0.1:${REF_PORT}" \
  --save-path "${SAVE_PATH}" \
  --gen-devices "${GEN_DEVICES}"
