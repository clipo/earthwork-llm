#!/bin/bash
# Serve the EarthworkLLM reasoning model (V9.1) with vLLM.
#
# Reported in the manuscript as the deployed checkpoint: Qwen3-VL-30B-A3B-Thinking
# fine-tuned with QLoRA, V9.1 checkpoint-1250 (the best-balanced checkpoint that
# passes every benchmark category threshold). The adapter targets attention
# projections + the MoE router gate, so it serves as a LoRA module — no merge.
#
# NOTE: run this on the GPU host (DGX Spark / CUDA). It cannot run on a CPU-only
# box. Verify once on hardware after any path change.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
# V9.1 selected checkpoint (see docs/RESEARCH_PAPER.md Sec. 2.5.3).
ADAPTER="${ADAPTER:-checkpoints/v91_lf_rebalance/checkpoint-1250}"
ALIAS="${ALIAS:-terrallm-v91}"       # the --model name inference scripts pass
PORT="${PORT:-8000}"

if [ ! -d "$ADAPTER" ]; then
    echo "Error: adapter directory not found at $ADAPTER" >&2
    echo "Point ADAPTER= at the V9.1 checkpoint-1250 directory (adapter_config.json + adapter_model.safetensors)." >&2
    exit 1
fi

echo "Serving $BASE_MODEL + LoRA adapter $ADAPTER as '$ALIAS' on port $PORT ..."

# Requires vllm (Linux/CUDA). --max-lora-rank 64 matches the adapter (r=64).
# --max-model-len gives room for the 6-panel image tokens plus the Thinking
# model's reasoning trace.
vllm serve "$BASE_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enable-lora --max-lora-rank 64 \
    --lora-modules "${ALIAS}=${ADAPTER}" \
    --dtype bfloat16 \
    --port "$PORT"

# Query it with, e.g.:
#   python scripts/earthwork_query.py --query "Find mounds" --lidar tile.las \
#     --api-url http://localhost:8000/v1/chat/completions --model terrallm-v91
