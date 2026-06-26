#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Run benchmark_serving.py against an already-running vLLM server for a
# prefill-heavy, a decode-heavy, and a single-prefill-step workload.
#
# Intended usage: start `vllm serve` yourself (e.g. with
# FUSE_ROPE_INTO_ATTN_KERNEL=true/false), then run this script once per
# server configuration with a different RUN_TAG so results don't collide.
#
# Usage:
#   MODEL=Qwen/Qwen3-4B RUN_TAG=rope_true  bash scripts/vllm/benchmarking/rope_fusion_sweep.sh
#   MODEL=Qwen/Qwen3-4B RUN_TAG=rope_false bash scripts/vllm/benchmarking/rope_fusion_sweep.sh
#
# Required environment variables:
#   MODEL  - HF model id or local path the server was started with
#
# Optional environment variables:
#   RUN_TAG         - label used in result/log filenames (default: "default")
#   MAX_CONCURRENCY (default: 16)
#   SEED            (default: 0)
#   RESULT_DIR      (default: /tmp/rope_fusion_sweep)
#   PORT            (default: 8000)

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL environment variable must be set (e.g. MODEL=Qwen/Qwen3-4B)." >&2
    exit 1
fi

RUN_TAG="${RUN_TAG:-default}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
SEED="${SEED:-0}"
RESULT_DIR="${RESULT_DIR:-/tmp/rope_fusion_sweep}"
PORT="${PORT:-8000}"

mkdir -p "$RESULT_DIR"

# Scenarios: name, random-input-len, random-output-len, num-prompts
# - prefill_heavy: large prompt, short generation -> dominated by RPAm (prefill)
# - decode_heavy:  short prompt, long generation  -> dominated by RPAd (decode)
# - single_prefill: matches the offline profiling setup (1 request, 2048-token prefill)
SCENARIOS=(
    "prefill_heavy 1000 1 1000"
    "decode_heavy 128 1024 16"
    "single_prefill 2048 1 1"
)

for scenario in "${SCENARIOS[@]}"; do
    read -r scenario_name input_len output_len num_prompts <<< "$scenario"

    max_concurrency="$MAX_CONCURRENCY"
    if [ "$scenario_name" = "single_prefill" ]; then
        max_concurrency=1
    fi

    tag="${scenario_name}_${RUN_TAG}"
    bench_log="${RESULT_DIR}/bench_${tag}.log"

    echo "=================================================================="
    echo "Running scenario=${scenario_name} run_tag=${RUN_TAG}"
    echo "  input_len=${input_len} output_len=${output_len} num_prompts=${num_prompts} max_concurrency=${max_concurrency}"
    echo "=================================================================="

    python3 scripts/vllm/benchmarking/benchmark_serving.py \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --seed "$SEED" \
        2>&1 | tee "$bench_log"
done

echo "=================================================================="
echo "All runs complete. Logs in: $RESULT_DIR"
echo "=================================================================="
