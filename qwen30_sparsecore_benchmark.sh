#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 <variant> [trials]" \
    "" \
    "Variants:" \
    "  basic              Original MoE preprocessing + original SparseCore settings" \
    "  sparse_core_only   Original MoE preprocessing + improved SparseCore settings" \
    "  moe_gather_only    Simplified MoE preprocessing + original SparseCore settings" \
    "  improvement_all    Simplified MoE preprocessing + improved SparseCore settings" \
    "" \
    "Examples:" \
    "  $0 basic 3" \
    "  $0 sparse_core_only 3" \
    "  $0 moe_gather_only 3" \
    "  $0 improvement_all 3" \
    "" \
    "The script starts a fresh server, waits for /health, runs every trial, and" \
    "then stops only the server process group that it created."
}

VARIANT="${1:-}"
TRIALS="${2:-3}"

case "$VARIANT" in
  basic|sparse_core_only|moe_gather_only|improvement_all) ;;
  -h|--help|"")
    usage
    exit 0
    ;;
  *)
    printf 'ERROR: unknown variant: %s\n\n' "$VARIANT" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: trials must be a positive integer; received %s\n' "$TRIALS" >&2
  exit 2
fi

# Override these paths or workload values through environment variables when needed.
TPU_INFERENCE_REPO="${TPU_INFERENCE_REPO:-$HOME/tpu-inference}"
INFERENCEX_REPO="${INFERENCEX_REPO:-$HOME/InferenceX}"
BENCH_CLIENT="${BENCH_CLIENT:-$INFERENCEX_REPO/utils/bench_serving/benchmark_serving.py}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC="${CONC:-512}"
DP_SIZE="${DP_SIZE:-8}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-5120}"
NUM_WARMUPS="${NUM_WARMUPS:-1024}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/qwen3-30b-sparsecore-benchmark}"
RESULT_DIR="$RESULT_ROOT/$EXPERIMENT_ID/$VARIANT"
BASE_URL="http://127.0.0.1:$PORT"

MAX_MODEL_LEN=$((ISL + OSL + 20))
MAX_NUM_BATCHED_TOKENS=$((ISL / DP_SIZE))
(( MAX_NUM_BATCHED_TOKENS < 1024 )) && MAX_NUM_BATCHED_TOKENS=1024
(( MAX_NUM_BATCHED_TOKENS > 2048 )) && MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=$((CONC * 2 / DP_SIZE))
(( MAX_NUM_SEQS < 1 )) && MAX_NUM_SEQS=1

# Common environment: identical for every variant.
export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export USE_MOE_EP_KERNEL=0
export DP_SCHED_BATCH_PREFILL=1
export MOE_ROUTE_PADDING_TO_EXPERT0=0
export MIN_TOKEN_BUCKET=16
export ATTN_BUCKETIZED_NUM_REQS=0
export ONEHOT_MOE_PERMUTE_THRESHOLD=0
export VLLM_MOE_CHUNK_SIZE=0
export SLICE_ROPE_CACHE=0
export MOE_LOG_RAGGED_GATHER_V2_STATS=0
export MOE_LOG_COMBINE_VALID_ROWS_STATS=0

# Variant matrix. Every relevant flag is assigned explicitly so values cannot leak
# from a previously run server configuration.
case "$VARIANT" in
  basic)
    export MOE_TOKEN_INDICES_USE_GATHER=1
    export MOE_GROUP_SIZES_USE_ONEHOT=1
    export MOE_VALID_ROWS_MASK_USE_GATHER=1
    export RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE=0
    export RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS=0
    ;;
  sparse_core_only)
    export MOE_TOKEN_INDICES_USE_GATHER=1
    export MOE_GROUP_SIZES_USE_ONEHOT=1
    export MOE_VALID_ROWS_MASK_USE_GATHER=1
    export RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE=2048
    export RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS=8
    export RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS=1
    ;;
  moe_gather_only)
    export MOE_TOKEN_INDICES_USE_GATHER=0
    export MOE_GROUP_SIZES_USE_ONEHOT=0
    export MOE_VALID_ROWS_MASK_USE_GATHER=0
    export RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE=0
    export RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS=0
    ;;
  improvement_all)
    export MOE_TOKEN_INDICES_USE_GATHER=0
    export MOE_GROUP_SIZES_USE_ONEHOT=0
    export MOE_VALID_ROWS_MASK_USE_GATHER=0
    export RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE=2048
    export RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS=8
    export RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS=4
    export RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS=1
    ;;
esac

print_configuration() {
  printf '%s\n' \
    "variant=$VARIANT" \
    "model=$MODEL" \
    "isl=$ISL" \
    "osl=$OSL" \
    "concurrency=$CONC" \
    "dp_size=$DP_SIZE" \
    "max_model_len=$MAX_MODEL_LEN" \
    "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    "max_num_seqs=$MAX_NUM_SEQS" \
    "num_prompts=$NUM_PROMPTS" \
    "num_warmups=$NUM_WARMUPS" \
    "request_rate=inf" \
    "random_range_ratio=1.0" \
    "MOE_TOKEN_INDICES_USE_GATHER=$MOE_TOKEN_INDICES_USE_GATHER" \
    "MOE_GROUP_SIZES_USE_ONEHOT=$MOE_GROUP_SIZES_USE_ONEHOT" \
    "MOE_VALID_ROWS_MASK_USE_GATHER=$MOE_VALID_ROWS_MASK_USE_GATHER" \
    "RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE=$RAGGED_GATHER_V2_DENSE_FALLBACK_MAX_OUT_SIZE" \
    "RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS=$RAGGED_GATHER_V2_MAX_NUM_ROW_SUBCHUNKS" \
    "RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS=$RAGGED_GATHER_REDUCE_V2_MAX_NUM_ROW_SUBCHUNKS" \
    "RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS=$RAGGED_GATHER_REDUCE_V2_INTERLEAVE_ROW_PARTITIONS"
}

if [[ "${PRINT_CONFIG_ONLY:-0}" == 1 ]]; then
  print_configuration
  exit 0
fi

for required_command in vllm python3 curl setsid git; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    printf 'ERROR: required command is unavailable: %s\n' "$required_command" >&2
    exit 1
  fi
done

if [[ ! -d "$TPU_INFERENCE_REPO/.git" ]]; then
  printf 'ERROR: TPU_INFERENCE_REPO is not a Git checkout: %s\n' "$TPU_INFERENCE_REPO" >&2
  exit 1
fi

CURRENT_BRANCH="$(git -C "$TPU_INFERENCE_REPO" branch --show-current)"
if [[ "$CURRENT_BRANCH" != sparsecore-moe-investigation ]]; then
  printf 'ERROR: expected branch sparsecore-moe-investigation, found %s\n' "$CURRENT_BRANCH" >&2
  exit 1
fi

if [[ ! -f "$BENCH_CLIENT" ]]; then
  printf 'ERROR: benchmark client not found: %s\n' "$BENCH_CLIENT" >&2
  exit 1
fi

if curl --silent --fail "$BASE_URL/health" >/dev/null 2>&1; then
  printf 'ERROR: a server is already healthy at %s. Stop it before running this script.\n' "$BASE_URL" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
CONFIG_FILE="$RESULT_DIR/config.txt"
SERVER_LOG="$RESULT_DIR/server.log"

{
  print_configuration
  printf 'tpu_inference_commit=%s\n' "$(git -C "$TPU_INFERENCE_REPO" rev-parse HEAD)"
  if [[ -d "$INFERENCEX_REPO/.git" ]]; then
    printf 'inferencex_commit=%s\n' "$(git -C "$INFERENCEX_REPO" rev-parse HEAD)"
  else
    printf 'inferencex_commit=unknown\n'
  fi
  printf 'utc_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$CONFIG_FILE"

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'Stopping server process group %s...\n' "$SERVER_PID"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true

    local deadline=$((SECONDS + 60))
    while kill -0 "$SERVER_PID" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 2
    done

    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

server_args=(
  "$MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --no-enable-prefix-caching
  --gpu-memory-utilization 0.9
  --tensor-parallel-size 8
  --enable-expert-parallel
  --dtype bfloat16
  --async-scheduling
  --enable-chunked-prefill
  --port "$PORT"
  --additional_config='{"sharding":{"sharding_strategy":{"enable_dp_attention":true, "attn_dp_size":8}}}'
)

printf 'Starting %s server. Full log: %s\n' "$VARIANT" "$SERVER_LOG"
cd "$TPU_INFERENCE_REPO"
setsid vllm serve "${server_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

printf 'Waiting up to %ss for %s/health...\n' "$SERVER_READY_TIMEOUT" "$BASE_URL"
READY_DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --silent --fail "$BASE_URL/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'ERROR: server exited before becoming healthy. Last 100 log lines:\n' >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    printf 'ERROR: server did not become healthy before timeout. Last 100 log lines:\n' >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

printf 'Server is healthy. Running %s trial(s).\n' "$TRIALS"
trial=1
while (( trial <= TRIALS )); do
  RESULT_FILENAME="${VARIANT}_trial${trial}.json"
  CLIENT_LOG="$RESULT_DIR/${VARIANT}_trial${trial}.log"
  printf 'Running %s trial %s/%s...\n' "$VARIANT" "$trial" "$TRIALS"

  python3 "$BENCH_CLIENT" \
    --model "$MODEL" \
    --backend vllm \
    --base-url "$BASE_URL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --max-concurrency "$CONC" \
    --request-rate inf \
    --ignore-eos \
    --use-chat-template \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$RESULT_FILENAME" \
    2>&1 | tee "$CLIENT_LOG"

  trial=$((trial + 1))
done

printf 'utc_finish=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CONFIG_FILE"
printf 'Completed %s. Results: %s\n' "$VARIANT" "$RESULT_DIR"
stop_server
trap - EXIT INT TERM
