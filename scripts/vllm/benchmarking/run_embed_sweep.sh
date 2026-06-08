#!/bin/bash
# Benchmark sweep: MERGE_MIXED_SEQS x input_len x compute_size
# Usage: bash run_embed_sweep.sh 2>&1 | tee sweep_results.txt

set -e
cd "$(dirname "$0")"

COMMON_ARGS="--model Qwen/Qwen3-Embedding-8B
             --num-prompts 1024
             --batch-size 1024
             --num-iters-warmup 5
             --num-iters 10
             --trust-remote-code
             --max-num-seqs 1024"

run_one() {
    local merge=$1 csize=$2 ilen=$3
    echo ""
    echo "=========================================="
    echo " MERGE_MIXED_SEQS=$merge  COMPUTE_SIZE=$csize  input_len=$ilen"
    echo "=========================================="
    MERGE_MIXED_SEQS=$merge COMPUTE_SIZE=$csize \
        python benchmark_offline_embedding.py \
        $COMMON_ARGS \
        --random-input-len $ilen 2>&1 \
      | grep -E "(===|---|Requests processed|Total input|Avg input|Duration|Request throughput|Token throughput|Mean latency|Median latency|Std latency|P99 latency)"
}

# MERGE=0: COMPUTE_SIZE has no effect, run once per input_len
for ilen in 4 8 16; do
    run_one 0 128 $ilen
done

# MERGE=1: sweep all COMPUTE_SIZE values
#for csize in 128 256 512; do
#    for ilen in 4 8 16; do
#        run_one 1 $csize $ilen
#    done
#done

echo ""
echo "Sweep complete."
