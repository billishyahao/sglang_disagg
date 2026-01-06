#!/bin/bash

#Sample Commands
# Parallelism Configuration:
# PREFILL_TP_SIZE: Tensor Parallelism size for Prefill nodes (default: 8)
# PREFILL_ENABLE_EP: Enable Expert Parallelism for Prefill nodes (true/false, default: true)
# PREFILL_ENABLE_DP: Enable Data Parallelism for Prefill nodes (true/false, default: true)
# DECODE_TP_SIZE: Tensor Parallelism size for Decode nodes (default: 8)
# DECODE_ENABLE_EP: Enable Expert Parallelism for Decode nodes (true/false, default: true)
# DECODE_ENABLE_DP: Enable Data Parallelism for Decode nodes (true/false, default: true)
# Note: When EP/DP is enabled, its size will be set equal to TP_SIZE

# Benchmark Configuration:
# BENCH_INPUT_LEN: Input sequence length for benchmark (default: 1024)
# BENCH_OUTPUT_LEN: Output sequence length for benchmark (default: 1024)
# BENCH_RANDOM_RANGE_RATIO: Random range ratio for benchmark (default: 1)
# BENCH_NUM_PROMPTS_MULTIPLIER: Number of prompts = max_concurrency * multiplier (default: 10)
# BENCH_MAX_CONCURRENCY: Maximum concurrency for benchmark (default: 512) [can be a single value or a list like "512x128", only using the first value for now]

set -x

export SLURM_ACCOUNT="thshan@amd.com"
export SLURM_PARTITION="compute"
export TIME_LIMIT="24:00:00"
export MODEL_PATH="/it-share/data"
export MODEL_NAME="DeepSeek-R1"
#export CONTAINER_IMAGE="rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1218"
export CONTAINER_IMAGE="rocm/pytorch-private:sglang-0.5.6.post1-rocm700-mi35x-mori-1224-fix-topomap-dscp"
export PREFILL_NODES=1
export PREFILL_WORKERS=1
export DECODE_NODES=2
export DECODE_WORKERS=1
export ISL=1024
export OSL=1024
export CONCURRENCIES="2048"
export REQUEST_RATE="inf"
export PREFILL_ENABLE_EP=true
export PREFILL_ENABLE_DP=true
export DECODE_ENABLE_EP=true
export DECODE_ENABLE_DP=true
export DECODE_MTP_SIZE=2

JOB_ID=$(bash submit_disagg.sh \
    $PREFILL_NODES $PREFILL_WORKERS $DECODE_NODES $DECODE_WORKERS \
    $ISL $OSL $CONCURRENCIES $REQUEST_RATE \
    $PREFILL_ENABLE_EP $PREFILL_ENABLE_DP \
    $DECODE_ENABLE_EP $DECODE_ENABLE_DP)

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
