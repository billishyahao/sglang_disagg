#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

IPADDRS="${IPADDRS:-localhost}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"
# Parallelism Configuration
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

# Dry Run for debugging purpose
DRY_RUN="${DRY_RUN:-0}"


# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $SGL_WS_PATH/set_env_vars.sh

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

# Validate MORI_RDMA_TC and hostname consistency
if [[ -n "${MORI_RDMA_TC}" ]]; then
    echo "MORI_RDMA_TC is set to: $MORI_RDMA_TC"

    if [[ "$MORI_RDMA_TC" -eq 104 ]]; then
        if [[ "$host_name" != mia1* ]]; then
            echo "ERROR: MORI_RDMA_TC=104 should be applied on Node with prefix 'mia' but Host '$host_name' does not comply "
            exit 1
        fi
        echo "Host '$host_name' has been configured with MORI_RDMA_TC=104"
    elif [[ "$MORI_RDMA_TC" -eq 96 ]]; then
        if [[ "$host_name" == GPU* || "$host_name" == smci355-ccs-aus* ]]; then
            echo "MORI_RDMA_TC compliance check pass.. "
        else
            echo "ERROR: MORI_RDMA_TC=96 should be applied on Node with prefix 'GPU' or 'smci355-ccs-aus' but Host '$host_name' does not comply "
            exit 1
        fi
        echo "Host '$host_name' has been configured with MORI_RDMA_TC=96"
    else
        echo "ERROR: MORI_RDMA_TC=$MORI_RDMA_TC should be either 104 or 96. Please apply the recommended QoS/DSCP configs."
        exit 1
    fi
else
    echo "ERROR: MORI_RDMA_TC is not set. "
    exit 1
fi

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

# Common configurations shared by both prefill and decode (base)

declare -A MODEL_BASE_CONFIGS=(
    ["DeepSeek-V3"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-V3-0324"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-0528"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-MXFP4"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-0528-MXFP4"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
    ["DeepSeek-R1-0528-MXFP4-th"]="--decode-log-interval 100 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori"
)


# MTP configurations (only when DECODE_MTP_SIZE is set and greater than zero)
if [[ "$DECODE_MTP_SIZE" =~ ^[0-9]+$ ]] && [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    declare -A MODEL_MTP_CONFIGS=(
        ["DeepSeek-V3"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-V3-0324"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1-0528"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1-MXFP4"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"        
        ["DeepSeek-R1-0528-MXFP4"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1-0528-MXFP4-Preview"]="--speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
        ["DeepSeek-R1-0528-MXFP4-th"]="--speculative-draft-model-path SGLang/DeepSeek-R1-NextN --speculative-algorithm NEXTN --speculative-num-steps ${DECODE_MTP_SIZE} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
    )
fi

# DP-specific common configurations (only when DP is enabled)
declare -A MODEL_DP_CONFIGS=(
    ["DeepSeek-V3"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-V3-0324"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-0528"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-MXFP4"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-0528-MXFP4"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
    ["DeepSeek-R1-0528-MXFP4-th"]="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
)


# Prefill-specific configurations
# Set parameters based on DP enable status
if [[ "$PREFILL_ENABLE_DP" == "true" ]]; then
    prefill_max_running_requests=$((MORI_MAX_DISPATCH_TOKENS_DECODE * DECODE_TP_SIZE))
    prefill_chunked_prefill_size=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * PREFILL_TP_SIZE))
else
    prefill_max_running_requests=256
    prefill_chunked_prefill_size=262144
fi


declare -A MODEL_PREFILL_CONFIGS=(
    ["DeepSeek-V3"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}   --disable-radix-cache"
    ["DeepSeek-V3-0324"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}   --disable-radix-cache"
    ["DeepSeek-R1"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}   --disable-radix-cache "
    ["DeepSeek-R1-0528"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}   --disable-radix-cache "
    ["DeepSeek-R1-MXFP4"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}    --disable-radix-cache"
    ["DeepSeek-R1-0528-MXFP4"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}    --disable-radix-cache"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}    --disable-radix-cache"
    ["DeepSeek-R1-0528-MXFP4-th"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size}    --disable-radix-cache"
)


# Decode-specific configurations
# Set parameters based on DP enable status
if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq 1 "$MORI_MAX_DISPATCH_TOKENS_DECODE"))
    decode_max_running_requests=$((MORI_MAX_DISPATCH_TOKENS_DECODE * DECODE_TP_SIZE))

elif [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq 1 256))
    decode_max_running_requests=256
else
    decode_cuda_graph_bs=($(seq 1 128))
    decode_max_running_requests=128
fi

if [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
fi


##FIXME(billishyahao): This is only workaround for now. We will eliminate this chunked-prefill-size for decode node in the future
declare -A MODEL_DECODE_CONFIGS=(
    ["DeepSeek-V3"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-V3-0324"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-0528"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-MXFP4"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-0528-MXFP4"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-0528-MXFP4-Preview"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
    ["DeepSeek-R1-0528-MXFP4-th"]="--mem-fraction-static 0.85 --max-running-requests ${decode_max_running_requests}  --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
)


# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

# Ceiling division by 8 for nodes-per-worker
PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / 8))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / 8))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build prefill arguments dynamically based on xP
PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

# Build decode arguments dynamically based on yD
DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[@]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[@]}"

# =============================================================================
# Configuration Builder Functions
# =============================================================================

build_server_config() {
    local mode="$1"
    local model_name="$2"
    local tp_size="$3"
    local enable_ep="$4"
    local enable_dp="$5"
    local decode_mtp_size="$6"
    
    # Calculate EP and DP sizes based on enable flags
    local ep_size=1
    local dp_size=1
    
    if [[ "$enable_ep" == "true" ]]; then
        ep_size=$tp_size
    fi
    
    if [[ "$enable_dp" == "true" ]]; then
        dp_size=$tp_size
    fi
    
    # Build parallelism arguments
    local parallel_args="--tp-size ${tp_size}"
    
    if [[ "$enable_ep" == "true" ]]; then
        parallel_args="$parallel_args --ep-size ${ep_size}"
    fi
    
    if [[ "$enable_dp" == "true" ]]; then
        parallel_args="$parallel_args --dp-size ${dp_size}"
    fi
    
    # Get model-specific configuration
    local base_config=""
    local mtp_config=""
    local dp_config=""
    local specific_config=""
    
    if [[ -n "$model_name" ]]; then
        # Get base configuration
        if [[ -n "${MODEL_BASE_CONFIGS[$model_name]}" ]]; then
            base_config="${MODEL_BASE_CONFIGS[$model_name]}"
        fi

        # Get MTP-related configuration (only if MTP is enabled)
        if [ "$decode_mtp_size" -gt 0 ] && [[ -n "${MODEL_MTP_CONFIGS[$model_name]}" ]]; then
            mtp_config="${MODEL_MTP_CONFIGS[$model_name]}"
        fi
        
        # Get DP-related configuration (only if DP is enabled)
        if [[ "$enable_dp" == "true" ]] && [[ -n "${MODEL_DP_CONFIGS[$model_name]}" ]]; then
            dp_config="${MODEL_DP_CONFIGS[$model_name]}"
        fi
        
        # Get mode-specific configuration
        if [[ "$mode" == "prefill" ]]; then
            if [[ -n "${MODEL_PREFILL_CONFIGS[$model_name]}" ]]; then
                specific_config="${MODEL_PREFILL_CONFIGS[$model_name]}"
            fi
        elif [[ "$mode" == "decode" ]]; then
            if [[ -n "${MODEL_DECODE_CONFIGS[$model_name]}" ]]; then
                specific_config="${MODEL_DECODE_CONFIGS[$model_name]}"
            fi
        fi
    fi
    
    # Combine all configurations: parallel args + base config + mtp config + dp config + specific config
    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
    fi
    if [[ -n "$mtp_config" ]] && [[ "$mode" == "decode" ]]; then
        full_config="$full_config $mtp_config"
    fi
    if [[ -n "$dp_config" ]]; then
        full_config="$full_config $dp_config"
    fi
    if [[ -n "$specific_config" ]]; then
        full_config="$full_config $specific_config"
    fi
    
    echo "$full_config"
}

# Build complete server configurations
PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "$MODEL_NAME" "$PREFILL_TP_SIZE" "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" "$DECODE_MTP_SIZE")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "$MODEL_NAME" "$DECODE_TP_SIZE" "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" "$DECODE_MTP_SIZE")

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $SGL_WS_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300


# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}, MTP size=${DECODE_MTP_SIZE}"
    echo "Decode  parallelism: TP=${DECODE_TP_SIZE},  EP enabled: ${DECODE_ENABLE_EP},  DP enabled: ${DECODE_ENABLE_DP},  MTP size=${DECODE_MTP_SIZE}"
    echo "Prefill servers ($((PREFILL_TP_SIZE/8)) nodes): ${PREFILL_ARGS}"
    echo "Decode servers  ($((DECODE_TP_SIZE/8))  nodes): ${DECODE_ARGS}"
    echo "Prefill env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: ${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    echo "Decode env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    echo "================================================"
    
    # start the head prefill server
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/$MODEL_NAME \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG}"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${$PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
        set +x
        prefill0_pid=$!
    fi

    
    echo "Waiting for all prefill and decode servers to be up . . ."


    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi
    echo "Congratulations!!! All prefill and decode servers are up . . ."

    ROUTER_CMD="python -m sglang_router.launch_router \
        --pd-disaggregation \
        --port 30000 \
        --policy random \
        --prefill-policy random \
        --decode-policy random  \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS}"


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        set -x
        eval "$ROUTER_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
        set +x

        BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-health \
        --health-endpoint /readiness \
        --timeout 1800"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $BARRIER_CMD"
        else
            eval "$BARRIER_CMD"
        fi

        echo "Router is ready for benchmarking"
    fi
    

    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd /sglang_disagg

    # Export IS_MTP based on whether MTP is enabled
    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi

    # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
    BENCH_CMD="bash /sglang_disagg/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} "${BENCH_MAX_CONCURRENCY}" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    if [ ! -d /sglang_disagg/logs ]; then
        mkdir -p /sglang_disagg/logs
        echo "Created directory: /sglang_disagg/logs"
    fi

    # Copy the bench.sh result from tmp folder into shared nfs folder
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} /sglang_disagg/logs/
    fi

    echo "Killing the proxy server and prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $proxy_pid
        kill $prefill0_pid
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"


    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/${MODEL_NAME} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG}"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[$prefill_idx]} --nnodes ${$PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $NODE_RANK prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $prefill_pid
    fi

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"

    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server \
        --model-path ${MODEL_DIR}/${MODEL_NAME} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${DECODE_SERVER_CONFIG}"

    if [ "$DECODE_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &

        set +x
        decode_pid=$!
    fi


    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi


    echo "Waiting until proxy server closes..."
    WAIT_CMD="python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $RANK decode server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $decode_pid
    fi

fi

echo "Script completed successfully"
exit 0
