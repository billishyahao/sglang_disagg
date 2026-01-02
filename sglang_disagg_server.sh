#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
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

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

source $SGL_WS_PATH/set_env_vars.sh

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

# Common configurations shared by both prefill and decode (base)
declare -A MODEL_BASE_CONFIGS=(
    ["DeepSeek-R1"]="--decode-log-interval 1 --watchdog-timeout 1000000 --chunked-prefill-size 262144 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter"
)


# MTP configurations (only when DECODE_MTP_SIZE is set and greater than zero)
declare -A MODEL_MTP_CONFIGS=(
    ["DeepSeek-R1"]="--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens ${DECODE_MTP_SIZE}"
)


# DP-specific common configurations (only when DP is enabled)
declare -A MODEL_DP_CONFIGS=(
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head"
)

# Prefill-specific configurations
# Set parameters based on DP enable status
if [[ "$PREFILL_ENABLE_DP" == "true" ]]; then
    prefill_cuda_graph_bs=($(seq 1 3))
    prefill_max_running_requests=8
else
    prefill_cuda_graph_bs=($(seq 1 128))
    prefill_max_running_requests=128
fi

declare -A MODEL_PREFILL_CONFIGS=(
    ["DeepSeek-R1"]="--mem-fraction-static 0.8 --max-running-requests ${prefill_max_running_requests} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} --disable-radix-cache"
)

# Decode-specific configurations
# Set parameters based on DP enable status
if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq 1 128))
    decode_max_running_requests=8192
else
    decode_cuda_graph_bs=($(seq 1 256))
    decode_max_running_requests=256
fi

declare -A MODEL_DECODE_CONFIGS=(
    ["DeepSeek-R1"]="--mem-fraction-static 0.6 --max-running-requests ${decode_max_running_requests} --cuda-graph-bs ${decode_cuda_graph_bs[*]} --prefill-round-robin-balance"
)



# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_NODES_PER_WORKER=$((PREFILL_TP_SIZE / 8))
DECODE_NODES_PER_WORKER=$((DECODE_TP_SIZE / 8))
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
    echo "================================================"
    
    # start the head prefill server
    PREFILL_CMD="python3 -m sglang.launch_server \
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

    set -x 
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    set +x

    prefill0_pid=$!
    
    echo "Waiting for all prefill and decode servers to be up . . ."
    python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --timeout 1200

    set -x 
    python -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --port 30000 \
    ${PREFILL_ARGS} \
    ${DECODE_ARGS} \
    2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    set +x
    
    proxy_pid=$!

    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd /sglang_disagg

    # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
    if [ ! -d /sglang_disagg/logs ]; then
        mkdir -p /sglang_disagg/logs
        echo "Created directory: /sglang_disagg/logs"
    fi

    bash /sglang_disagg/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /sglang_disagg/logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} "${BENCH_MAX_CONCURRENCY}" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}
 
    echo "Killing the proxy server and prefill server"
    kill $proxy_pid
    kill $prefill0_pid

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"

    PREFILL_CMD="python3 -m sglang.launch_server \
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

    set -x 
    
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    set +x

    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 30000 \
        --timeout 1200

    echo "Waiting until proxy server closes..."
    python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 30000

    echo "Killing the prefill server"
    kill $prefill_pid

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"
    
    DECODE_CMD="python3 -m sglang.launch_server \
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

    set -x 
    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
    
    decode_pid=$!
    set +x 

    echo "Waiting for proxy server to be up..."
    python $SGL_WS_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 30000 \
        --timeout 1200

    echo "Waiting until proxy server closes..."
    python $SGL_WS_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 30000

    echo "Killing the decode server"
    kill $decode_pid

fi

echo "Script completed successfully"
exit 0
