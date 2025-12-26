#!/bin/bash
#SBATCH --job-name=1p2d_bench-serving    # Specify a custom string for your slurm batch job
#SBATCH -N 3            # CHECK this to be right in batch jobs 
#SBATCH -n 3          # CHECK this to be right in batch jobs
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="slurm_job-%j.out"
#SBATCH --error="slurm_job-%j.err"


# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# Define valid model names
VALID_MODELS=( \
    "DeepSeek-V3" \
    "DeepSeek-R1" 
)


# Each model has an associated run file - Set it here
declare -A MODEL_RUNFILES=(
    ["DeepSeek-V3"]="sglang_disagg_server.sh"
    ["DeepSeek-R1"]="sglang_disagg_server.sh"
)

# Check if MODEL_NAME exists and fetch runfile
if [[ -n "${MODEL_RUNFILES[$MODEL_NAME]}" ]]; then
    RUN_FILE="${MODEL_RUNFILES[$MODEL_NAME]}"
    echo "Model found: $MODEL_NAME"
    echo "Runfile set: $RUN_FILE"
else
    echo "Error: Model '$MODEL_NAME' not found in MODEL_RUNFILES"
    echo "Available models: ${!MODEL_RUNFILES[@]}"
    exit 1
fi

export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224}"
if test -z "${DOCKER_IMAGE_NAME}"; then
  echo "Error: DOCKER_IMAGE_NAME is not set or empty."
  exit 1
fi

# Set current directory to be REPO directory with all relevant scripts
export DI_REPO_DIR=$(pwd)

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

# Parallelism Configuration with defaults
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE=${DECODE_MTP_SIZE:-0} # 0 for disabling MTP

# Benchmark Configuration with defaults
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-"/nfsdata"}"

validate_model_name() {
  local is_valid_model=false

  for model in "${VALID_MODELS[@]}"; do
    if [[ "$MODEL_NAME" == "$model" ]]; then
      is_valid_model=true
      break
    fi
  done

  if ! $is_valid_model; then
    printf "Error: Invalid MODEL_NAME: '%s'\nValid models are:\n" "$MODEL_NAME"
    for model in "${VALID_MODELS[@]}"; do
      printf "  - %s\n" "$model"
    done
    exit 1
  fi

  echo "MODEL_NAME '$MODEL_NAME' is valid."
  return 0
}

validate_model_name "${MODEL_NAME}"


# ------------------------
# Model path validation and selection across all nodes
# ------------------------
echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

# Get all allocated nodes
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

# Function to check model path on all nodes
check_model_path() {
    local path=$1
    local check_name=$2
    
    echo "Checking $check_name: $path"
    
    # Run check on all nodes in parallel
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
        if [ -d '$path' ]; then 
            echo \"\$(hostname): ✓ Found $path\"
            exit 0
        else 
            echo \"\$(hostname): ✗ Missing $path\"
            exit 1
        fi
    "
    
    # Check if all nodes succeeded (exit code 0)
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $check_name available on ALL nodes"
        return 0
    else
        echo "✗ $check_name NOT available on all nodes"
        return 1
    fi
}

# Check model weights exist on "$MODEL_DIR/$MODEL_NAME"
if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
else
    echo ""
    echo "✗ FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in the following:"
    echo "  - $MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "Model must be accessible from all nodes for distributed execution."
    echo "Please ensure the model is available on all allocated nodes."
    exit 1
fi

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""

NUM_NODES="${NUM_NODES}"

# ------------------------
# Extract first NUM_NODES from SLURM allocation and update SLURM variables
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Get the full nodelist and extract first NUM_NODES
FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

# Create new nodelist in SLURM format
# This is a simplified approach - for complex ranges, you might need more sophisticated parsing
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/bill}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-m2m}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-1p1d_bench-serving}"

echo ""
echo "Updated SLURM Environment Variables:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_QOS: $SLURM_JOB_QOS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_CONF: $SLURM_CONF"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# Node information
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 1.1.1.1')
NODE0_ADDR=$(echo "$NODE0_ADDR" | awk '/src/ {print $7}')

IPS=()

GW_NIC=$(ip route | awk '/^default/ {print $5; exit}')
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1')
    IP=$(echo "$IP" | awk '/src/ {print $7}')
    IPS+=("$IP")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

SGL_WS_PATH="/sglang_disagg"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "NODE0_ADDR is ${NODE0_ADDR}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${DI_REPO_DIR}"
echo "USER_NAME is ${USER_NAME}"



export DI_REPO_DIR=$DI_REPO_DIR
export SGL_WS_PATH=$SGL_WS_PATH
export NNODES=$NNODES
export NODE0_ADDR=$NODE0_ADDR
export MODEL_PATH=$MODEL_PATH
export MODEL_DIR=$MODEL_DIR
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export PREFILL_TP_SIZE=$PREFILL_TP_SIZE
export PREFILL_ENABLE_EP=$PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP=$PREFILL_ENABLE_DP
export DECODE_TP_SIZE=$DECODE_TP_SIZE
export DECODE_ENABLE_EP=$DECODE_ENABLE_EP
export DECODE_ENABLE_DP=$DECODE_ENABLE_DP
export DECODE_MTP_SIZE=$DECODE_MTP_SIZE
export BENCH_INPUT_LEN=$BENCH_INPUT_LEN
export BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN
export BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY

export DOCKER_CONT_NAME="container_sbatch_${USER_NAME}_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$SGL_WS_PATH/${RUN_FILE}"


# Clean-up handler for TERM (and Ctrl-C/INT if you want)
cleanup() {
  echo "[${SLURM_JOB_ID}] SIGTERM received; starting clean-up..."
  # Example: save checkpoint, stop background workers, flush logs
  # pkill -TERM -P $$   # propagate to child processes if needed
  # tar -czf state_${SLURM_JOB_ID}.tgz ./state
  # Optionally, request requeue if your site supports it:
  # scontrol requeue "${SLURM_JOB_ID}"
  srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'sudo docker stop -f $DOCKER_CONT_NAME'
  echo "[${SLURM_JOB_ID}] clean-up done."
  exit 0
}


trap cleanup EXIT INT TERM


# Use only the selected nodes for srun execution
SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

# Remove all containers with prefix `container_sbatch_` before running the new job so making it idempotent

# Force NFS cache refresh on all nodes before running Docker to avoid stale file handle errors
echo "Refreshing NFS caches on all nodes..."
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
    sync
    # Force re-stat of the mounted directory to refresh NFS handles
    ls -la '"$DI_REPO_DIR"' > /dev/null 2>&1
    stat '"$DI_REPO_DIR"'/sglang_disagg_server.sh > /dev/null 2>&1
    cat '"$DI_REPO_DIR"'/sglang_disagg_server.sh > /dev/null 2>&1
    # Drop caches if we have permission (optional, requires root)
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    echo "NFS cache refreshed on $(hostname)"
'

srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
sudo docker ps --filter "name=^container_sbatch_" --format "{{.ID}}" | xargs --no-run-if-empty sudo docker stop || true;
sudo docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v ${MODEL_DIR}:/models \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v $DI_REPO_DIR:$SGL_WS_PATH \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID  \
    -e NODE0_ADDR=$NODE0_ADDR \
    -e MODEL_DIR=/models    \
    -e SGL_WS_PATH=$SGL_WS_PATH \
    -e xP=$xP \
    -e yD=$yD \
    -e MODEL_NAME=$MODEL_NAME \
    -e IPADDRS=$IPADDRS \
    -e PREFILL_TP_SIZE=$PREFILL_TP_SIZE \
    -e PREFILL_ENABLE_EP=$PREFILL_ENABLE_EP \
    -e PREFILL_ENABLE_DP=$PREFILL_ENABLE_DP \
    -e DECODE_TP_SIZE=$DECODE_TP_SIZE \
    -e DECODE_ENABLE_EP=$DECODE_ENABLE_EP \
    -e DECODE_ENABLE_DP=$DECODE_ENABLE_DP \
    -e DECODE_MTP_SIZE=$DECODE_MTP_SIZE \
    -e BENCH_INPUT_LEN=$BENCH_INPUT_LEN \
    -e BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN \
    -e BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO \
    -e BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER \
    -e BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME bash -c "
        mkdir -p /run_logs/slurm_job-${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/pd_sglang_bench_serving.sh_NODE${SLURM_PROCID}.log
    "
DOCKER_EXIT_CODE=$?
if [ $DOCKER_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Docker container on $(hostname) exited with code $DOCKER_EXIT_CODE"
    exit $DOCKER_EXIT_CODE
fi
'

cleanup

