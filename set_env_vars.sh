#!/bin/bash

# export TORCH_NCCL_HIGH_PRIORITY=1
# export NCCL_CHECKS_DISABLE=1

# use ibv_devinfo
# export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

# export NCCL_CROSS_NIC=0

# export NCCL_IGNORE_CPU_AFFINITY=1

# use <ip addr> command to get the thernetname
# use <ls /sys/class/net> to see all NICs
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1


# Automatically Fetch the default interface instead of Hard coding.
# export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)
# export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)

# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME},mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9

set -x
NODENAME=$(hostname)
if [[ $NODENAME == GPU* ]]; then
    export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
elif [[ $NODENAME == smci355-ccs-aus* ]]; then
    export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
elif [[ $NODENAME == mia1* ]]; then
    export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
else
    echo "[Error] unable to fetch the hostname"
    exit 1
fi


# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export RCCL_MSCCL_ENABLE=0
# export TOKENIZERS_PARALLELISM=false
# export HSA_NO_SCRATCH_RECLAIM=1
# export RCCL_MSCCLPP_ENABLE=0
# export HSA_ENABLE_IPC_MODE_LEGACY=1
export NCCL_IB_HCA=$IBDEVICES

export SGLANG_USE_AITER=1

export SGLANG_MORI_DISPATCH_DTYPE=auto
export SGLANG_MORI_FP8_COMB=true
export SGLANG_MORI_QP_PER_TRANSFER=4
export SGLANG_MORI_NUM_WORKERS=4

export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768 
export MORI_IO_QP_MAX_SGE=4

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600

# Disable allocating memory in one pass
export MORI_SHMEM_MODE=ISOLATION

# Enable spec v2 
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export SGLANG_LOG_MS=true
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32


export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
export MORI_MAX_DISPATCH_TOKENS_DECODE=512
export MORI_MOE_MAX_INPUT_TOKENS_DECODE=4096

# set MTP size=1 when EP16
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO

export MORI_APP_LOG_LEVEL=INFO

ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

TC=$(( 4 * $ND_DSCP ))

export MORI_RDMA_SL=$ND_PRIO
export MORI_RDMA_TC=$TC

export MORI_IO_SL=$ND_PRIO
export MORI_IO_TC=$TC
export MORI_IO_TC_DISABLE=0

# WA for latest upstream 0218 image 
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

set +x
