# AMD InferenceMAX Distributed Inference MI355X Recipe

## List of Models - supported in this recipe, more models support are coming 

- DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- DeepSeek-R1 (https://huggingface.co/deepseek-ai/DeepSeek-R1)
- DeepSeek-R1-0528 (https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

This repository contains scripts and documentation to launch multi nodes distributed inference through using the SGlang framework for above models. You will find setup instructions, node assignment details and benchmarking commands.

## 📝 Prerequisites

- A Slurm cluster with required Nodes -> xP + yD  (minimum size 2: xP=1 and yD=1)
- A prebuilt rocm docker image supporting MI355(GFX950) contains all dependency library including SGLang, AITER, MoRI, AINIC driver e.g. `rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224`
- Access to a shared filesystem for log collection( cluster specific)


## Scripts and Benchmarking

Few files of significance:

| File | Description |
|------|-------------|
| `run_submit_disagg.sh` | Run sbatch job automatically, this is entrypoint for CI integation |
| `run_interactive_disagg.sh` | Run interactive slurm job so before running, user need to pre-salloc |
| `run_xPyD_models.slurm` | Core slurm script to launch docker containers on all nodes using either sbatch or salloc |
| `sglang_disagg_server.sh` | Script that runs inside each docker to start required router, prefill and decode services |
| `bench.sh` | Benchmark script to run vllm/sglang benchmarking tool for performance measurement |
| `benchmark_parser.py` | Log parser script to be run on CONCURRENY benchmark log file to generate tabulated data |

## Specify your IB Devices
Run the following command to list all available InfiniBand (IB) devices:

```bash
ibv_devinfo -l
```

Example output:

```text
8 HCAs found:
        ionic_0
        ionic_1
        ionic_2
        ionic_3
        ionic_4
        ionic_5
        ionic_6
        ionic_7
```

Update `set_env_vars.sh` with the comma-separated list of device names found on your system:

```bash
export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
``` 

## Sbatch run command (non-interactive)

Before submitting the job, ensure you update the following environment variables to match your specific cluster configuration and requirements in `run_submit_disagg.sh`:

```bash

# SLURM Job Configuration
export SLURM_ACCOUNT="amd"       # The account name for SLURM job accounting and resource allocation
export SLURM_PARTITION="compute" # The specific cluster partition (queue) to submit the job to
export TIME_LIMIT="24:00:00"     # Maximum wall time for the job (Hours:Minutes:Seconds)

# Model Configuration
export MODEL_PATH="/nfsdata"     # Base directory where the model weights are stored
export MODEL_NAME="DeepSeek-R1"  # Specific model directory name (joined with MODEL_PATH)
export CONTAINER_IMAGE="rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1223" # Docker image to use for the environment

# Cluster Topology (Disaggregation Setup)
export PREFILL_NODES=1           # Number of prefill nodes
export PREFILL_WORKERS=1         # Number of prefill workers
export DECODE_NODES=2            # Number of decode nodes
export DECODE_WORKERS=2          # Number of decode workers

# Benchmark/Workload Parameters
export ISL=1024                  # Input Sequence Length (number of tokens in the prompt)
export OSL=1024                  # Output Sequence Length (number of tokens to generate)
export CONCURRENCIES="2048"      # Total number of concurrent requests to simulate in the benchmark. The value can be "32,64,128"
export REQUEST_RATE="inf"        # Request per second rate. "inf" means send all requests immediately

# Parallelism Strategies
export PREFILL_ENABLE_EP=true    # Enable Expert Parallelism (EP) for the prefill phase 
export PREFILL_ENABLE_DP=true    # Enable Data Parallelism (DP) for the prefill phase
export DECODE_ENABLE_EP=true     # Enable Expert Parallelism (EP) for the decode phase
export DECODE_ENABLE_DP=true     # Enable Data Parallelism (DP) for the decode phase
```

Then submit the batch job into slurm cluster through `bash ./run_submit_disagg.sh`

## Srun run command (interactive)

Make sure applying for an interactive allocation through salloc 

```bash
salloc -N 3 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
```

Then modifying the following env accordingly in `run_interactive_disagg.sh`:
```bash
# Topology Configuration
export xP=1                          # Number of nodes assigned for prefill
export yD=2                          # Number of nodes assigned for decode

# Model Location
export MODEL_DIR="/nfsdata"          # Base directory path where model weights are stored
export MODEL_NAME=DeepSeek-R1        # Specific subdirectory name for the model (e.g., /nfsdata/DeepSeek-R1)

# Prefill Node Configuration
export PREFILL_TP_SIZE=8             # Tensor Parallelism number for Prefill (usually equals GPUs per node)
export PREFILL_ENABLE_EP=true        # Enable Expert Parallelism (EP) for Prefill
export PREFILL_ENABLE_DP=true        # Enable Data Parallelism (DP) for Prefill

# Decode Node Configuration
export DECODE_TP_SIZE=8              # Tensor Parallelism number for Decode (usually equals GPUs per node)
export DECODE_ENABLE_EP=true         # Enable Expert Parallelism (EP) for Decode
export DECODE_ENABLE_DP=true         # Enable Data Parallelism (DP) for Decode

# Benchmark Settings
export BENCH_INPUT_LEN=1024          # Input Sequence Length (number of tokens in the prompt)
export BENCH_OUTPUT_LEN=1024         # Output Sequence Length (number of tokens to generate)
export BENCH_RANDOM_RANGE_RATIO=1    # Variance ratio for sequence lengths
export BENCH_NUM_PROMPTS_MULTIPLIER=10 # Multiplier to determine total prompts (e.g., 10 * concurrency or batch size)
export BENCH_MAX_CONCURRENCY=2048    # Maximum number of concurrent requests to simulate during the test
```

And run it through `bash ./run_interactive_disagg.sh`



## Post execution Log files:
After execution, a directory named `slurm_job-$SLURM_JOB_ID` is created inside `/tmp` containing the logs. 

Inside that folder:
``` bash
pd_sglang_bench_serving.sh_NODE${NODE_RANK}.log - Overall log per ser Node 
decode_NODE${NODE_RANK}.log - Decode services
prefill_NODE${NODE_RANK}.log - prefill services
```

## Benchmark parser ( for CONCURRENCY logs) to tabulate different data
```
# Display results on screen
python3 benchmark_parser.py /tmp/slurm_job-$SLURM_JOB_ID/pd_sglang_bench_serving.sh_NODE$NODE_RANK.log

# Save to specified CSV file
python3 benchmark_parser.py /tmp/slurm_job-$SLURM_JOB_ID/pd_sglang_bench_serving.sh_NODE$NODE_RANK.log --csv results.csv

# Save to auto-named CSV file
python3 benchmark_parser.py /tmp/slurm_job-$SLURM_JOB_ID/pd_sglang_bench_serving.sh_NODE$NODE_RANK.log --csv
```

## History and Acknowledgement

This project is served as a helper repository for supporting ROCm inferenceMAX recipe
The first version of this project benefited a lot from the following projects:

- [MAD](https://github.com/ROCm/MAD): MAD (Model Automation and Dashboarding) is a comprehensive AI/ML model automation platform from AMD
- [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX): Open Source Inference Frequent Benchmarking published by Semi Analysis