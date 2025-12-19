# AMD InferenceMAX Distributed Inference MI355X Recipe

## List of Models - supported in this recipe, more models support are coming 

- DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- DeepSeek-R1 (https://huggingface.co/deepseek-ai/DeepSeek-R1)
- DeepSeek-R1-0528 (https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

This repository contains scripts and documentation to launch multi nodes distributed inference through using the SGlang framework for above models. You will find setup instructions, node assignment details and benchmarking commands.

## 📝 Prerequisites

- A Slurm cluster with required Nodes -> xP + yD  (minimum size 2: xP=1 and yD=1)
- A prebuilt rocm docker image supporting MI355(GFX950) contains all dependency library including SGLang, AITER, MoRI, AINIC driver e.g. rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1218
- Access to a shared filesystem for log collection( cluster specific)


## Scripts and Benchmarking

Few files of significance:

- sglang_disagg/run_submit_disagg.sh - Run sbatch job automatically, this is entrypoint for CI integation
- sglang_disagg/run_interactive_disagg.sh - Run interactive slurm job so before running, user need to pre-salloc
- sglang_disagg/run_xPyD_models.slurm - Core slurm script to launch docker containers on all nodes using either sbatch or salloc
- sglang_disagg/sglang_disagg_server.sh - Script that runs inside each docker to start required router, prefill and decode services
- sglang_disagg/benchmark.sh - Benchmark script to run vllm/sglang benchmarking tool for performance measurement
- sglang_disagg/benchmark_parser.py - Log parser script to be run on CONCURRENY benchmark log file to generate tabulated data

## Sbatch run command (non-interactive)

Make sure specifying the slurm job need

```bash

export SLURM_ACCOUNT="amd"
export SLURM_PARTITION="compute"
export TIME_LIMIT="24:00:00"
export MODEL_PATH="/nfsdata"
export MODEL_NAME="DeepSeek-R1"
export CONTAINER_IMAGE="rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1218"
export PREFILL_NODES=1
export PREFILL_WORKERS=1
export DECODE_NODES=2
export DECODE_WORKERS=2
export ISL=1024
export OSL=1024
export CONCURRENCIES="2048"
export REQUEST_RATE="inf"
export PREFILL_ENABLE_EP=true
export PREFILL_ENABLE_DP=true
export DECODE_ENABLE_EP=true
export DECODE_ENABLE_DP=true
```

Then submit the batch job into slurm cluster through bash ./run_submit_disagg.sh

## Srun run command (interactive)

Make sure applying for a interactive allocation through salloc 

```bash
salloc -N 3 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
```

Then modifying the following env:
```bash
export xP=1
export yD=2
export MODEL_DIR="/nfsdata"
export MODEL_NAME=DeepSeek-R1
export PREFILL_TP_SIZE=8
export PREFILL_ENABLE_EP=true
export PREFILL_ENABLE_DP=true
export DECODE_TP_SIZE=8
export DECODE_ENABLE_EP=true
export DECODE_ENABLE_DP=true
export BENCH_INPUT_LEN=1024
export BENCH_OUTPUT_LEN=1024
export BENCH_RANDOM_RANGE_RATIO=1
export BENCH_NUM_PROMPTS_MULTIPLIER=10
export BENCH_MAX_CONCURRENCY=2048
```

And run it through bash ./run_interactive_disagg.sh



## Post execution Log files:
A directory inside the LOG_PATH variable in the slurm script is created by the name of slurm_job_ID. 

Inside that folder:

pd_sglang_bench_serving.sh_NODE<>.log - Overall log per ser Node 
decode_NODE<>.log - Decode services
prefill_NODE<>.log - prefill services


## Benchmark parser ( for CONCURRENCY logs) to tabulate different data

```bash
python3 benchmark_parser.py <log_path/benchmark_XXX_CONCURRENCY.log
```

## History and Acknowledgement

This project is served as a helper repository for supporting ROCm inferenceMAX recipe
The first version of this project benefited a lot from the following projects:

- [MAD](https://github.com/ROCm/MAD): MAD (Model Automation and Dashboarding) is a comprehensive AI/ML model automation platform from AMD
- [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX): Open Source Inference Frequent Benchmarking published by Semi Analysis