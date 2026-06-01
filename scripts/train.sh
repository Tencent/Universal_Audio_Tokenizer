#!/bin/bash

echo "Start running Universal Audio Tokenizer training script..."

export PATH="/path/to/your/conda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate universal-audio-tokenizer

code_dir=/path/to/your/Universal_Audio_Tokenizer
cd $code_dir

export PYTHONPATH=./:$PYTHONPATH
export CUDA_HOME=/path/to/your/cuda
config_file=${1:-"configs/Universal_Audio_Tokenizer.yaml"}
echo "Config file: $config_file"

# Uncomment these lines for debugging distributed training issues
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

your_hf_token="<your_hf_token_here>"
export HF_TOKEN=${your_hf_token}
export HUGGINGFACE_HUB_TOKEN=${your_hf_token}

# In CUDA 10.2 and later, some cuBLAS-based operations (like F.linear) are not deterministic by default.
# To ensure determinism, cuBLAS requires setting this environment variable to specify the workspace configuration.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

current_date=$(date +%Y%m%d)
echo "DATE: $current_date"

# Use an LLM deployed with VLLM to evaluate the consistency scores of generated attributes
# such as accent, prosody, and non-Linguistic Events with the ground truth.
export VLLM_SERVER_IP=${2:-"123.45.67.89"}  # Replace with your VLLM server IP
export VLLM_SERVER_PORT=${3:-"1234"}    # Replace with your VLLM server port
export no_proxy="${no_proxy},${VLLM_SERVER_IP}"

echo "Start training..."

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
src/train/train.py \
    --config_file $config_file
