#!/bin/bash
#SBATCH --job-name=ryan
#SBATCH --account=eleuther
#SBATCH --partition=a100-cu117
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=4
#SBATCH --time=14-0:00
#SBATCH --mail-type=END
#SBATCH --open-mode=append
#SBATCH --output=/mnt/nvme/ryan01/Anh/ahn/logs/%x-%j.out
# %x: jobname, %j: jobid
source /mnt/nvme/home/ryan01/miniconda3/bin/activate eleuther
# export huggingface model download dir
export TRANSFORMERS_CACHE="/mnt/nvme/home/ryan01/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/mnt/nvme/home/ryan01/.cache/huggingface/datasets"
export TORCH_HOME="/mnt/nvme/home/ryan01/.cache/torch"
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export OMP_NUM_THREADS=1
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

cd /mnt/nvme/ryan01/Anh/ahn/training
# CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 train.py -c configs/xglm-train.yaml
srun --jobid $SLURM_JOBID bash -c 'CUDA_LAUNCH_BLOCKING=1 torchrun \
--nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
train_zero.py -c configs/xglm-train-zero.yaml'
