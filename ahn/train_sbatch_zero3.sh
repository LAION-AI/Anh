#!/bin/bash
#SBATCH --job-name=ahn_training_ryan
#SBATCH --account=oslo
#SBATCH --partition=g40
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --time=14-0:00
#SBATCH --mail-type=END
#SBATCH --requeue
#SBATCH --output=/fsx/ryan/project/laion/Anh/ahn/logs/slurm-%j.out

source /fsx/ryan/miniconda3/bin/activate eleuther
# export huggingface model download dir
export TRANSFORMERS_CACHE="/fsx/ryan/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/fsx/ryan/.cache/huggingface/datasets"
export TORCH_HOME="/fsx/ryan/.cache/torch"
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
# cat /admin/home-ryan01/.huggingface/token > ~/.cache/huggingface/token
# cat /admin/home-ryan01/.netrc > ~/.netrc
cd /fsx/ryan/project/laion/Anh/ahn/training
# torchrun --nproc_per_node=8 train.py -c configs/xglm-train.yaml
deepspeed --num_gpus=8 train_zero3.py -c configs/xglm-train-zero3.yaml
