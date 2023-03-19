#!/bin/bash
#SBATCH --job-name=ahn_training
#SBATCH --account=oslo
#SBATCH --partition=g40
#SBATCH --output=training.out
#SBATCH --error=training.err
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=END

source /fsx/kevin.ai/Anaconda/bin/activate oslo
# export huggingface model download dir
export TRANSFORMERS_CACHE="/fsx/kevin.ai/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/fsx/kevin.ai/.cache/huggingface/datasets"
export TORCH_HOME="/fsx/kevin.ai/.cache/torch"

cd /fsx/kevin.ai/laion/Anh/ahn/training
torchrun train.py -c configs/xglm-train.yaml
