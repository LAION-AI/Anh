"""
torchrun --nproc_per_node=8 train.oslo.py -c configs/xglm-train.yaml
"""
import math
import random
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.backends import cudnn, cuda
from tqdm import tqdm

from _factory import ModelFactory, TokenizerFactory
from _utils import default_setup, optimized_params, add_tokens, fuse_gelu_megatron
from transformers import set_seed

import torch.distributed as dist
import wandb
from datasets import load_dataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from oslo.torch.nn.parallel import DistributedDataParallel, TensorParallel
from oslo.torch.distributed import ParallelContext, ParallelMode
import time

import oslo

# # 1. Default setup for multi-gpu training
# default_setup()

# 2. Get config path
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--local_rank", type=int, default=0)
config_path = parser.parse_args().config

# 3. Load config and set seed
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(config["training"]["seed"])

# # 4. Get data loading and train functions
model_type = config["model_and_tokenizer"]["pretrained_model_type"]
if "causal" in model_type.lower():
    from _decoder import get_data_loader
else:
    raise Exception("Unknown model type: {}".format(model_type))

# 병렬 사이즈 설정
tp_size = 4
tp_depth = 1
dp_size = 4 // tp_size

model_name = config["model_and_tokenizer"]["pretrained_model_name"]

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=dp_size,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    tensor_parallel_depth=1,
)


# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(
  model_name
)

# 모델 생성 및 병렬화 수행
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float32, low_cpu_mem_usage=True
)

tokenizer, model = add_tokens(tokenizer, model, config["data"]["special_tokens"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = TensorParallel(model, parallel_context)
# model = DistributedDataParallel(model, parallel_context)
oslo.ready(model, parallel_context)


if dist.get_rank() == 0:
    print(model)


# dataset load
train_data_loader = get_data_loader(config, tokenizer, "train")
valid_data_loader = get_data_loader(config, tokenizer, "valid")

#. Skip training steps if model is already trained
if config["training"]["current_step"] > 0:
    for _ in range(config["training"]["current_step"]):
        next(train_data_loader)


# 옵티마이저 생성
optimizer_tp = Adam(model.parameters(), lr=3e-5)

# Load config and set seed
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    if dist.get_rank() == 0:
        print(config)
    set_seed(config["training"]["seed"])


# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="laion", name=f"{model_name}_tp2d")
    cur = time.time()

# 모니터링 생성 대기
dist.barrier()

# 학습 시작
for i, train_data in enumerate(train_data_loader):
    optimizer_tp.zero_grad()
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in train_data.items()}

    loss_tp = model(**inputs).loss
    
    if dist.get_rank() == 0:
        print(loss_tp)

    _ = loss_tp.backward()
    if dist.get_rank() == 0:
        print("loss_tp.backward() finished")
    
    optimizer_tp.step()
    if dist.get_rank() == 0:
        print(loss_tp)
        wandb.log({"loss": loss_tp})


    # if i % config["training"]["eval_interval"] == 0:
    #     if dist.get_rank() == 0:
    #         print("START VALIDATION")
    #         wandb.log({"eval_start": time.time() - cur})

    #     val_losses = []
    #     val_samples = []
        
    #     for j, valid_data in enumerate(valid_data_loader):
    #         with torch.no_grad():
    #             inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in valid_data.items()}
                
    #             # multi gpu pdb
    #             import pdb
    #             if dist.get_rank() == 0:
    #                 pdb.set_trace()
    #             dist.barrier()
    #             loss_tp = model(**inputs).loss

    #             if dist.get_rank() == 0:
    #                 pdb.set_trace()
    #             dist.barrier()
    #             val_losses.append(loss_tp)
    #             val_samples.append(inputs["input_ids"][0])

    #             if dist.get_rank() == 0:
    #                 print("START SAMPLE GENERATION")
    #                 val_loss = sum(val_losses) / len(val_losses)

    #             generated_samples = model.generate(
    #                 inputs["input_ids"][0].unsqueeze(0),
    #                 max_length=512,
    #                 num_beams=5,
    #                 no_repeat_ngram_size=2,
    #                 early_stopping=True)

    #             if dist.get_rank() == 0:
    #                 print(f"val_loss:{val_loss}")
    #                 wandb.log({"val_loss": val_loss})
    #                 wandb.log({"eval_end": time.time() - cur})

    #                 # generation
    #                 generation_input_string = tokenizer.decode(
    #                     inputs["input_ids"][0], skip_special_tokens=True
    #                 )
    #                 generation_output_string = tokenizer.decode(
    #                     generated_samples[0],
    #                     skip_special_tokens=True,
    #                 )

    #                 table = wandb.Table(
    #                     columns=["input", "output"],
    #                     data=[[generation_input_string, generation_output_string]],
    #                 )

    #                 wandb.log(
    #                     {
    #                         "val_ppl": math.exp(val_loss),
    #                         "generation": table,
    #                     },
    #                     step=config["training"]["current_step"],
    #                 )

    #                 # 저장 (result/model_name/date)
    #                 import datetime
    #                 model.save_pretrained(f"result/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}", merge_checkpint=True)




    


dist.barrier()