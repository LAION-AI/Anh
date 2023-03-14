"""
torchrun --nproc_per_node=8 train.py -c configs/example-230314.yaml
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

# 1. Default setup for multi-gpu training
default_setup()

# 2. Get config path
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--local_rank", type=int, default=0)
config_path = parser.parse_args().config

# 3. Load config and set seed
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    if dist.get_rank() == 0:
        print(config)
    set_seed(config["training"]["seed"])

# 4. Get data loading and train functions
model_type = config["model_and_tokenizer"]["pretrained_model_type"]
if "causal" in model_type.lower():
    from _decoder import get_data_loader
else:
    raise Exception("Unknown model type: {}".format(model_type))

# 5. Load tokenizer
tokenizer = (
    TokenizerFactory()
    .get(config["model_and_tokenizer"]["pretrained_tokenizer_type"])
    .from_pretrained(
        config["model_and_tokenizer"]["pretrained_tokenizer_name"],
        use_auth_token=True,
    )
)
tokenizer.model_input_names = config["model_and_tokenizer"]["model_input_names"]

# 7. Load model
model = (
    ModelFactory()
    .get(config["model_and_tokenizer"]["pretrained_model_type"])
    .from_pretrained(
        config["model_and_tokenizer"]["pretrained_model_name"],
        cache_dir=config["model_and_tokenizer"]["cache_dir"],
        low_cpu_mem_usage=config["model_and_tokenizer"]["low_cpu_mem_usage"],
        use_auth_token=True,
    )
)

# 8. Fuse the efficient activation function
if config["efficiency"]["activation_fusion"]:
    model = fuse_gelu_megatron(model)

# 9. Setup special tokens
tokenizer, model = add_tokens(tokenizer, model, config["data"]["special_tokens"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 10. Load dataloaders
train_data_loader = get_data_loader(config, tokenizer, "train")
valid_data_loader = get_data_loader(config, tokenizer, "valid")

# 11. Setup gradient checkpointing
if config["efficiency"]["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()

# 12. Setup cuda and cudnn settings
cudnn.benchmark = False
if config["efficiency"]["allow_tf32"]:
    cuda.matmul.allow_tf32 = True

# 13. Setup deepspeed engine
engine = deepspeed.initialize(
    model=model,
    config=config["deepspeed"],
    model_parameters=optimized_params(
        model,
        config["deepspeed"]["optimizer"]["params"]["weight_decay"],
    ),
)[0]

# 14. Setup wandb monitoring
if dist.get_rank() == 0:
    wandb.init(
        name=config["training"]["exp_name"],
        project=config["training"]["project"],
    )

# 15. Skip training steps if model is already trained
if config["training"]["current_step"] > 0:
    for _ in range(config["training"]["current_step"]):
        next(train_data_loader)

table_data = []

# 16. Start training
while True:
    if config["training"]["current_step"] >= config["training"]["total_step"]:
        break

    for i, train_data in enumerate(train_data_loader):
        model.train()
        loss = engine(
            **{k: v.cuda() if torch.is_tensor(v) else v for k, v in train_data.items()}
        ).loss

        if dist.get_rank() == 0:
            print(
                f"EPOCH: {round(config['training']['current_step']/len(train_data_loader), 3)} "
                f"STEP: {config['training']['current_step']}/{len(train_data_loader)}, LOSS: {loss}"
            )
            wandb.log(
                data={"train_loss": loss.item(), "train_ppl": math.exp(loss.item())},
                step=config["training"]["current_step"],
            )

        engine.backward(loss)
        engine.step()

        if i % config["training"]["eval_interval"] == 0:
            if dist.get_rank() == 0:
                print("START VALIDATION")

            with torch.no_grad():
                model.eval()
                val_losses = []
                val_samples = []

                _valid_data_loader = valid_data_loader
                if dist.get_rank() == 0:
                    _valid_data_loader = tqdm(valid_data_loader)
                for valid_data in _valid_data_loader:
                    random_valid_sample = random.choice(valid_data["input_ids"])
                    val_samples.append(random_valid_sample)
                    val_loss = engine(
                        **{
                            k: v.cuda() if torch.is_tensor(v) else v
                            for k, v in valid_data.items()
                        }
                    ).loss
                    val_losses.append(val_loss.detach().item())
                val_loss = sum(val_losses) / len(val_losses)

                random_valid_sample = random.choice(val_samples)
                generation_input = random_valid_sample[:10]
                generation_input_string = tokenizer.decode(generation_input)
                generation_output = model.generate(
                    input_ids=generation_input.unsqueeze(0).cuda(),
                    top_p=0.7,
                    num_beams=4,
                    max_length=100,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    use_cache=True,
                )
                generation_output_string = tokenizer.decode(generation_output[0])
                table_data.append(
                    [
                        config["training"]["current_step"],
                        generation_input_string,
                        generation_output_string,
                    ]
                )
                table = wandb.Table(
                    columns=["step", "input", "output"], data=table_data
                )

                if dist.get_rank() == 0:
                    print("=" * 100)
                    print(
                        f"STEP: {config['training']['current_step']}/{len(valid_data_loader)}, LOSS: {loss}"
                    )
                    print("=" * 100)

                    wandb.log(
                        data={
                            "val_loss": val_loss,
                            "val_ppl": math.exp(val_loss),
                            "generation": table,
                        },
                        step=config["training"]["current_step"],
                    )
                    model.save_pretrained(
                        f"{config['training']['save_path']}/"
                        f"{config['training']['exp_name']}/"
                        f"steps={config['training']['current_step']}-val-loss={round(val_loss, 5)}",
                    )

        config["training"]["current_step"] += 1
