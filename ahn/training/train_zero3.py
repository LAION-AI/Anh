"""
torchrun --nproc_per_node=8 train.py -c configs/example-230314.yaml
"""
import os
import math
import json
import random
from argparse import ArgumentParser
import deepspeed
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.backends import cudnn, cuda
from accelerate.logging import get_logger
from _factory import ModelFactory, TokenizerFactory
from _utils import (
    default_setup_deepspeed,
    optimized_params,
    add_tokens,
    fuse_gelu_megatron,
    get_timestamp,
)
from transformers import set_seed
from deepspeed.accelerator import get_accelerator


logger = get_logger(__name__)


def main():
    # 1. Default setup for multi-gpu training
    default_setup_deepspeed()

    # 2. Get config path
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    config_path = parser.parse_args().config

    # 3. Load config and set seed
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(json.dumps(config, indent=2))
        set_seed(config["training"]["seed"])

    # 4. Get data loading and train functions
    model_type = config["model_and_tokenizer"]["pretrained_model_type"]
    if "causal" in model_type.lower():
        from _decoder import get_data_loader_hfstyle
    else:
        raise Exception("Unknown model type: {}".format(model_type))

    # use xglm-564M for debug
    if config.get("debug") and config["debug"] == "small":
        config["training"]["exp_name"] = "xglm-564M"
        config["model_and_tokenizer"][
            "pretrained_tokenizer_name"
        ] = "facebook/xglm-564M"
        config["model_and_tokenizer"]["pretrained_model_name"] = "facebook/xglm-564M"

    # sync all processes for the same exp_name
    dist.barrier()
    exp_name = f'{config["training"]["exp_name"]}-{get_timestamp()}'
    save_dir = exp_name

    # 5. Load tokenizer
    logger.info("Loading tokenizer...")
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
    logger.info("Loading model...")
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
    logger.info("Loading train data loaders...")
    train_data_loader = get_data_loader_hfstyle(config, tokenizer, "train")
    logger.info("Loading valid data loaders...")
    valid_data_loader = get_data_loader_hfstyle(config, tokenizer, "valid")

    # 11. Setup gradient checkpointing
    if config["efficiency"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # 12. Setup cuda and cudnn settings
    cudnn.benchmark = False
    if config["efficiency"]["allow_tf32"]:
        cuda.matmul.allow_tf32 = True

    # 13. Setup deepspeed engine
    logger.info("Setup deepspeed engine...")
    engine = deepspeed.initialize(
        model=model,
        config=config["deepspeed"],
        model_parameters=optimized_params(
            model,
            config["deepspeed"]["optimizer"]["params"]["weight_decay"],
        ),
    )[0]

    # 14. Resume from checkpoint if available
    if config["training"].get("resume_from_checkpoint"):
        logger.info(
            f"Try resuming checkpoint from {config['training']['resume_from_checkpoint']}"
        )
        try:
            load_path, client_state = engine.load_checkpoint(
                config["training"]["resume_from_checkpoint"],
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            logger.info(f"Loaded checkpoint successfully from {load_path}", main_process_only=False)
            save_dir = config["training"]["resume_from_checkpoint"].split("/")[-1]
            logger.info(f"Resuming save_dir to {save_dir}")
            if client_state.get("step") and client_state["step"] > 0:
                config["training"]["resume_step"] = client_state["step"]
                logger.info(
                    f"Resuming resume_step to {config['training']['resume_step']}"
                )
            else:
                with open(
                    os.path.join(
                        config["training"]["resume_from_checkpoint"], "latest"
                    ),
                    "r",
                ) as f:
                    resume_step = int(f.readline().strip().replace("global_step", ""))
                    config["training"]["resume_step"] = resume_step
                logger.info(
                    f"Resuming resume_step to {config['training']['resume_step']}"
                )
        except Exception as e:
            logger.info(f"Failed to load checkpoint with error: {e}")
    else:
        config["training"]["resume_step"] = 0

    # 15. Setup wandb monitoring
    # this is for time sync
    logger.info(f"Experiment name: {exp_name}")
    if dist.get_rank() == 0:
        wandb.init(
            project=config["training"]["project"],
            name=exp_name,
        )

    table_data = []
    if config.get("debug"):
        config["training"]["total_step"] = len(train_data_loader) + 3
        config["training"]["eval_interval"] = len(train_data_loader)

    if not config["training"].get("save_interval"):
        config["training"]["save_interval"] = config["training"]["eval_interval"]

    # make sure save_interval is larger than or equal to eval_interval and make it a multiple of eval_interval
    config["training"]["save_interval"] = max(
        config["training"]["eval_interval"]
        * int(
            config["training"]["save_interval"] / config["training"]["eval_interval"]
        ),
        config["training"]["eval_interval"],
    )

    # 16. Start training
    curr_step = 0
    train_losses = []
    engine.module.train()
    while True:
        if curr_step >= config["training"]["total_step"]:
            break
        for train_data in train_data_loader:
            if curr_step <= config["training"]["resume_step"]:
                curr_step += 1
                continue
            train_loss = engine(
                **{
                    k: v.cuda() if torch.is_tensor(v) else v
                    for k, v in train_data.items()
                }
            ).loss
            train_losses.append(train_loss.item())

            if (
                curr_step % config["training"]["train_print_interval"] == 0
                or curr_step == len(train_data_loader) - 1
                or curr_step == config["training"]["total_step"] - 1
            ):
                avg_train_loss = sum(train_losses) / len(train_losses)
                logger.info(
                    f"[train] EPOCH: {(curr_step + 1)/len(train_data_loader):.3f} "
                    f"STEP: {curr_step}/{config['training']['total_step']}, LOSS: {avg_train_loss:.5f}"
                )
                train_losses = []
            if dist.get_rank() == 0:
                wandb.log(
                    data={
                        "train_loss": train_loss.item(),
                        "train_ppl": math.exp(train_loss.item()),
                    },
                    step=curr_step,
                )

            engine.backward(train_loss)
            engine.step()

            if curr_step > 0 and (
                curr_step % config["training"]["eval_interval"] == 0
                or curr_step == config["training"]["total_step"] - 1
            ):
                engine.module.eval()
                logger.info("Start Validation")

                val_losses, val_samples = [], []
                with torch.no_grad():
                    for j, valid_data in enumerate(valid_data_loader):
                        random_valid_sample = random.choice(valid_data["input_ids"])
                        val_samples.append(random_valid_sample)
                        val_loss = engine(
                            **{
                                k: v.cuda() if torch.is_tensor(v) else v
                                for k, v in valid_data.items()
                            }
                        ).loss
                        val_losses.append(val_loss.item())
                        if (
                            j % config["training"]["train_print_interval"] == 0
                            or j == len(valid_data_loader) - 1
                        ):
                            logger.info(
                                f"[valid] STEP: {j}/{len(valid_data_loader) - 1}, LOSS: {val_loss:.5f}"
                            )
                    random_valid_sample = random.choice(val_samples)
                    generation_input = random_valid_sample[:10]
                    generation_input_string = tokenizer.decode(generation_input)
                    generation_output = engine.module.generate(
                        input_ids=generation_input.unsqueeze(0).cuda(),
                        top_p=0.7,
                        num_beams=4,
                        max_length=100,
                        no_repeat_ngram_size=3,
                        synced_gpus=True,
                        early_stopping=True,
                        use_cache=True,
                    )

                generation_output_string = tokenizer.decode(generation_output[0])
                avg_val_loss = sum(val_losses) / len(val_losses)
                logger.info(f"[valid] AVG_LOSS: {avg_val_loss:.5f}")
                logger.info(f"Input: {generation_input_string}")
                logger.info(f"Output: {generation_output_string}")

                # wandb log
                if dist.get_rank() == 0:
                    table_data.append(
                        [
                            curr_step,
                            generation_input_string,
                            generation_output_string,
                        ]
                    )
                    table = wandb.Table(
                        columns=["step", "input", "output"], data=table_data
                    )
                    wandb.log(
                        data={
                            "val_loss": val_loss,
                            "val_ppl": math.exp(val_loss),
                            "generation": table,
                        },
                        step=curr_step,
                    )
                engine.module.train()
            if curr_step > 0 and (
                curr_step % config["training"]["save_interval"] == 0
                or curr_step == config["training"]["total_step"] - 1
            ):
                dist.barrier()
                engine.save_checkpoint(
                    save_dir=os.path.join(config["training"]["save_path"], save_dir),
                    client_state={"step": curr_step},
                )
            curr_step += 1
            get_accelerator().empty_cache()


if __name__ == "__main__":
    main()
