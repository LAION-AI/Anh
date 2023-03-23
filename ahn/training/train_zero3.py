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
        config["training"]["exp_name"] = "debug_xglm-564M"
        config["model_and_tokenizer"][
            "pretrained_tokenizer_name"
        ] = "facebook/xglm-564M"
        config["model_and_tokenizer"]["pretrained_model_name"] = "facebook/xglm-564M"

    # sync all processes for the same exp_name
    dist.barrier()
    exp_name = f'{config["training"]["exp_name"]}-{get_timestamp()}'

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
    if config.get("resume_from_checkpoint"):
        logger.info(f"Try resuming checkpoint from {config['resume_from_checkpoint']}")
        try:
            _, client_state = engine.load_checkpoint(
                config["resume_from_checkpoint"],
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            exp_name = config["resume_from_checkpoint"].split("/")[-1]
            logger.info(f"Resuming exp_name to {exp_name}")
            if client_state.get("step") and client_state["step"] > 0:
                config["training"]["current_step"] = client_state["step"]
                logger.info(
                    f"Resuming current_step to {config['training']['current_step']}"
                )
        except Exception as e:
            logger.info(f"Failed to load checkpoint with error: {e}")

    # 14. Setup wandb monitoring
    # this is for time sync
    logger.info(f"Experiment name: {exp_name}")
    if dist.get_rank() == 0:
        wandb.init(
            project=config["training"]["project"],
            name=exp_name,
        )

    # 15. Skip training steps if model is already trained
    if config["training"]["current_step"] > 0:
        for _ in range(config["training"]["current_step"]):
            next(train_data_loader)

    table_data = []
    if config.get("debug"):
        config["training"]["total_step"] = len(train_data_loader) + 3
        config["training"]["eval_interval"] = len(train_data_loader)

    # 16. Start training
    while True:
        if config["training"]["current_step"] >= config["training"]["total_step"]:
            break
        engine.module.train()
        for i, train_data in enumerate(train_data_loader):
            if config["training"]["current_step"] >= config["training"]["total_step"]:
                break
            loss = engine(
                **{
                    k: v.cuda() if torch.is_tensor(v) else v
                    for k, v in train_data.items()
                }
            ).loss

            if (
                config["training"]["current_step"]
                % config["training"]["train_print_every"]
                == 0
                or config["training"]["current_step"] == len(train_data_loader) - 1
                or config["training"]["current_step"]
                == config["training"]["total_step"] - 1
            ):
                logger.info(
                    f"[train] EPOCH: {(config['training']['current_step'] + 1)/len(train_data_loader):.3f} "
                    f"STEP: {config['training']['current_step'] + 1}/{config['training']['total_step']}, LOSS: {loss}"
                )
            if dist.get_rank() == 0:
                wandb.log(
                    data={
                        "train_loss": loss.item(),
                        "train_ppl": math.exp(loss.item()),
                    },
                    step=config["training"]["current_step"],
                )

            engine.backward(loss)
            engine.step()

            if (i + 1) % config["training"]["eval_interval"] == 0 or config["training"][
                "current_step"
            ] == config["training"]["total_step"] - 1:
                engine.module.eval()
                logger.info("Start Validation")

                with torch.no_grad():
                    val_losses = []
                    val_samples = []

                    for j, valid_data in enumerate(valid_data_loader):
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
                if (
                    j % config["training"]["eval_print_every"] == 0
                    or j == len(valid_data_loader) - 1
                ):
                    logger.info(
                        f"[valid] STEP: {j}/{len(valid_data_loader) - 1}, LOSS: {loss}"
                    )
                    logger.info(f"Input: {generation_input_string}")
                    logger.info(f"Output: {generation_output_string}")
                if dist.get_rank() == 0:
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
                    # logger.info("=" * 100)
                    # logger.info("=" * 100)

                    wandb.log(
                        data={
                            "val_loss": val_loss,
                            "val_ppl": math.exp(val_loss),
                            "generation": table,
                        },
                        step=config["training"]["current_step"],
                    )
                    # model.save_pretrained(
                    #     f"{config['training']['save_path']}/"
                    #     f"{config['training']['exp_name']}/"
                    #     f"steps={config['training']['current_step']}-val-loss={round(val_loss, 5)}",
                    # )
                # val_loss_tensor = torch.tensor(val_loss).cuda()
                # dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                # dist.barrier()
                # val_loss_tensor /= dist.get_world_size()

                dist.barrier()
                engine.save_checkpoint(
                    save_dir=os.path.join(config["training"]["save_path"], exp_name),
                    client_state={"step": config["training"]["current_step"]},
                )
                torch.cuda.empty_cache()
            config["training"]["current_step"] += 1


if __name__ == "__main__":
    main()
