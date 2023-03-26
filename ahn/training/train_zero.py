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
    # 1. Get config path
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config

    # 2. Default setup for multi-gpu training
    default_setup_deepspeed(args.local_rank)

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
        config["training"]["exp_name"] = "debug-xglm-564M"
        config["model_and_tokenizer"][
            "pretrained_tokenizer_name"
        ] = "facebook/xglm-564M"
        config["model_and_tokenizer"]["pretrained_model_name"] = "facebook/xglm-564M"

    # sync all processes for the same exp_name
    dist.barrier()
    timestamp = get_timestamp()
    jobid = os.environ["SLURM_JOBID"]
    exp_name = f'{config["training"]["exp_name"]}-{jobid}-{timestamp}'
    world_size = dist.get_world_size()
    batch_size_per_gpu = config["deepspeed"]["train_micro_batch_size_per_gpu"]
    zero_stage = config["deepspeed"]["zero_optimization"]["stage"]
    project = f'{config["training"]["project"]}-zero{zero_stage}-ngpus{world_size}-bspg{batch_size_per_gpu}'
    logger.info(f"Project name: {project}")
    logger.info(f"Experiment name: {exp_name}")

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
    if os.path.exists(config["training"]["resume_from_checkpoint"]) and zero_stage != 3:
        with open(os.path.join(config["training"]["resume_from_checkpoint"], "latest"), "r") as f:
            resume_step = int(f.readline().strip().replace("global_step", ""))
            config['training']['resume_step'] = resume_step
        checkpoint_path = os.path.join(config["training"]["resume_from_checkpoint"], f"global_step{resume_step}")
        logger.info(
            f"Try resuming checkpoint from {config['training']['resume_from_checkpoint']}"
        )
        logger.info(
            f"Resuming resume_step to {config['training']['resume_step']}"
        )
    else:
        checkpoint_path = config["model_and_tokenizer"]["pretrained_model_name"]
    model = (
        ModelFactory()
        .get(config["model_and_tokenizer"]["pretrained_model_type"])
        .from_pretrained(
            checkpoint_path,
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
    logger.info(f"Train data loader length: {len(train_data_loader)}")
    avg_length, max_length = 0, 0
    for sample in train_data_loader.dataset:
        avg_length += len(sample["tokens"])
        max_length = max(max_length, len(sample["tokens"]))
    avg_length = avg_length / len(train_data_loader.dataset)
    logger.info(f"Average length of tokens in train dataset: {avg_length:.1f}")
    logger.info(f"Max length of tokens in train dataset: {max_length:.1f}")

    logger.info("Loading valid data loaders...")
    valid_data_loader = get_data_loader_hfstyle(config, tokenizer, "valid")
    logger.info(f"Valid data loader length: {len(valid_data_loader)}")
    avg_length, max_length = 0, 0
    for sample in valid_data_loader.dataset:
        avg_length += len(sample["tokens"])
        max_length = max(max_length, len(sample["tokens"]))
    avg_length = avg_length / len(valid_data_loader.dataset)
    logger.info(f"Average length of tokens in valid dataset: {avg_length:.1f}")
    logger.info(f"Max length of tokens in valid dataset: {max_length:.1f}")
    
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
    if config["training"].get("resume_from_checkpoint") and zero_stage == 3:
        logger.info(
            f"Try resuming checkpoint from {config['training']['resume_from_checkpoint']}"
        )
        try:
            load_path, client_state = engine.load_checkpoint(
                config["training"]["resume_from_checkpoint"],
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            logger.info(
                f"Loaded checkpoint successfully from {load_path}",
                main_process_only=False,
            )
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

    # 15. Setup wandb monitoring
    logger.info(f"Project name: {project}")
    logger.info(f"Experiment name: {exp_name}")
    if dist.get_rank() == 0:
        try:
            wandb.init(
                project=project,
                name=exp_name,
            )
        except Exception as e:
            logger.info(f"Failed to initialize wandb with error: {e}")
            logger.info(f"Try to initialize wandb with local key...")
            with open("wandb.key", "r") as f:
                wandb_key = f.readline().strip()
            wandb.login(key=wandb_key)
            wandb.init(
                project=project,
                name=exp_name,
            )
            logger.info(f"Successfully initialized wandb with local key.")
    dist.barrier()

    # if epoch is specified, we will calculate total_step based on epoch
    if config["training"].get("epochs"):
        config["training"]["total_step"] = len(train_data_loader) * config["training"]["epochs"]

    # if save_interval is not specified, we will set it to eval_interval
    if not config["training"].get("save_interval"):
        config["training"]["save_interval"] = config["training"]["eval_interval"]

    # if model is not resumed from checkpoint, we will set resume_step to 0
    if not config['training'].get("resume_step"):
        config["training"]["resume_step"] = 0

    # if debug mode is on, we will set total_step to len(train_data_loader) + 7
    # and eval_interval to len(train_data_loader)
    if config.get("debug"):
        config["training"]["total_step"] = (
            len(train_data_loader) + 3 
            if config["training"]["resume_step"] == 0 
            else config["training"]["resume_step"] + 3
        )
        config["training"]["eval_interval"] = len(train_data_loader)

    logger.info(f'TOTAL TRAINING STEPS: {config["training"]["total_step"]}')
    logger.info(f'Training start after resume step: {config["training"]["resume_step"]}')

    # 16. Start training
    curr_step, fail_count = 0, 0
    train_losses = []
    engine.module.train()
    while True:
        if curr_step >= config["training"]["total_step"]:
            break
        for i, train_data in enumerate(train_data_loader):
            if curr_step < config["training"]["resume_step"]:
                curr_step += 1
                continue
            if curr_step >= config["training"]["total_step"]:
                break
            try:
                train_loss = engine(
                    **{
                        k: v.cuda() if torch.is_tensor(v) else v
                        for k, v in train_data.items()
                    }
                ).loss
                train_losses.append(train_loss.item())

                if (
                    (curr_step + 1) % config["training"]["train_print_interval"] == 0
                    or curr_step == len(train_data_loader) - 1
                    or curr_step == config["training"]["total_step"] - 1
                ):
                    avg_train_loss = sum(train_losses) / len(train_losses)
                    logger.info(
                        f"[train] EPOCH: {(curr_step + 1)/len(train_data_loader):.3f} "
                        f"STEP: {curr_step + 1}/{config['training']['total_step']} ({i + 1}/{len(train_data_loader)}) "
                        f"LOSS: {avg_train_loss:.5f}"
                    )
                    train_losses = []
                if dist.get_rank() == 0:
                    wandb.log(
                        data={
                            "train_loss": train_loss.item(),
                            "train_ppl": math.exp(train_loss.item()),
                        },
                        step=curr_step + 1,
                    )
                engine.backward(train_loss)
                engine.step()
                fail_count = 0
            except Exception as e:
                logger.info(f"Failed to train with error: {e}")
                logger.info(
                    f"[train FAIL] STEP: {curr_step + 1}/{config['training']['total_step']} "
                    f"INPUT_IDS_LENGTH: {len(train_data['input_ids'][0])}"
                )
                if fail_count >= 10:
                    logger.info("Too many failures, aborting")
                    return
                fail_count += 1
            get_accelerator().empty_cache()

            # validation every eval_interval steps or at the end of training
            if (
                (curr_step + 1) % config["training"]["eval_interval"] == 0
                or curr_step == config["training"]["total_step"] - 1
            ):
                engine.module.eval()
                logger.info("Start Validation")

                table_data, val_losses = [], []
                for j, valid_data in enumerate(valid_data_loader):
                    with torch.no_grad():
                        random_valid_sample = random.choice(valid_data["input_ids"])
                        try:
                            val_loss = engine(
                                **{
                                    k: v.cuda() if torch.is_tensor(v) else v
                                    for k, v in valid_data.items()
                                }
                            ).loss
                            val_losses.append(val_loss.item())
                            if (
                                (j + 1) % config["training"]["eval_print_interval"] == 0
                                or j == len(valid_data_loader) - 1
                            ):
                                logger.info(
                                    f"[valid] STEP: {j + 1}/{len(valid_data_loader)}, LOSS: {val_loss:.5f}"
                                )
                        except Exception as e:
                            logger.info(f"Failed to validate with error: {e}")
                            logger.info(
                                f"[valid FAIL] STEP: {j + 1}/{len(valid_data_loader)} "
                                f"INPUT_IDS_LENGTH: {len(valid_data['input_ids'])}"
                            )
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

                    table_data.append(
                        [
                            curr_step + 1,
                            generation_input_string,
                            generation_output_string,
                        ]
                    )
                avg_val_loss = sum(val_losses) / len(val_losses)
                logger.info(f"[valid] AVG_LOSS: {avg_val_loss:.5f}")
                # wandb log
                if dist.get_rank() == 0:
                    table = wandb.Table(
                        columns=["step", "input", "output"], data=table_data
                    )
                    wandb.log(
                        data={
                            "val_loss": avg_val_loss,
                            "val_ppl": math.exp(avg_val_loss),
                            "generation": table,
                        },
                        step=curr_step + 1,
                    )
                dist.barrier()
                engine.module.train()

            # save model every save_interval steps or at the end of training
            if (
                (curr_step + 1) % config["training"]["save_interval"] == 0 
                or curr_step == config["training"]["total_step"] - 1
            ):
                dist.barrier()
                if zero_stage == 3:
                    engine.save_checkpoint(
                        save_dir=os.path.join(config["training"]["save_root_path"], project),
                        tag=f"global_step{curr_step + 1}",
                        client_state={"step": curr_step + 1},
                    )
                elif dist.get_rank() == 0:
                    save_path = os.path.join(
                        config["training"]["save_root_path"], project, f"global_step{curr_step + 1}"
                    )
                    model.save_pretrained(save_path, max_shard_size='200GB')
                    logger.info(f"Saved model to {save_path}")
                    with open(os.path.join(
                        config["training"]["save_root_path"], project, "latest"
                    ), 'w') as f:
                        f.write(f"global_step{curr_step + 1}")
                    logger.info(f"Update latest to global_step{curr_step + 1}")
                dist.barrier()

            curr_step += 1
            get_accelerator().empty_cache()


if __name__ == "__main__":
    main()
