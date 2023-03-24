import functools
import logging
from datetime import datetime
import os
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
import datasets
import transformers
from torch.utils.data import Dataset, Sampler


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def default_setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    torch.cuda.set_device(dist.get_rank())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def default_setup_deepspeed():
    import deepspeed

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if Accelerator().is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    torch.cuda.set_device(dist.get_rank())
    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl", verbose=False)


def add_tokens(tokenizer, model, tokens):
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def optimized_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


class SmartBatchSampler(Sampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        preload_cache: bool = False,
    ) -> None:
        super(SmartBatchSampler, self).__init__(data_source=data_source)
        self.batch_size = batch_size
        self.data_source = data_source

        sentence_lengths = [
            len(items["tokens"]) if items["tokens"].dim() != 0 else 1
            for items in data_source
        ]
        sentence_indices = [idx for idx in range(len(data_source))]

        pack_by_length = list(zip(sentence_lengths, sentence_indices))
        if preload_cache:
            sort_by_length = sorted(pack_by_length, reverse=True)
        else:
            sort_by_length = sorted(pack_by_length)
        sentence_lengths, sentence_indices = zip(*sort_by_length)

        self.bins = [
            sentence_indices[i : i + batch_size]
            for i in range(0, len(sentence_indices), batch_size)
        ]
        self.bins = list(chain.from_iterable(self.bins))

        if not preload_cache:
            np.random.shuffle(self.bins)

    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)


def fuse_gelu_megatron(model):
    @torch.jit.script
    def gelu_fwd(x):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def gelu_bwd(g, x):
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
        ) + 0.5 * (1 + tanh_out)
        return ff * g

    class _FusedGeLUFunction(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, input):
            ctx.input_tensor = input
            return gelu_fwd(input)

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.input_tensor
            tmp = gelu_bwd(grad_output, input)
            return tmp

    class FusedGelu(torch.nn.Module):
        def forward(self, input):
            return _FusedGeLUFunction.apply(input)

    fused_gelu_module = FusedGelu()
    hf_gelu_functions = [
        transformers.activations.GELUActivation,
        transformers.activations.FastGELUActivation,
        transformers.activations.NewGELUActivation,
        transformers.activations.QuickGELUActivation,
    ]

    for name, module in model.named_modules():
        for hf_gelu_function in hf_gelu_functions:
            if isinstance(module, hf_gelu_function):
                rsetattr(model, name, fused_gelu_module)

    return model


def get_timestamp():
    now = datetime.now()
    return now.strftime("%y%m%d-%H%M")


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)
