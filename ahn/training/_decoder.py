import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch.distributed as dist
from catalyst.data import DistributedSamplerWrapper
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from _utils import SmartBatchSampler


class DecoderDataset(Dataset):
    def __init__(self, sentences, tokens):
        self.sentences = sentences
        self.tokens = tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentences = self.sentences[item]
        tokens = self.tokens[item]

        return {
            "sentences": sentences,
            "tokens": tokens,
        }


def tokenize(config, tokenizer, data, data_type):
    outputs = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as pool:
        imap = pool.map(
            lambda x: tokenizer(
                x,
                padding=False,
                truncation=False,
                max_length=config["model_and_tokenizer"]["max_length"],
                return_tensors="pt",
                return_attention_mask=False,
                return_token_type_ids=False,
            ).input_ids.squeeze(),
            data,
        )

        if dist.get_rank() == 0:
            print(f"Tokenizing {data_type} data...")
            imap = tqdm(imap, total=len(data))
        for token in imap:
            outputs.append(token)

    return outputs


def get_data_loader(config, tokenizer, data_type):
    if data_type == "train":
        raw_data = open(config["data"]["train_data_path"]).read().splitlines()
        data = []
        err = 0
        for i in raw_data:
            if len(i) > 10:
                try:
                    data.append(json.loads(i)["text"])
                except:
                    err += 1
        print(f"error train data: {err}")

    elif data_type == "valid":
        raw_data = open(config["data"]["valid_data_path"]).read().splitlines()
        data = []
        err = 0
        for i in raw_data:
            if len(i) > 10:
                try:
                    data.append(json.loads(i)["text"])
                except:
                    err += 1
        print(f"error valid data: {err}")

    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    tokens = tokenize(config, tokenizer, data, data_type)
    dataset = DecoderDataset(data, tokens)

    def collate_fn(batch):
        _tokens, _sentences = [], []
        for sample in batch:
            _tokens.append(sample["tokens"])
            _sentences.append(sample["sentences"])

        enc = tokenizer.pad(encoded_inputs={"input_ids": _tokens}, padding=True)
        input_ids = enc.input_ids[:, : config["model_and_tokenizer"]["max_length"]]
        attention_mask = enc.attention_mask[
            :, : config["model_and_tokenizer"]["max_length"]
        ]

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "use_cache": False,
        }

        if (
            "token_type_ids" in enc
            and "token_type_ids" in config["model_and_tokenizer"]["model_input_names"]
        ):
            outputs["token_type_ids"] = enc.token_type_ids[
                :, : config["model_and_tokenizer"]["max_length"]
            ]

        return outputs

    if config["efficiency"]["smart_batching"]:
        sampler = SmartBatchSampler(
            dataset, batch_size=config["deepspeed"]["train_micro_batch_size_per_gpu"]
        )
        sampler = DistributedSamplerWrapper(sampler=sampler, shuffle=True)
    else:
        sampler = DistributedSampler(dataset=dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=config["deepspeed"]["train_micro_batch_size_per_gpu"],
        num_workers=os.cpu_count() // dist.get_world_size(),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
