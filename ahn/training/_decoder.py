import jsonlines
import os
from concurrent.futures import ThreadPoolExecutor
import datasets
from accelerate import Accelerator
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


def get_data_loader_hfstyle(config, tokenizer, split):
    accelerator = Accelerator()
    if split == "train":
        dataset = datasets.load_dataset(
            "json", data_files=config["data"]["train_data_path"], split="all"
        )
    elif split == "valid":
        dataset = datasets.load_dataset(
            "json", data_files=config["data"]["valid_data_path"], split="all"
        )
    else:
        raise NotImplementedError(f"split {split} is not implemented")
    if config.get("debug"):
        dataset = dataset.select(range(200))

    def process_fn(examples):
        result = tokenizer(
            examples[config["data"]["train_data_key"]],
            padding=False,
            truncation=True,
            max_length=config["model_and_tokenizer"]["max_length"],
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return result

    with accelerator.main_process_first():
        dataset = dataset.map(
            process_fn, batched=True, batch_size=1000, remove_columns=["metadata"]
        )
    if config.get("debug"):
        dataset = dataset.select(range(100))
    dataset = dataset.rename_columns({"input_ids": "tokens", "text": "sentences"})
    dataset.set_format("pt", columns=["tokens"], output_all_columns=True)

    def collate_fn(batch):
        _tokens, _sentences = [], []
        for sample in batch:
            _tokens.append(sample["tokens"])
            _sentences.append(sample["sentences"])

        enc = tokenizer.pad(
            encoded_inputs={"input_ids": _tokens}, padding="longest", verbose=False
        )
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
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )


def get_data_loader(config, tokenizer, data_type):
    rank = dist.get_rank()
    if data_type == "train":
        data = []
        err = 0
        with jsonlines.open(config["data"]["train_data_path"]) as reader:
            for sample in reader.iter():
                try:
                    data.append(sample[config["data"]["train_data_key"]])
                except:
                    err += 1
                if config.get("debug"):
                    if len(data) == 50:
                        break
        # raw_data = open(config["data"]["train_data_path"]).read().splitlines()
        # data = []
        # err = 0
        # for i in raw_data:
        #     if len(i) > 10:
        #         try:
        #             data.append(json.loads(i)[config["data"]["train_data_key"]])
        #         except:
        #             err += 1
        print(f"number of err when loading train data at rank {rank}: {err}")

    elif data_type == "valid":
        data = []
        err = 0
        with jsonlines.open(config["data"]["valid_data_path"]) as reader:
            for sample in reader.iter():
                try:
                    data.append(sample[config["data"]["valid_data_key"]])
                except:
                    err += 1
                if config.get("debug"):
                    if len(data) == 20:
                        break
        # raw_data = open(config["data"]["valid_data_path"]).read().splitlines()
        # data = []
        # err = 0
        # for i in raw_data:
        #     if len(i) > 10:
        #         try:
        #             data.append(json.loads(i)[config["data"]["valid_data_key"]])
        #         except:
        #             err += 1
        print(f"number of err when loading valid data at rank {rank}: {err}")

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
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
