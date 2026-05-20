from typing import Literal
from datasets import load_dataset
import datasets
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

import os

os.environ["HF_DATASETS_OFFLINE"] = "1"


class DataPreprocessor:
    def __init__(self, dataset_path, dataset_name, context_len, batch_size):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.context_len = context_len
        self.batch_size = batch_size

        self.dataset = load_dataset(self.dataset_path, self.dataset_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tokenize(self, feature: Literal["train", "test", "validation"]):
        """tokenizes the dataset and adds eos token between each sample."""
        tokenized, tokenized_id = [], []

        for sentence in self.dataset[feature]:
            if sentence["text"].strip() != "":
                sub_tokens = self.tokenizer(sentence["text"])

                tokenized.extend(sub_tokens)
                tokenized_id.extend(sub_tokens["input_ids"])

            else:
                continue

            tokenized.append(self.tokenizer.eos_token)
            tokenized_id.append(self.tokenizer.eos_token_id)

        return tokenized, tokenized_id

    def pack_chunk(self, tokens_list):
        """
        Packs the list into chunks, generating both inputs and shifted labels.
        """
        inputs = []
        labels = []

        # We grab context_len + 1 tokens so we have room to shift
        chunk_size = self.context_len + 1

        for i in range(0, len(tokens_list) - chunk_size + 1, chunk_size):
            chunk = tokens_list[i : i + chunk_size]

            # input: [token_1, token_2, ..., token_n]
            inputs.append(chunk[:-1])

            # label: [token_2, token_3, ..., token_n+1]
            labels.append(chunk[1:])

        return inputs, labels

    def get_dataloader(self, feature: Literal["train", "test", "validation"] = "train"):
        _, tokenized_id = self.tokenize(feature)
        inputs, labels = self.pack_chunk(tokens_list=tokenized_id)

        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(inputs_tensor, labels_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(feature == "train"),
            drop_last=True,
        )

        return dataloader
