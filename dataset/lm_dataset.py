import json
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainedDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        # nn.CrossEntropyLoss默认忽略标签为-100的token，因此将padding token的标签设置为-100
        labels[
            input_ids == self.tokenizer.pad_token_id
        ] = -100  # Ignore padding tokens in loss computation

        return input_ids, labels
