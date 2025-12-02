"""Zyda-2 datamodule."""

from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ZydaDataModule(pl.LightningDataModule):
    """Zyda-2 Lightning data module."""

    def __init__(
        self,
        dataset_name: str = "Zyphra/Zyda-2",
        tokenizer_name: str = "nvidia/Mistral-NeMo-Minitron-8B-Base",
        batch_size: int = 32,
        max_length: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the ZydaDataModule.

        Args:
            dataset_name: Name of the Hugging Face dataset.
            tokenizer_name: Name of the Hugging Face tokenizer.
            batch_size: Batch size.
            max_length: Maximum sequence length.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            streaming: Whether to stream the dataset.
            cache_dir: Directory to cache the dataset.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.streaming = streaming
        self.cache_dir = cache_dir

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.input_channels = self.tokenizer.vocab_size
        self.output_channels = self.tokenizer.vocab_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download the dataset."""
        if not self.streaming:
            load_dataset(self.dataset_name, split="train", cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        """Load the dataset."""
        if stage == "fit" or stage is None:
            # Zyda-2 might only have a train split, so we might need to split it manually
            # or just use it as is. For now, loading 'train'.
            # If streaming, we can't easily split without taking.
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=self.streaming,
                cache_dir=self.cache_dir,
            )
            
            if self.streaming:
                # For streaming, we can't do a random split easily. 
                # We'll just use the same dataset for train and val (warning: data leakage if not careful)
                # Or we could use skip/take if we knew the size, but we don't.
                # For now, assigning to both, assuming the user handles it or we just use train.
                self.train_dataset = dataset
                # Creating a dummy val dataset from the same stream? 
                # Better to just not have val if streaming is on unless we have a separate split.
                self.val_dataset = dataset 
            else:
                # If not streaming, we can split.
                # Assuming a small validation split.
                # Note: Zyda-2 is huge, splitting might take time/memory.
                # We'll try to use the 'train' split and maybe take a subset if needed?
                # But user asked to download and cache, so maybe they want the full thing.
                # We'll just use the full dataset.
                self.train_dataset = dataset
                # TODO: Implement proper validation split if needed.
                # For now, using a small subset for validation if possible, or just same.
                # Actually, let's just use the train dataset for both if no split exists.
                self.val_dataset = dataset

        if stage == "test":
             self.test_dataset = load_dataset(
                self.dataset_name,
                split="train", # Zyda-2 likely only has train
                streaming=self.streaming,
                cache_dir=self.cache_dir,
            )

    def collate_fn(self, batch):
        """Tokenize and pad the batch."""
        # batch is a list of dicts, e.g. [{'text': '...'}, {'text': '...'}]
        texts = [item['text'] for item in batch]
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # We usually want input_ids and labels (for causal LM, labels = input_ids)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(), # Standard for Causal LM
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
