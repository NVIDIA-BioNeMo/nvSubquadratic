"""WikiText-103 DataModule for causal language modeling.

Downloads WikiText-103 via HuggingFace datasets, tokenizes with GPT-2 BPE tokenizer,
concatenates all text, and chunks into fixed-length sequences for next-token prediction.

Compatible with AutoregressiveWrapper in discrete mode.
"""

import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class WikiText103Dataset(Dataset):
    """Chunked, tokenized WikiText-103 dataset for causal LM.

    Args:
        token_ids: 1D tensor of all token IDs (concatenated).
        seq_len: Length of each training sequence.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        # Drop remainder that doesn't fill a complete sequence (+1 for target shift)
        n_tokens = len(token_ids)
        n_sequences = n_tokens // (seq_len + 1)
        self.data = token_ids[: n_sequences * (seq_len + 1)].reshape(n_sequences, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return full sequence; AutoregressiveWrapper handles input/target split
        return self.data[idx]


class WikiText103DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WikiText-103 causal LM.

    Args:
        seq_len: Sequence length for chunking.
        batch_size: Batch size.
        data_dir: Directory to cache downloaded/tokenized data.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        tokenizer_name: HuggingFace tokenizer name.
    """

    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 32,
        data_dir: str = "/ivi/zfs/s0/original_homes/dwessel/data",
        num_workers: int = 4,
        pin_memory: bool = True,
        tokenizer_name: str = "gpt2",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenizer_name = tokenizer_name

        # GPT-2 tokenizer vocab size
        self.vocab_size = 50257

    def prepare_data(self):
        """Download dataset and tokenizer (run on rank 0 only)."""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        cache_dir = str(self.data_dir / "wikitext103")
        tokenized_dir = self.data_dir / "wikitext103_tokenized"

        # Download raw dataset
        load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

        # Download tokenizer
        AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=str(self.data_dir / "tokenizers"),
        )

        # Tokenize and cache if not already done
        if not tokenized_dir.exists():
            tokenized_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=str(self.data_dir / "tokenizers"),
            )
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

            for split in ["train", "validation", "test"]:
                print(f"Tokenizing {split} split...")
                # Concatenate all text and tokenize
                texts = ds[split]["text"]
                all_ids = []
                for text in texts:
                    if text.strip():  # Skip empty lines
                        ids = tokenizer.encode(text)
                        all_ids.extend(ids)

                token_tensor = torch.tensor(all_ids, dtype=torch.long)
                torch.save(token_tensor, tokenized_dir / f"{split}.pt")
                print(f"  {split}: {len(all_ids):,} tokens saved")

    def setup(self, stage: Optional[str] = None):
        """Load tokenized data from disk."""
        tokenized_dir = self.data_dir / "wikitext103_tokenized"

        if stage == "fit" or stage is None:
            train_ids = torch.load(tokenized_dir / "train.pt", weights_only=True)
            val_ids = torch.load(tokenized_dir / "validation.pt", weights_only=True)
            self.train_ds = WikiText103Dataset(train_ids, self.seq_len)
            self.val_ds = WikiText103Dataset(val_ids, self.seq_len)
            print(f"WikiText-103 train: {len(self.train_ds):,} sequences of length {self.seq_len}")
            print(f"WikiText-103 val: {len(self.val_ds):,} sequences of length {self.seq_len}")

        if stage == "test" or stage is None:
            test_ids = torch.load(tokenized_dir / "test.pt", weights_only=True)
            self.test_ds = WikiText103Dataset(test_ids, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Convert tensor to dict format expected by AutoregressiveWrapper.

        The batch is a tensor of shape [B, seq_len + 1].
        AutoregressiveWrapper._prepare_batch will split into input/target.
        """
        return {
            "input": batch,
            "label": batch,
            "condition": None,
        }
