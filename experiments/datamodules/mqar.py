"""Multi-Query Associative Recall (MQAR) DataModule.

Generates synthetic sequences following the Zoology paper (Arora, Eyuboglu et al., ICLR 2024).

Sequence format:
    Inputs: [k1, v1, k2, v2, ..., kN, vN, <noise>, q_i, <noise>, q_j, <noise>, ...]
    Labels: [-100, ..., -100, v_i, -100, ..., v_j, -100, ...]

The model must recall which value was associated with each queried key.
Attention can solve this perfectly; convolution/SSM models struggle with many queries.
"""

import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MQARDataset(Dataset):
    """Procedural MQAR dataset.

    Args:
        num_examples: Number of examples to generate.
        vocab_size: Vocabulary size (must be > seq_len).
        seq_len: Sequence length.
        num_kv_pairs: Number of key-value pairs per sequence.
        power_a: Power law parameter controlling query spacing (0.01 = clustered, 1.0 = uniform).
        random_non_queries: If True, fill non-query positions with random tokens.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_examples: int,
        vocab_size: int = 8192,
        seq_len: int = 256,
        num_kv_pairs: int = 8,
        power_a: float = 0.01,
        random_non_queries: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert seq_len % 2 == 0, "seq_len must be even"
        assert vocab_size > seq_len, "vocab_size must be > seq_len"
        assert num_kv_pairs * 4 <= seq_len, "Need seq_len >= 4 * num_kv_pairs (kv context + query space)"

        self.num_examples = num_examples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs

        # Generate all examples upfront (fast, fits in RAM for reasonable sizes)
        self.inputs, self.labels = self._generate(
            num_examples=num_examples,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_kv_pairs=num_kv_pairs,
            power_a=power_a,
            random_non_queries=random_non_queries,
            seed=seed,
        )

    @staticmethod
    def _generate(
        num_examples: int,
        vocab_size: int,
        seq_len: int,
        num_kv_pairs: int,
        power_a: float,
        random_non_queries: bool,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate MQAR data following Zoology's implementation (vectorized)."""
        rng = np.random.default_rng(seed)

        context_size = num_kv_pairs * 2  # key-value pairs

        # Split vocab: first half for keys, second half for values
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        # Vectorized sampling: generate random permutations and take first num_kv_pairs
        # This replaces the slow np.apply_along_axis(np.random.choice, ...) calls
        key_perms = np.array([rng.permutation(key_choices)[:num_kv_pairs] for _ in range(num_examples)])
        value_perms = np.array([rng.permutation(value_choices)[:num_kv_pairs] for _ in range(num_examples)])

        # Create key-value context: [k1, v1, k2, v2, ...]
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0::2] = key_perms
        kvs[:, 1::2] = value_perms

        # Compute power law distribution for query spacing
        space = (seq_len - context_size) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()

        # Vectorized query position sampling with power law spacing
        gaps = np.array([rng.choice(space, size=num_kv_pairs, replace=False, p=p) for _ in range(num_examples)])

        # Build full sequences with queries placed at gap positions
        queries = np.zeros((num_examples, seq_len - context_size + 1), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=key_perms, axis=1)
        examples = np.concatenate([kvs, queries], axis=1)

        # Build labels: only query-answer positions are labeled
        labels = np.full((num_examples, seq_len + 1), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=value_perms, axis=1)

        inputs = torch.tensor(examples[:, :-1])
        labels = torch.tensor(labels[:, 1:])

        # Replace zeros with random noise tokens
        if random_non_queries:
            mask = inputs == 0
            inputs[mask] = torch.randint(vocab_size, size=inputs.shape)[mask]

        return inputs, labels

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


class MQARDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MQAR.

    Args:
        vocab_size: Vocabulary size.
        seq_len: Sequence length.
        num_kv_pairs: Number of key-value pairs per sequence.
        num_train_examples: Number of training examples.
        num_val_examples: Number of validation examples.
        num_test_examples: Number of test examples.
        batch_size: Batch size.
        power_a: Power law parameter for query spacing.
        random_non_queries: Fill non-query positions with random tokens.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        seed: Random seed.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        seq_len: int = 256,
        num_kv_pairs: int = 8,
        num_train_examples: int = 100_000,
        num_val_examples: int = 5_000,
        num_test_examples: int = 5_000,
        batch_size: int = 64,
        power_a: float = 0.01,
        random_non_queries: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples
        self.batch_size = batch_size
        self.power_a = power_a
        self.random_non_queries = random_non_queries
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """Generate train/val/test datasets with different seeds."""
        if stage == "fit" or stage is None:
            self.train_ds = MQARDataset(
                num_examples=self.num_train_examples,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_kv_pairs=self.num_kv_pairs,
                power_a=self.power_a,
                random_non_queries=self.random_non_queries,
                seed=self.seed,
            )
            self.val_ds = MQARDataset(
                num_examples=self.num_val_examples,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_kv_pairs=self.num_kv_pairs,
                power_a=self.power_a,
                random_non_queries=self.random_non_queries,
                seed=self.seed + 1,
            )
        if stage == "test" or stage is None:
            self.test_ds = MQARDataset(
                num_examples=self.num_test_examples,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_kv_pairs=self.num_kv_pairs,
                power_a=self.power_a,
                random_non_queries=self.random_non_queries,
                seed=self.seed + 2,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Convert (inputs, labels) tuple to dict format expected by AutoregressiveWrapper."""
        inputs, labels = batch
        return {
            "input": inputs,
            "label": labels,
            "condition": None,
        }
