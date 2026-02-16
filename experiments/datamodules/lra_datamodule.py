from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class _LRADataset(Dataset):
    """Generic LRA dataset wrapper for Hugging Face datasets."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x, y = item['input'], item['label']
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(y, dtype=torch.long)

class LRADataModule(pl.LightningDataModule):
    """Unified DataModule for Long Range Arena tasks."""

    def __init__(
        self,
        task: Literal["image", "text", "listops"],
        data_dir: str = "data/lra",
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 1024,
        seed: int = 42,
    ):
        super().__init__()
        self.task = task
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.seed = seed
        
        if task == "image":
            # Pixel-level CIFAR-10
            self.num_classes = 10
            self.input_channels = 3
        elif task == "text":
            # Byte-level IMDB
            self.num_classes = 2
            self.input_channels = 1
        elif task == "listops":
            self.num_classes = 10
            self.input_channels = 1
            
    def prepare_data(self):
        if self.task == "image":
            load_dataset("cifar10", cache_dir=str(self.data_dir))
        elif self.task == "text":
            load_dataset("imdb", cache_dir=str(self.data_dir))
        elif self.task == "listops":
            # Note: ListOps might require specific source if not on standard HF
            pass

    def setup(self, stage: Optional[str] = None):
        if self.task == "image":
            ds = load_dataset("cifar10", cache_dir=str(self.data_dir))
            
            def preprocess_image(examples):
                # Flatten 32x32x3 to (1024, 3) 
                # Actually ClassificationResNet expects [B, *spatial, C]
                # For LRA Image, sequence is 1024.
                images = [torch.tensor(img.convert("RGB")).permute(1, 2, 0).flatten(0, 1) / 255.0 for img in examples["img"]]
                return {"input": images, "label": examples["label"]}
            
            self.train_ds = ds["train"].with_transform(preprocess_image)
            self.val_ds = ds["test"].with_transform(preprocess_image) # Use test as val for simplicity in LRA
            
        elif self.task == "text":
            ds = load_dataset("imdb", cache_dir=str(self.data_dir))
            
            def preprocess_text(examples):
                # Byte-level encoding: encode text to bytes (0-255)
                # Pad/truncate to max_length
                encoded = []
                for text in examples["text"]:
                    b = list(text.encode("utf-8"))[:self.max_length]
                    b += [0] * (self.max_length - len(b))
                    encoded.append(torch.tensor(b, dtype=torch.float32).unsqueeze(-1) / 255.0)
                return {"input": encoded, "label": examples["label"]}
            
            self.train_ds = ds["train"].with_transform(preprocess_text)
            self.val_ds = ds["test"].with_transform(preprocess_text)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # Already formatted by with_transform
        return {
            "input": batch["input"],
            "label": batch["label"],
            "condition": None
        }
