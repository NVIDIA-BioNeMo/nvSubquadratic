"""Script to download Zyda-2 dataset."""

import os
from experiments.datamodules.zyda_datamodule import ZydaDataModule

def download_zyda():
    """Download Zyda-2 dataset."""
    print("Initializing ZydaDataModule for download...")
    # streaming=False triggers download in prepare_data
    datamodule = ZydaDataModule(
        dataset_name="Zyphra/Zyda-2",
        tokenizer_name="nvidia/Mistral-NeMo-Minitron-8B-Base",
        streaming=False, 
        num_workers=30 # Increase workers for faster download/processing if applicable
    )
    
    print("Starting download (this may take a long time)...")
    datamodule.prepare_data()
    print("Download complete.")

if __name__ == "__main__":
    download_zyda()
