import os
import time
import argparse
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from experiments.datamodules.imagenet import ImageNetDataModule
from experiments.datamodules.imagenet_wds import ImageNetWebDataModule

def benchmark_loader(loader, name, num_batches=100, device="cuda"):
    print(f"\n--- Benchmarking {name} ---")
    
    iterator = iter(loader)
    
    print("Starting benchmark (this simulates training loops)...")
    start_time = time.time()
    
    batches_processed = 0
    samples_processed = 0
    
    for i in range(num_batches):
        try:
            # Measure time fetching batch
            fetch_start = time.time()
            batch = next(iterator)
            fetch_time = time.time() - fetch_start
        except StopIteration:
            print("Reached end of dataset before num_batches.")
            break
            
        # Parse batch format from different datamodules
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            # WebDataset
            x = batch[0]
            y = batch[1]
        else:
            x = batch
            
        # Move to GPU to simulate actual training data transfer
        if isinstance(x, torch.Tensor):
            x = x.to(device, non_blocking=True)
            samples_processed += x.shape[0]
            
        batches_processed += 1
        
        if batches_processed % 20 == 0:
            elapsed = time.time() - start_time
            print(f"[{batches_processed}/{num_batches}] "
                  f"{samples_processed / elapsed:.2f} images/sec (last batch fetch: {fetch_time:.3f}s)")
    
    total_time = time.time() - start_time
    img_per_sec = samples_processed / total_time
    
    print(f"\nResult for {name}:")
    print(f"Speed: {img_per_sec:.2f} images/sec")
    print(f"Total time for {samples_processed} images: {total_time:.2f}s")
    return img_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Force HF datasets offline to prevent Hub checks
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Requested batches per test: {args.num_batches}")
    
    # Standard transform pipeline mirroring training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    
    # 1. ImageFolder
    print("\n" + "="*50)
    print("Testing ImageFolder")
    print("="*50)
    try:
        if_dataset = ImageFolder("data/imagenet_folder/train", transform=transform)
        if_loader = DataLoader(
            if_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0
        )
        results["ImageFolder"] = benchmark_loader(if_loader, "ImageFolder", args.num_batches, device)
    except Exception as e:
        print(f"Error testing ImageFolder: {e}")
        
    # 2. WebDataset
    print("\n" + "="*50)
    print("Testing WebDataset")
    print("="*50)
    try:
        wds_dm = ImageNetWebDataModule(
            data_dir="data/imagenet-wds",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=42,
            task="classification",
            image_size=224,
            final_image_size=224
        )
        wds_dm.setup("fit")
        # Use a plain DataLoader instead of wds_dm.train_dataloader() which
        # wraps in wds.WebLoader + .with_epoch() — those are needed for
        # Lightning epoch tracking but cause len() errors in standalone use.
        wds_loader = DataLoader(
            wds_dm.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        results["WebDataset"] = benchmark_loader(wds_loader, "WebDataset", args.num_batches, device)
    except Exception as e:
        print(f"Error testing WebDataset: {e}")

    # 3. HF Arrow (ImageNetDataModule)
    print("\n" + "="*50)
    print("Testing HuggingFace Arrow (Original)")
    print("="*50)
    try:
        hf_dm = ImageNetDataModule(
            data_dir="data/imagenet",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=42,
            task="classification",
            image_size=224,
            final_image_size=224,
            hf_dataset_name="ILSVRC/imagenet-1k",  # Match the cache folder exactly
        )
        hf_dm.setup("fit")
        results["HF Arrow"] = benchmark_loader(hf_dm.train_dataloader(), "HuggingFace Arrow", args.num_batches, device)
    except Exception as e:
        print(f"Error testing HF Arrow: {e}")
        
    print("\n" + "*"*50)
    print("FINAL RESULTS (Images / Second)")
    print("*"*50)
    for name, speed in results.items():
        print(f"{name:25s}: {speed:>8.2f} img/s")
        
if __name__ == "__main__":
    main()
