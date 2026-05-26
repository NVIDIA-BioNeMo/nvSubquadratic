import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_fidelity.metric_fid import fid_featuresdict_to_statistics
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached

from experiments.datamodules._deprecated.ref_imagenet import ImageNetDataModule


def _default_cache_dir() -> str:
    return (
        os.environ.get("IMAGENET_PATH")
        or os.environ.get("IMAGENET_CACHE")
        or os.environ.get("IMAGENET_CACHE_DIR")
        or str(Path.cwd() / "imagenet")
    )


class _FIDReadyDataset(Dataset):
    """Wrap ImageNetDataModule output and emit uint8 RGB tensors for torch-fidelity."""

    def __init__(self, base_dataset: Dataset, datamodule: ImageNetDataModule) -> None:
        self.base_dataset = base_dataset
        self.datamodule = datamodule

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.base_dataset[index]  # CHW, normalized for diffusion ([-1, 1])
        image = self.datamodule.unnormalize(image)  # CHW, [0, 1]
        image = torch.clamp(image * 255.0, 0.0, 255.0).to(torch.uint8)
        return image


def _build_reference_dataset(args: argparse.Namespace) -> Dataset:
    datamodule = ImageNetDataModule(
        data_dir=args.cache_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=args.seed,
        image_size=args.resize_image_size,
        final_image_size=args.image_size,
        center_crop=True,
        drop_labels=False,
        hf_dataset_name="imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=os.environ.get("HF_TOKEN"),
        task="generation",
    )

    if args.split == "train":
        datamodule.setup(stage="fit")
        if datamodule.train_dataset is None:
            raise RuntimeError("ImageNetDataModule did not initialize the training dataset.")
        base_dataset = datamodule.train_dataset
    else:
        datamodule.setup(stage="validate")
        if datamodule.val_dataset is None:
            raise RuntimeError("ImageNetDataModule did not initialize the validation dataset.")
        base_dataset = datamodule.val_dataset

    dataset: Dataset = _FIDReadyDataset(base_dataset, datamodule)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FID reference statistics using the exact ImageNetDataModule generation preprocessing pipeline."
        )
    )
    parser.add_argument("--cache_dir", type=str, default=_default_cache_dir())
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Final generated image size (ImageNetDataModule.final_image_size).",
    )
    parser.add_argument(
        "--resize_image_size",
        type=int,
        default=256,
        help="Pre-crop resize size used by generation datamodule (ImageNetDataModule.image_size).",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=64, help="Feature extraction batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed (affects train split flips).")
    parser.add_argument(
        "--output",
        type=str,
        default="examples/imagenet_diffusion/fid_stats/jit_in64_stats.npz",
        help="Output .npz path for mu/sigma statistics.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating FID statistics for split={args.split}, "
        f"image_size={args.image_size}, resize_image_size={args.resize_image_size}"
    )
    print(f"Using ImageNet cache dir: {args.cache_dir}")

    dataset = _build_reference_dataset(args)
    print(f"Processing {len(dataset)} images...")

    use_cuda = torch.cuda.is_available()
    feature_extractor = create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=use_cuda,
        verbose=True,
    )
    featuresdict = extract_featuresdict_from_input_id_cached(
        1,
        feature_extractor,
        input1=dataset,
        cuda=use_cuda,
        batch_size=args.batch_size,
        verbose=True,
    )

    stats = fid_featuresdict_to_statistics(featuresdict, "2048")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, mu=stats["mu"], sigma=stats["sigma"])
    print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()
