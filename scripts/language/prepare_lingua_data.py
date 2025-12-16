import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def prepare_lingua_data(dataset_name, output_dir, split="train", chunk_size=10000, subset=None):
    """
    Downloads/loads a HF dataset and converts it to Lingua's JSONL chunk format.

    Format: output_dir/source_name/source_name.chunk.XX.jsonl
    """
    print(f"Loading dataset {dataset_name} (split={split})...")
    # Load dataset
    # If subset is provided (e.g. for testing), use it
    if subset:
        dataset = load_dataset(dataset_name, split=f"{split}[:{subset}]")
    else:
        dataset = load_dataset(dataset_name, split=split)

    source_name = dataset_name.split("/")[-1].lower().replace("-", "_")
    output_path = Path(output_dir) / source_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {output_path}...")

    chunk_idx = 0
    file_handle = None

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        if i % chunk_size == 0:
            if file_handle:
                file_handle.close()

            filename = f"{source_name}.chunk.{chunk_idx:02d}.jsonl"
            file_handle = open(output_path / filename, "w")
            chunk_idx += 1

        # Lingua expects {"text": ...} or {"content": ...}
        # We assume the dataset has a 'text' column.
        if "text" in item:
            line = json.dumps({"text": item["text"]})
        elif "content" in item:
            line = json.dumps({"text": item["content"]})
        else:
            # Fallback: dump the whole item
            line = json.dumps(item)

        file_handle.write(line + "\n")

    if file_handle:
        file_handle.close()

    print(f"Done. Created {chunk_idx} chunks in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Zyphra/Zyda-2", help="HF dataset name")
    parser.add_argument("--output_dir", type=str, default="data/lingua_zyda", help="Output root directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Lines per chunk")
    parser.add_argument("--subset", type=int, default=None, help="Number of examples to process (for testing)")

    args = parser.parse_args()

    prepare_lingua_data(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split=args.split,
        chunk_size=args.chunk_size,
        subset=args.subset,
    )
