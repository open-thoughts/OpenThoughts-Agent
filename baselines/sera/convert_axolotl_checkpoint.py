"""
Convert a checkpoint with _checkpoint_wrapped_module in weight names to standard format.

This fixes checkpoints saved with PyTorch gradient checkpointing enabled, which adds
'_checkpoint_wrapped_module.' to layer names. vLLM and other inference engines expect
standard weight names without this prefix.

Usage:
    ./convert_checkpoint.py /path/to/input/model /path/to/output/model
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file
from tqdm import tqdm


def fix_key(key: str) -> str:
    """Remove _checkpoint_wrapped_module. from weight key names."""
    return key.replace("._checkpoint_wrapped_module.", ".")


def convert_checkpoint(input_dir: Path, output_dir: Path) -> None:
    """Convert checkpoint to standard weight names."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all safetensor files
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {input_dir}")

    print(f"Converting {len(safetensor_files)} safetensor files...")

    # Convert each safetensor file
    for sf_path in tqdm(safetensor_files, desc="Converting"):
        tensors = load_file(sf_path)

        # Rename keys
        new_tensors = {fix_key(k): v for k, v in tensors.items()}

        # Save to output
        output_path = output_dir / sf_path.name
        save_file(new_tensors, output_path)

    # Update the index file if it exists
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        print("Updating model.safetensors.index.json...")
        with open(index_path) as f:
            index = json.load(f)

        # Fix keys in weight_map
        if "weight_map" in index:
            index["weight_map"] = {
                fix_key(k): v for k, v in index["weight_map"].items()
            }

        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Copy other files (config, tokenizer, etc.)
    print("Copying other files...")
    for file_path in input_dir.iterdir():
        if file_path.suffix == ".safetensors":
            continue  # Already converted
        if file_path.name == "model.safetensors.index.json":
            continue  # Already updated

        dest = output_dir / file_path.name
        if file_path.is_file():
            shutil.copy2(file_path, dest)

    print(f"Done! Converted checkpoint saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint with _checkpoint_wrapped_module to standard format"
    )
    parser.add_argument("input_dir", type=Path, help="Input model directory")
    parser.add_argument("output_dir", type=Path, help="Output model directory")

    args = parser.parse_args()
    convert_checkpoint(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()