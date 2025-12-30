#!/usr/bin/env python3
"""
Setup script for downloading and preparing the Deal-or-No-Deal dataset.

This script downloads the Facebook Deal-or-No-Deal dataset and prepares it
for use with the negotiation chatbot application.
"""

import os
import sys
import json
import urllib.request
import shutil
from pathlib import Path

# Dataset URLs from Facebook Research
DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/train.txt",
    "validation": "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/val.txt",
    "test": "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/test.txt"
}

# Alternative: Direct download links (if above don't work)
ALTERNATIVE_URLS = {
    "train": "https://dl.fbaipublicfiles.com/parlai/projects/negotiation/train.txt",
    "validation": "https://dl.fbaipublicfiles.com/parlai/projects/negotiation/val.txt",
    "test": "https://dl.fbaipublicfiles.com/parlai/projects/negotiation/test.txt"
}

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        file_size = os.path.getsize(destination)
        print(f"✓ Downloaded successfully ({file_size:,} bytes)")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def parse_dond_line(line):
    """Parse a Deal-or-No-Deal dataset line into JSONL format."""
    # The format is typically:
    # <input> count_0=X count_1=Y count_2=Z value_0=A value_1=B value_2=C </input>
    # <dialogue> turn1 <eos> turn2 <eos> ... </dialogue>
    # <output> item0=X item1=Y item2=Z item0=A item1=B item2=C </output>

    parts = line.strip().split('\t')
    if len(parts) < 3:
        return None

    try:
        # Parse input section
        input_text = parts[0]
        counts = []
        values = []
        for part in input_text.split():
            if part.startswith("count_"):
                counts.append(int(part.split('=')[1]))
            elif part.startswith("value_"):
                values.append(int(part.split('=')[1]))

        # Parse dialogue
        dialogue = parts[1].strip()

        # Parse output section
        output_text = parts[2] if len(parts) > 2 else ""
        output_items = []
        for part in output_text.split():
            if part.startswith("item"):
                output_items.append(int(part.split('=')[1]))

        # Create JSONL entry
        entry = {
            "dialogue": dialogue,
            "input": {
                "count": counts,
                "value": values
            },
            "partner_input": {
                "value": values  # Simplified - in real dataset, partner values may differ
            },
            "output": " ".join(map(str, output_items))
        }

        return entry
    except Exception as e:
        print(f"Warning: Failed to parse line: {e}")
        return None

def convert_to_jsonl(input_file, output_file):
    """Convert Deal-or-No-Deal format to JSONL."""
    print(f"\nConverting {input_file} to JSONL format...")

    if not os.path.exists(input_file):
        print(f"✗ Input file not found: {input_file}")
        return False

    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip():
                    entry = parse_dond_line(line)
                    if entry:
                        outfile.write(json.dumps(entry) + '\n')
                        count += 1

    print(f"✓ Converted {count} samples to {output_file}")
    return count > 0

def main():
    """Main setup function."""
    print("=" * 60)
    print("Deal-or-No-Deal Dataset Setup")
    print("=" * 60)
    print()

    # Create directory structure
    data_dir = Path("deal_or_no_dialog/exported")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {data_dir}")

    # Download and convert each split
    for split, url in DATASET_URLS.items():
        print(f"\n--- Processing {split} split ---")

        # Download raw file
        raw_file = data_dir / f"{split}.txt"
        jsonl_file = data_dir / f"{split}.jsonl"

        # Try primary URL first
        success = download_file(url, raw_file)

        # Try alternative URL if primary fails
        if not success and split in ALTERNATIVE_URLS:
            print(f"Trying alternative URL...")
            success = download_file(ALTERNATIVE_URLS[split], raw_file)

        if success:
            # Convert to JSONL
            if convert_to_jsonl(raw_file, jsonl_file):
                # Clean up raw file
                raw_file.unlink()
                print(f"✓ Cleaned up raw file")
            else:
                print(f"✗ Conversion failed")
        else:
            print(f"✗ Download failed for {split} split")

    # Verify setup
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    required_files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    all_present = True

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {filename}: {size:,} bytes")
        else:
            print(f"✗ {filename}: NOT FOUND")
            all_present = False

    print()
    if all_present:
        print("✅ Setup complete! All dataset files are ready.")
        print(f"\nDataset location: {data_dir.absolute()}")
        print("\nYou can now run the Gradio UI with DoND features enabled:")
        print("  python -m negotiation_chatbot.gradio_ui")
    else:
        print("⚠️  Setup incomplete. Some files are missing.")
        print("\nManual setup instructions:")
        print("1. Download the dataset from: https://github.com/facebookresearch/end-to-end-negotiator")
        print("2. Place train.jsonl, validation.jsonl, and test.jsonl in:")
        print(f"   {data_dir.absolute()}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
