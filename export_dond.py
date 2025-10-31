# export_dond.py
"""
Exports the Deal-or-No-Dialog dataset splits to local JSONL files.

Usage

python export_dond.py --out_dir exported
"""

import argparse, os
from datasets import load_dataset

def export_split(split: str, out_dir: str):
    ds = load_dataset(
        "mikelewis0/deal_or_no_dialog",
        split=split,
        trust_remote_code=True       # allow the script loader
    )
    path = os.path.join(out_dir, f"{split}.jsonl")
    ds.to_json(path, orient="records", lines=True)
    print(f"✓ wrote {path}  ({len(ds):,} records)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="exported",
                    help="directory to store train.jsonl, validation.jsonl, test.jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_split("train",       args.out_dir)
    export_split("validation",  args.out_dir)
    export_split("test",        args.out_dir)

if __name__ == "__main__":
    main()
