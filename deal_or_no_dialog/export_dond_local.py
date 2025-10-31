# export_dond_local.py   (inside the cloned deal_or_no_dialog folder)
import os, datasets

splits = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",   # only this spelling exists
    "test": "test.jsonl"
}

for split, out_name in splits.items():
    ds = datasets.load_dataset(
        path="deal_or_no_dialog.py",   # dataset script in the folder
        split=split,
        trust_remote_code=True
    )
    os.makedirs("exported", exist_ok=True)
    out_path = os.path.join("exported", out_name)
    ds.to_json(out_path, orient="records", lines=True)
    print(f"✓ wrote {out_path}  ({len(ds):,} rows)")
