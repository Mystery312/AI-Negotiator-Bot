# app/dond_data.py  (overwrite previous version)

from dataclasses import dataclass
from typing import List
import json, os
import re

ROOT = os.getenv("DOND_DATA_DIR", "deal_or_no_dialog/exported")
SPLIT_FILE = {
    "train":       "train.jsonl",
    "validation":  "validation.jsonl",   # ← correct name
    "test":        "test.jsonl"
}

def get_root_path() -> str:
    """Get the root path with fallbacks."""
    root = os.getenv("DOND_DATA_DIR", "deal_or_no_dialog/exported")
    
    # Check if root path exists
    if os.path.exists(root):
        print(f"Found data directory at primary path: {os.path.abspath(root)}")
        return root
        
    # Try alternative paths
    alt_paths = [
        "deal_or_no_dialog/exported",
        os.path.join("..", "deal_or_no_dialog", "exported"),
        os.path.join("app", "deal_or_no_dialog", "exported"),
        "data",
        os.path.join("app", "data")
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            print(f"Found data directory at alternative path: {os.path.abspath(path)}")
            return path
            
    # Return original path if nothing found (will cause FileNotFoundError later)
    return root

# Update ROOT path
ROOT = get_root_path()

@dataclass
class DialogSample:
    turns: List[str]
    counts: List[int]
    my_values: List[int]
    partner_values: List[int]
    my_final: List[int]
    partner_final: List[int]

def _parse_output(out: str):
    """
    Works for both formats:
      "1 0 2 0 1 1"
      "item0=1 item1=0 item2=2 item0=0 item1=1 item2=1"
    Returns (my_alloc, partner_alloc) as integer lists.
    """
    nums = [int(n) for n in re.findall(r"\d+", out)]
    k = len(nums) // 2
    return nums[:k], nums[k:]

def load_dond(split: str = "train") -> List[DialogSample]:
    path = os.path.join(ROOT, SPLIT_FILE[split])
    
    # Debug: Print the path being checked
    print(f"Looking for data file at: {os.path.abspath(path)}")
    
    if not os.path.exists(path):
        # Try alternative paths
        alt_paths = [
            os.path.join("deal_or_no_dialog", "exported", SPLIT_FILE[split]),
            os.path.join("..", "deal_or_no_dialog", "exported", SPLIT_FILE[split]),
            os.path.join("data", SPLIT_FILE[split]),
            os.path.join("app", "data", SPLIT_FILE[split])
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Found data file at alternative path: {os.path.abspath(alt_path)}")
                path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"{path} not found. Make sure you ran deal_or_no_dialog.py "
                "and/or set DOND_DATA_DIR to the directory that contains "
                "train.jsonl, dev.jsonl, test.jsonl. "
                f"Also checked: {alt_paths}"
            )

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            my_alloc, partner_alloc = _parse_output(d["output"])
            # Split dialogue on <eos> and clean up
            dialogue_turns = d["dialogue"].split(" <eos> ")
            # Remove any remaining <selection> markers and clean whitespace
            clean_turns = []
            for turn in dialogue_turns:
                turn = turn.replace("<selection>", "").strip()
                if turn:  # Only add non-empty turns
                    clean_turns.append(turn)
            
            samples.append(
                DialogSample(
                    turns=clean_turns,
                    counts=d["input"]["count"],
                    my_values=d["input"]["value"],
                    partner_values=d["partner_input"]["value"],
                    my_final=my_alloc,
                    partner_final=partner_alloc
                )
            )
    return samples
