# train_pref.py
"""
Fine-tunes PreferenceEstimator on the Deal-or-No-Dialog dataset.

Usage

python train_pref.py \
       --model_out checkpoints/pref_estimator.pt \
       --epochs 3 \
       --batch_size 32 \
       --lr 3e-5 \
       --device cuda:0            # or "cpu"
"""

import argparse, math, os, json, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dond_data import load_dond            # loader you created
from preference import PreferenceEstimator # tiny encoder-heads net


# Helpers
def make_dataset(split="train"):
    samples = load_dond(split)
    texts, y_you, y_them = [], [], []
    for s in samples:
        # use full dialog; you can truncate to last N turns if preferred
        texts.append(" ".join(s.turns))
        # normalise value vectors to a probability simplex
        v_you  = torch.tensor(s.my_values, dtype=torch.float)
        v_them = torch.tensor(s.partner_values, dtype=torch.float)
        y_you.append(v_you / v_you.sum())
        y_them.append(v_them / v_them.sum())
    return texts, torch.stack(y_you), torch.stack(y_them), len(s.my_values)


def collate(batch, tok, max_len=256, device="cpu"):
    txts, y1, y2 = zip(*batch)
    enc = tok(list(txts), padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        torch.stack(y1).to(device),
        torch.stack(y2).to(device)
    )



# Training loop

def train(args):
    print("Loading data …")
    texts, tgt_you, tgt_them, n_items = make_dataset("train")
    val_texts, val_you, val_them, _ = make_dataset("validation")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = PreferenceEstimator(n_items, base=args.backbone).to(args.device)

    # data loaders 
    train_dl = DataLoader(
        list(zip(texts, tgt_you, tgt_them)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, device=args.device)
    )
    val_dl = DataLoader(
        list(zip(val_texts, val_you, val_them)),
        batch_size=args.batch_size,
        collate_fn=lambda b: collate(b, tokenizer, device=args.device)
    )

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    criterion = torch.nn.KLDivLoss(reduction="batchmean")  # softmax preds v. true probs

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, attn, y1, y2 in train_dl:
            opt.zero_grad()
            p1, p2 = model(ids, attn)           # already softmaxed
            loss = criterion(p1.log(), y1) + criterion(p2.log(), y2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item()

        avg_train = running / len(train_dl)
        #  validation 
        model.eval(); running = 0.0
        with torch.no_grad():
            for ids, attn, y1, y2 in val_dl:
                p1, p2 = model(ids, attn)
                loss = criterion(p1.log(), y1) + criterion(p2.log(), y2)
                running += loss.item()
        avg_val = running / len(val_dl)
        print(f"Epoch {epoch:02d} | train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "n_items": n_items,
                "backbone": args.backbone
            }, args.model_out)
            print(f"  ↳ new best → saved to {args.model_out}")

    print("Done. Best val KL:", best_val)



# Entrypoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_out", default="checkpoints/pref_estimator.pt")
    args = p.parse_args()
    train(args)
