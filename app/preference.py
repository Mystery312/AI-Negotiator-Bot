# app/preference.py
import os
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple

class PreferenceEstimator(nn.Module):
    """
    One-line BERT encoder → 2×linear heads producing softmax
    weight vectors for you & partner (dim = #issues)
    """
    def __init__(self, n_issues, base="distilbert-base-uncased"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        hid = self.enc.config.hidden_size
        self.head_you  = nn.Linear(hid, n_issues)
        self.head_them = nn.Linear(hid, n_issues)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids, attention_mask).last_hidden_state[:,0]   # [CLS]
        return (
            torch.softmax(self.head_you(h),  -1),
            torch.softmax(self.head_them(h), -1)
        )

#  helpers called from coach.py 
_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
_pref_model = PreferenceEstimator(n_issues=3)     # apples, bananas, pears, etc.

def load_pref_model(ckpt_path: str | None = None, device: str = "cpu") -> Tuple[PreferenceEstimator, AutoTokenizer]:
    """Load the preference model and tokenizer, move to device, and load checkpoint if provided."""
    model = _pref_model.to(device)
    model.eval()
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            # Direct state_dict or wrapped
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
    return model, _tokenizer

def estimate_preferences(turns: list[str]) -> tuple[list[float], list[float]]:
    text = " ".join(turns[-6:])                       # last ≤6 utterances
    tok = _tokenizer(text, return_tensors="pt", truncation=True)
    # Ensure tensors are on the same device as the model
    device = next(_pref_model.parameters()).device
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        w_you, w_them = _pref_model(**tok)
    return w_you.squeeze().tolist(), w_them.squeeze().tolist()
