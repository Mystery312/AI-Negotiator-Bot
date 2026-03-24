"""N-party preference estimation.

Extends the 2-party ``PreferenceEstimator`` to support N parties.
For n_parties=2, falls back to the existing ``estimate_preferences()``.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class NPartyPreferenceEstimator(nn.Module):
    """Shared DistilBERT encoder with N linear heads (one per party)."""

    def __init__(self, n_parties: int, n_issues: int = 3, base: str = "distilbert-base-uncased"):
        super().__init__()
        self.n_parties = n_parties
        self.n_issues = n_issues
        self.enc = AutoModel.from_pretrained(base)
        hid = self.enc.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hid, n_issues) for _ in range(n_parties)])

    def forward(self, input_ids, attention_mask) -> List[torch.Tensor]:
        h = self.enc(input_ids, attention_mask).last_hidden_state[:, 0]  # [CLS]
        return [torch.softmax(head(h), -1) for head in self.heads]

    @classmethod
    def from_2party_model(
        cls,
        existing_model,  # PreferenceEstimator instance
        n_parties: int,
    ) -> "NPartyPreferenceEstimator":
        """Create an N-party model by copying encoder + first 2 heads from a 2-party model."""
        n_issues = existing_model.head_you.out_features
        model = cls(n_parties=n_parties, n_issues=n_issues)

        # Copy encoder weights
        model.enc.load_state_dict(existing_model.enc.state_dict())

        # Copy first two heads from existing model
        if n_parties >= 1:
            model.heads[0].load_state_dict(existing_model.head_you.state_dict())
        if n_parties >= 2:
            model.heads[1].load_state_dict(existing_model.head_them.state_dict())

        # Remaining heads (index >= 2) keep random init
        return model


# ── tokenizer (shared) ───────────────────────────────────────────────────────

_tokenizer: Optional[AutoTokenizer] = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _tokenizer


# ── inference ─────────────────────────────────────────────────────────────────

def estimate_n_party_preferences(
    turns: List[str],
    n_parties: int,
    model: Optional[NPartyPreferenceEstimator] = None,
) -> List[List[float]]:
    """Estimate preference weights for each of N parties.

    For ``n_parties == 2`` and no custom model, falls back to the existing
    2-party ``estimate_preferences()`` function.

    Args:
        turns: Recent conversation turn texts
        n_parties: Number of negotiating parties
        model: Optional pre-loaded NPartyPreferenceEstimator

    Returns:
        List of N weight vectors (each a list of floats summing to ~1.0)
    """
    if n_parties == 2 and model is None:
        # Delegate to the existing 2-party estimator
        try:
            from negotiation_chatbot.preference import estimate_preferences
        except ImportError:
            from preference import estimate_preferences
        w_me, w_opp = estimate_preferences(turns)
        return [w_me, w_opp]

    # N-party path
    if model is None:
        # Build an N-party model from the 2-party checkpoint
        try:
            from negotiation_chatbot.coach import pref_model as _2p_model
        except ImportError:
            from coach import pref_model as _2p_model

        if _2p_model is not None:
            model = NPartyPreferenceEstimator.from_2party_model(_2p_model, n_parties)
        else:
            # No model available — return uniform weights
            n_issues = 3
            return [[1.0 / n_issues] * n_issues for _ in range(n_parties)]

    model.eval()
    tokenizer = _get_tokenizer()
    text = " ".join(turns[-6:])
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        weight_tensors = model(enc["input_ids"], enc["attention_mask"])

    return [w.squeeze().tolist() for w in weight_tensors]
