"""Thompson Sampling over negotiation strategies.

Maintains Beta(alpha, beta) priors for each strategy and learns from
deal acceptance / rejection outcomes.  Persists weights to disk as JSON.
Thread-safe via threading.Lock().
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecord:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    total_uses: int = 0
    total_accepts: int = 0
    total_rejects: int = 0


class StrategyLearner:
    """Thompson Sampling strategy selector with outcome learning."""

    def __init__(
        self,
        strategy_names: List[str],
        persist_path: str = "data/strategy_weights.json",
    ) -> None:
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._pending: Dict[str, str] = {}  # conv_id -> strategy_name

        # Initialise uniform Beta(1,1) priors
        self._records: Dict[str, StrategyRecord] = {
            name: StrategyRecord(name=name) for name in strategy_names
        }
        self._load()

    # -- selection -------------------------------------------------------------

    def select_strategy(self, triggered_strategies: List[str]) -> str:
        """Sample from Beta posteriors and pick the highest-scoring triggered strategy."""
        if not triggered_strategies:
            return "fallback"

        with self._lock:
            best_name = triggered_strategies[0]
            best_sample = -1.0
            for name in triggered_strategies:
                rec = self._records.get(name)
                if rec is None:
                    # Unknown strategy — use uniform prior
                    sample = random.betavariate(1.0, 1.0)
                else:
                    sample = random.betavariate(rec.alpha, rec.beta)
                if sample > best_sample:
                    best_sample = sample
                    best_name = name
            return best_name

    # -- outcome tracking ------------------------------------------------------

    def record_advice_given(self, conv_id: str, strategy_name: str) -> None:
        """Record which strategy was used for a conversation (pending outcome)."""
        with self._lock:
            self._pending[conv_id] = strategy_name
            rec = self._records.get(strategy_name)
            if rec:
                rec.total_uses += 1

    def record_outcome(self, conv_id: str, accepted: bool) -> None:
        """Update Beta parameters based on deal outcome."""
        with self._lock:
            strategy_name = self._pending.pop(conv_id, None)
            if strategy_name is None:
                logger.debug("No pending strategy for conv_id=%s", conv_id)
                return
            rec = self._records.get(strategy_name)
            if rec is None:
                logger.debug("Unknown strategy %s", strategy_name)
                return
            if accepted:
                rec.alpha += 1.0
                rec.total_accepts += 1
            else:
                rec.beta += 1.0
                rec.total_rejects += 1
            logger.info(
                "Strategy %s updated: alpha=%.1f beta=%.1f (accepted=%s)",
                strategy_name, rec.alpha, rec.beta, accepted,
            )
        self._save()

    # -- monitoring ------------------------------------------------------------

    def get_weights(self) -> Dict[str, dict]:
        with self._lock:
            return {name: asdict(rec) for name, rec in self._records.items()}

    # -- persistence -----------------------------------------------------------

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            with self._lock:
                data = {name: asdict(rec) for name, rec in self._records.items()}
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save strategy weights: %s", exc)

    def _load(self) -> None:
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            with self._lock:
                for name, vals in data.items():
                    if name in self._records:
                        self._records[name] = StrategyRecord(**vals)
                    else:
                        self._records[name] = StrategyRecord(**vals)
            logger.info("Loaded strategy weights from %s", self._persist_path)
        except Exception as exc:
            logger.warning("Failed to load strategy weights: %s", exc)
