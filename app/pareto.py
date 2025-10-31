"""Pareto-utility helpers for Deal-or-No-Dialog

Changes v2

* **Frontier sweep with early exit** – return the first Pareto split that meets
  per-side slack targets (Fix A).
* **Dynamic slack (Fix B)** – the *weaker* side keeps the full `ratio` of its
  baseline utility; the stronger side is allowed an extra `DELTA` slack.
* Retains the equal-blend nudge as a final attempt.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple, Sequence

DELTA = 0.05  # extra slack allowed for the stronger side
EPS = 1e-9

# Exhaustive enumeration
def enumerate_allocations(counts: Dict[str, int]):
    keys, vals = zip(*counts.items())
    for split in product(*(range(v + 1) for v in vals)):
        yield dict(zip(keys, split))


def utility(split: Dict[str, int], weights: Sequence[float]) -> float:
    return sum(q * w for q, w in zip(split.values(), weights))

# Pareto frontier 


def _dominates(u1: Tuple[float, float], u2: Tuple[float, float]) -> bool:
    return (u1[0] >= u2[0] and u1[1] >= u2[1]) and (u1[0] > u2[0] or u1[1] > u2[1])


def pareto_frontier(counts: Dict[str, int], w_you: Sequence[float], w_them: Sequence[float]):
    frontier: List[Tuple[Dict[str, int], float, float]] = []
    for split in enumerate_allocations(counts):
        you_u = utility(split, w_you)
        them_u = utility({k: counts[k] - v for k, v in split.items()}, w_them)
        if any(_dominates((u, v), (you_u, them_u)) for _, u, v in frontier):
            continue
        # remove dominated points already stored
        frontier = [p for p in frontier if not _dominates((you_u, them_u), (p[1], p[2]))]
        frontier.append((split, you_u, them_u))
    return frontier


# Helper splits 


def _equal_split(counts: Dict[str, int]):
    return {k: q // 2 for k, q in counts.items()}


def _blend(a: Dict[str, int], b: Dict[str, int]):
    return {k: (a[k] + b[k]) // 2 for k in a}


# Main chooser --


def best_offer(
    counts: Dict[str, int],
    w_you: Sequence[float],
    w_them: Sequence[float],
    last_split: Dict[str, int] | None,
    base_you: float,
    base_them: float,
    ratio: float,
) -> Dict[str, int] | None:
    """Return a Pareto-efficient proposal satisfying per-side slack if possible.

    Parameters
    
    counts       : item totals
    w_you/w_them : preference weights (normalised)
    last_split   : last allocation you proposed (to avoid repeats)
    base_you/them: baseline utilities for each side (e.g. equal split)
    ratio        : minimal fraction of baseline the *weaker* side must keep
    """
    frontier = pareto_frontier(counts, w_you, w_them)
    if not frontier:
        return None

    # identify weaker vs stronger side wrt baseline
    if base_you <= base_them:
        low_target  = (ratio, ratio - DELTA)
    else:
        low_target  = (ratio - DELTA, ratio)

    # First, sweep every frontier point (Fix A) – natural order is fine.
    for split, u_you, u_them in frontier:
        tgt_you  = low_target[0] * base_you
        tgt_them = low_target[1] * base_them
        if u_you + EPS >= tgt_you and u_them + EPS >= tgt_them:
            if split != last_split:
                return split

    # If nothing met the slack, fall back to max-min fairness.
    best = max(frontier, key=lambda p: min(p[1], p[2]))
    split_best, u_you_best, u_opp_best = best

    # Optional equal-blend nudge (kept from v1)
    eq = _equal_split(counts)
    blended = _blend(split_best, eq)
    if blended != split_best and any(p[0] == blended for p in frontier):
        return blended if blended != last_split else None

    return None if split_best == last_split else split_best
