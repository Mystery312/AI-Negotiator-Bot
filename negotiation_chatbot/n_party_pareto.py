"""N-party Pareto solver with coalition support.

Generalises the 2-party ``pareto.py`` to N players.  For exactly 2 parties
with no coalitions the implementation delegates to the existing
``pareto.best_offer()`` to preserve identical behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from negotiation_chatbot.pareto import best_offer as _2p_best_offer, utility as _2p_utility
except ImportError:
    from pareto import best_offer as _2p_best_offer, utility as _2p_utility


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class Party:
    id: str
    name: str
    weights: List[float]
    coalition_id: Optional[str] = None


@dataclass
class Coalition:
    id: str
    member_ids: List[str]
    joint_weights: List[float] = field(default_factory=list)


# ── utility helpers ───────────────────────────────────────────────────────────

def n_party_utility(allocation: Dict[str, int], weights: Sequence[float]) -> float:
    """Compute utility for one party given its allocation and weights."""
    return sum(q * w for q, w in zip(allocation.values(), weights))


def _split_allocation(
    counts: Dict[str, int],
    splits: Tuple[Dict[str, int], ...],
) -> Dict[str, int]:
    """Compute the remainder allocation (last party gets what's left)."""
    remainder = dict(counts)
    for s in splits:
        for k, v in s.items():
            remainder[k] -= v
    return remainder


# ── enumeration ───────────────────────────────────────────────────────────────

def enumerate_n_party_allocations(
    counts: Dict[str, int], n_parties: int
) -> List[Tuple[Dict[str, int], ...]]:
    """Yield all ways to split *counts* among *n_parties*.

    Each result is a tuple of n_parties dicts.  The last party receives the
    remainder, so we only enumerate the first (n-1) parties' allocations.
    """
    keys = list(counts.keys())
    per_item_ranges = [range(counts[k] + 1) for k in keys]

    if n_parties == 1:
        return [(dict(counts),)]

    # For 2 parties, delegate to simple enumeration for efficiency
    if n_parties == 2:
        results = []
        for split in product(*per_item_ranges):
            alloc_0 = dict(zip(keys, split))
            alloc_1 = {k: counts[k] - alloc_0[k] for k in keys}
            if all(v >= 0 for v in alloc_1.values()):
                results.append((alloc_0, alloc_1))
        return results

    # General case: recursive
    results: list[Tuple[Dict[str, int], ...]] = []

    def _recurse(
        remaining: Dict[str, int],
        depth: int,
        partial: list[Dict[str, int]],
    ):
        if depth == n_parties - 1:
            # Last party gets whatever remains
            last = dict(remaining)
            if all(v >= 0 for v in last.values()):
                results.append(tuple(partial + [last]))
            return
        per_item = [range(remaining[k] + 1) for k in keys]
        for split in product(*per_item):
            alloc = dict(zip(keys, split))
            new_remaining = {k: remaining[k] - alloc[k] for k in keys}
            if all(v >= 0 for v in new_remaining.values()):
                _recurse(new_remaining, depth + 1, partial + [alloc])

    _recurse(dict(counts), 0, [])
    return results


# ── dominance & frontier ──────────────────────────────────────────────────────

def n_party_dominates(u1: Tuple[float, ...], u2: Tuple[float, ...]) -> bool:
    """True if u1 Pareto-dominates u2 (at least as good in all, strictly better in one)."""
    at_least = all(a >= b for a, b in zip(u1, u2))
    strictly = any(a > b for a, b in zip(u1, u2))
    return at_least and strictly


def n_party_pareto_frontier(
    counts: Dict[str, int],
    parties: List[Party],
    coalitions: Optional[List[Coalition]] = None,
) -> List[Tuple[Tuple[Dict[str, int], ...], Tuple[float, ...]]]:
    """Compute the N-dimensional Pareto frontier.

    Returns list of (allocations_tuple, utilities_tuple).
    """
    n = len(parties)
    # Resolve effective weights: if a party is in a coalition, use coalition weights
    effective_weights = []
    coalition_map = {}
    if coalitions:
        for c in coalitions:
            for mid in c.member_ids:
                coalition_map[mid] = c.joint_weights

    for p in parties:
        if p.id in coalition_map:
            effective_weights.append(coalition_map[p.id])
        else:
            effective_weights.append(p.weights)

    allocations = enumerate_n_party_allocations(counts, n)

    frontier: list[Tuple[Tuple[Dict[str, int], ...], Tuple[float, ...]]] = []

    for alloc_tuple in allocations:
        utils = tuple(
            n_party_utility(alloc_tuple[i], effective_weights[i])
            for i in range(n)
        )
        # Check if dominated by any existing frontier point
        if any(n_party_dominates(existing_u, utils) for _, existing_u in frontier):
            continue
        # Remove dominated points
        frontier = [
            (a, u) for a, u in frontier if not n_party_dominates(utils, u)
        ]
        frontier.append((alloc_tuple, utils))

    return frontier


# ── best offer selection ──────────────────────────────────────────────────────

def n_party_best_offer(
    counts: Dict[str, int],
    parties: List[Party],
    coalitions: Optional[List[Coalition]] = None,
    fairness: str = "nash",
) -> Optional[Tuple[Dict[str, int], ...]]:
    """Select the best N-party allocation from the Pareto frontier.

    For 2 parties with no coalitions, delegates to existing ``pareto.best_offer()``.

    Args:
        counts: Total items available  {item_id: quantity}
        parties: List of Party objects with preference weights
        coalitions: Optional coalition groupings
        fairness: Selection criterion — "nash", "egalitarian", or "utilitarian"

    Returns:
        Tuple of allocation dicts (one per party), or None.
    """
    n = len(parties)

    # Delegate to 2-party solver when applicable
    if n == 2 and not coalitions:
        try:
            equal_split = {k: v // 2 for k, v in counts.items()}
            base_you = _2p_utility(equal_split, parties[0].weights)
            base_them = _2p_utility(
                {k: counts[k] - v for k, v in equal_split.items()},
                parties[1].weights,
            )
            result = _2p_best_offer(
                counts,
                parties[0].weights,
                parties[1].weights,
                None,
                base_you,
                base_them,
                1.0,
            )
            if result:
                remainder = {k: counts[k] - v for k, v in result.items()}
                return (result, remainder)
        except Exception:
            pass
        return None

    frontier = n_party_pareto_frontier(counts, parties, coalitions)
    if not frontier:
        return None

    EPS = 1e-9

    if fairness == "nash":
        # Maximise product of utilities (Nash bargaining)
        best = max(frontier, key=lambda x: _product(x[1]))
    elif fairness == "egalitarian":
        # Maximise the minimum utility
        best = max(frontier, key=lambda x: min(x[1]))
    elif fairness == "utilitarian":
        # Maximise total utility
        best = max(frontier, key=lambda x: sum(x[1]))
    else:
        best = max(frontier, key=lambda x: _product(x[1]))

    return best[0]


def _product(values: Tuple[float, ...]) -> float:
    p = 1.0
    for v in values:
        p *= max(v, 1e-9)
    return p


# ── coalition helpers ─────────────────────────────────────────────────────────

def form_coalition(
    parties: List[Party], member_ids: List[str], coalition_id: str
) -> Coalition:
    """Form a coalition by averaging member weights."""
    members = [p for p in parties if p.id in member_ids]
    if not members:
        return Coalition(id=coalition_id, member_ids=member_ids, joint_weights=[])

    n_issues = len(members[0].weights)
    joint = [
        sum(m.weights[i] for m in members) / len(members)
        for i in range(n_issues)
    ]

    # Tag parties with coalition
    for p in parties:
        if p.id in member_ids:
            p.coalition_id = coalition_id

    return Coalition(id=coalition_id, member_ids=member_ids, joint_weights=joint)


def suggest_coalitions(parties: List[Party]) -> List[Coalition]:
    """Heuristic: suggest coalitions based on complementary preferences.

    Parties whose highest-priority items differ are good coalition candidates.
    """
    if len(parties) < 3:
        return []

    suggestions: list[Coalition] = []
    seen = set()

    for i, p1 in enumerate(parties):
        for j, p2 in enumerate(parties):
            if j <= i:
                continue
            pair = (p1.id, p2.id)
            if pair in seen:
                continue
            # Check if top priorities differ (complementary)
            top1 = max(range(len(p1.weights)), key=lambda k: p1.weights[k])
            top2 = max(range(len(p2.weights)), key=lambda k: p2.weights[k])
            if top1 != top2:
                cid = f"coalition_{p1.id}_{p2.id}"
                c = form_coalition(parties, [p1.id, p2.id], cid)
                suggestions.append(c)
                seen.add(pair)

    return suggestions
