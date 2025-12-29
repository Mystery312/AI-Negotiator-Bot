"""scripts/simulate_dond.py  (v7 – Pareto vs No-Pareto + new *statusquo* baseline)

Run two bots on the same subset of the Deal-or-No-Deal *validation* split:

1. **Pareto-bot** – calls `pareto.best_offer` to choose a Nash-product point
2. **No-Pareto**   – keeps the zero allocation (i.e. never proposes)

Both are compared against a baseline split that you choose with `--baseline`.
A trial is a *success* if **each** negotiator’s utility is at least
`ratio × baseline_utility` (use `--opp_ratio` to relax or tighten).

Examples

```
python scripts/simulate_dond.py --n 100 --baseline equal
python scripts/simulate_dond.py --baseline walkaway
python scripts/simulate_dond.py --baseline statusquo --opp_ratio 0.95
```
Each prints two lines, e.g.:
```
Pareto-bot ≥ equal baseline in 78% of 100 samples
No-Pareto   ≥ equal baseline in  3% of 100 samples
```"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Sequence, Optional

# Flexible imports for package/script modes
try:
    from app.dond_data import load_dond
    from app.pareto import best_offer, utility
    from app.preference import estimate_preferences
except ImportError:  # running as script
    from dond_data import load_dond
    from pareto import best_offer, utility
    from preference import estimate_preferences


# Baseline allocations 


def equal_split(counts: Dict[str, int]) -> Dict[str, int]:
    """Give each side half of every item (integer division)."""
    return {k: q // 2 for k, q in counts.items()}


def greedy(counts: Dict[str, int]) -> Dict[str, int]:
    """Opponent gets *nothing*; you (baseline) get everything."""
    return counts.copy()


def walkaway(counts: Dict[str, int]) -> Dict[str, int]:
    """Bot keeps nothing, opponent keeps everything – initial state of DOND."""
    return {k: 0 for k in counts}


def statusquo(counts: Dict[str, int]) -> Dict[str, int]:
    """Fair 50-50 split (alias for equal but called out explicitly)."""
    return {k: q // 2 for k, q in counts.items()}

BASELINES: dict[str, Callable[[Dict[str, int]], Dict[str, int]]] = {
    "equal": equal_split,
    "greedy": greedy,
    "walkaway": walkaway,
    "statusquo": statusquo,
}


# Misc helpers 


EPS = 1e-9  # avoid edge cases in ≥ checks

def _norm(ws: Sequence[float]) -> list[float]:
    """Normalise a weight vector to sum to 1 (clip negatives)."""
    ws = [max(0.0, float(x)) for x in ws]
    s = sum(ws) or 1.0
    return [w / s for w in ws]


# Evaluation functions -

def _utilities_for_split(
    counts: Dict[str, int],
    split_you: Dict[str, int],
    w_you: Sequence[float],
    w_them: Sequence[float],
) -> tuple[float, float]:
    """Return (u_you, u_them) for the given allocation."""
    u_you = utility(split_you, w_you)
    opp_split = {k: counts[k] - split_you[k] for k in counts}
    u_them = utility(opp_split, w_them)
    return u_you, u_them


def _is_success(
    u_you: float,
    u_them: float,
    base_you: float,
    base_them: float,
    ratio: float,
) -> bool:
    """Both parties must reach ratio × baseline utility."""
    return (
        u_you  + EPS >= ratio * base_you and
        u_them + EPS >= ratio * base_them
    )


def _proposal_for_bot(
    counts: Dict[str, int],
    w_you: Sequence[float],
    w_them: Sequence[float],
    base_you: float,
    base_them: float,
    ratio: float,
    use_pareto: bool,
) -> Dict[str, int]:
    """Return the bot's split (proposal)."""
    if not use_pareto:
        return {k: 0 for k in counts}  # bot stays with nothing

    prop = best_offer(
        counts=counts,
        w_you=w_you,
        w_them=w_them,
        last_split={k: 0 for k in counts},  # previous allocation (all zero)
        base_you=base_you,
        base_them=base_them,
        ratio=ratio,
    )
    return prop or {k: 0 for k in counts}


def evaluate_one(
    sample,
    baseline_fn: Callable[[Dict[str, int]], Dict[str, int]],
    ratio: float,
    use_pareto: bool,
) -> Optional[bool]:
    """Return success flag or None if we can’t estimate prefs."""
    counts = {f"item{i}": q for i, q in enumerate(sample.counts)}

    # estimate preferences
    w_you, w_them = estimate_preferences(sample.turns)
    if w_you is None or w_them is None:
        return None
    w_you, w_them = _norm(w_you), _norm(w_them)

    # baseline utilities
    base_split = baseline_fn(counts)
    base_you, base_them = _utilities_for_split(counts, base_split, w_you, w_them)

    # bot proposal
    proposal = _proposal_for_bot(counts, w_you, w_them, base_you, base_them, ratio, use_pareto)
    u_you, u_them = _utilities_for_split(counts, proposal, w_you, w_them)

    return _is_success(u_you, u_them, base_you, base_them, ratio)


def evaluate_dataset(
    n: int,
    baseline: str,
    ratio: float,
    use_pareto: bool,
) -> tuple[float, int]:
    """Return (success_rate, evaluated_samples)."""
    data = load_dond("validation")[:n]
    baseline_fn = BASELINES[baseline]

    total = success_cnt = 0
    for s in data:
        res = evaluate_one(s, baseline_fn, ratio, use_pareto)
        if res is None:
            continue  # skip when pref estimation failed
        total += 1
        if res:
            success_cnt += 1
    return (success_cnt / total if total else 0.0, total)


# Coach-rescue simulation 


def simulate_with_coach(
    n: int,
    baseline: str,
    ratio: float,
) -> dict:
    """
    Evaluate how often a coach (using Pareto best_offer) can turn non-success
    cases for a "No-Pareto" bot into successes relative to the chosen baseline.

    This simulates a two-stage process for each sample:
      1) Try No-Pareto (keeps zero allocation). If it's already a success
         against the baseline threshold, mark as success_without_coach.
      2) If not a success, invoke the coach (best_offer) once to propose a
         Pareto-efficient split and re-check utilities. If it now meets the
         threshold, count as rescued_by_coach.

    Returns a dictionary with summary statistics.
    """
    data = load_dond("validation")[:n]
    baseline_fn = BASELINES[baseline]

    total = len([s for s in data if s is not None])
    success_without_coach = 0
    rescued_by_coach = 0
    evaluated = 0
    transcripts: list[dict] = []

    for s in data:
        counts = {f"item{i}": q for i, q in enumerate(s.counts)}

        # estimate preferences
        w_you, w_them = estimate_preferences(s.turns)
        if w_you is None or w_them is None:
            continue
        w_you, w_them = _norm(w_you), _norm(w_them)

        # baseline utilities
        base_split = baseline_fn(counts)
        base_you, base_them = _utilities_for_split(counts, base_split, w_you, w_them)

        # Stage 1: No-Pareto bot (zero allocation)
        no_pareto_split = {k: 0 for k in counts}
        u_you_np, u_them_np = _utilities_for_split(counts, no_pareto_split, w_you, w_them)
        evaluated += 1
        if _is_success(u_you_np, u_them_np, base_you, base_them, ratio):
            success_without_coach += 1
            # Build transcript with original dialogue only
            messages = []
            for i, t in enumerate(s.turns):
                if not t.strip():
                    continue
                role = "You" if i % 2 == 0 else "Them"
                messages.append({"role": role, "text": t.strip()})
            transcripts.append({
                "sample_index": len(transcripts),
                "rescued": False,
                "messages": messages,
            })
            continue

        # Stage 2: Coach proposes Pareto best_offer
        coach_prop = best_offer(
            counts=counts,
            w_you=w_you,
            w_them=w_them,
            last_split=no_pareto_split,
            base_you=base_you,
            base_them=base_them,
            ratio=ratio,
        ) or no_pareto_split
        u_you_c, u_them_c = _utilities_for_split(counts, coach_prop, w_you, w_them)
        # Build transcript including coach proposal
        messages = []
        for i, t in enumerate(s.turns):
            if not t.strip():
                continue
            role = "You" if i % 2 == 0 else "Them"
            messages.append({"role": role, "text": t.strip()})

        # Compose a readable proposal string
        if coach_prop:
            prop_str = ", ".join(f"{v} {k}" for k, v in coach_prop.items())
        else:
            prop_str = "(no change)"
        coach_text = f"Coach proposes Pareto-guided split: {prop_str}"
        messages.append({"role": "Coach", "text": coach_text})

        rescued = _is_success(u_you_c, u_them_c, base_you, base_them, ratio)
        if rescued:
            rescued_by_coach += 1
            messages.append({"role": "Coach", "text": "Outcome: rescued (meets threshold after proposal)."})
        else:
            messages.append({"role": "Coach", "text": "Outcome: still below threshold (not rescued)."})

        transcripts.append({
            "sample_index": len(transcripts),
            "rescued": rescued,
            "messages": messages,
        })

    return {
        "total": evaluated,
        "success_without_coach": success_without_coach,
        "rescued_by_coach": rescued_by_coach,
        "rescue_rate": (rescued_by_coach / (evaluated - success_without_coach)) if (evaluated - success_without_coach) > 0 else 0.0,
        "overall_success_with_coach": (success_without_coach + rescued_by_coach) / evaluated if evaluated else 0.0,
        "transcripts": transcripts,
    }


# CLI 



def main():
    parser = argparse.ArgumentParser(description="Pareto vs No-Pareto benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument(
        "--baseline",
        choices=list(BASELINES),
        default="equal",
        help="Baseline: equal | greedy | walkaway | statusquo",
    )
    parser.add_argument(
        "--opp_ratio",
        type=float,
        default=1.0,
        help="Slack ratio for success threshold (≤1).",
    )
    args = parser.parse_args()

    if not (0 < args.opp_ratio <= 1.0):
        raise ValueError("--opp_ratio must be in (0,1]")

    ratio = args.opp_ratio

    pct_p, total = evaluate_dataset(args.n, args.baseline, ratio, use_pareto=True)
    pct_np, _    = evaluate_dataset(args.n, args.baseline, ratio, use_pareto=False)

    slack = f" (ratio {ratio})" if ratio < 1.0 else ""
    print(
        f"Pareto-bot ≥ {args.baseline} baseline{slack} in {pct_p:.0%} of {total} samples",
        f"No-Pareto   ≥ {args.baseline} baseline{slack} in {pct_np:.0%} of {total} samples",
        sep="\n",
    )


if __name__ == "__main__":
    main()
