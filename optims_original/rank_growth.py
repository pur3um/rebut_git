"""
Rank growth curves for the progressive low-rank Muon optimizers.

Naming convention (see also optims/auto_rank_registry.py)
--------------------------------------------------------
curve token : cos | lin | log | exp | lgs
curve name  : cosine | linear | log | exp | logistic
module      : optims/auto_<token>_inc_rank.py
class       : SingleDeviceAuto<Token>IncWithAuxAdam
CLI         : --optimizer auto_<token>_inc_rank
mode string : "<curve name>_inc" / "<curve name>_dec" / "constant"

Every curve is expressed as a normalized progress function

    p : [0, 1] -> [0, 1],   p(0) = 0,  p(1) = 1,  p monotonically increasing

and the scheduled rank is

    rank(step) = round(rank_start + (rank_end - rank_start) * p(t)),
    t = (step - 1) / (horizon - 1).

Because only the progress function changes, "increase" and "decrease" are the
same code path: rank_start < rank_end grows the rank, rank_start > rank_end
shrinks it. The cosine progress is bit-for-bit the same expression that
auto_cos_inc_rank.py / auto_cos_inc_rank_ablation.py already use, so the cosine
baseline is unchanged.
"""

import math
from typing import Callable, Dict

# Default steepness / midpoint for the parametric curves.
DEFAULT_GROWTH_K: Dict[str, float] = {
    "cosine": 0.0,    # unused
    "linear": 0.0,    # unused
    "log": 9.0,       # log1p(9 t) / log1p(9): fast early, saturating
    "exp": 5.0,       # (e^{5t} - 1) / (e^5 - 1): slow early, fast late
    "logistic": 12.0,  # sigmoid steepness
}
DEFAULT_LOGISTIC_MIDPOINT = 0.5

CURVE_TOKENS: Dict[str, str] = {
    "cosine": "cos",
    "linear": "lin",
    "log": "log",
    "exp": "exp",
    "logistic": "lgs",
}


def cosine_progress(t: float, k: float = 0.0, midpoint: float = 0.5) -> float:
    """S-shaped, symmetric. The original schedule."""
    return 0.5 * (1.0 - math.cos(math.pi * t))


def linear_progress(t: float, k: float = 0.0, midpoint: float = 0.5) -> float:
    """Constant rank velocity."""
    return t


def log_progress(t: float, k: float = 9.0, midpoint: float = 0.5) -> float:
    """Concave: most of the rank is added early, then saturates."""
    k = float(k)
    if k <= 0.0:
        return t
    return math.log1p(k * t) / math.log1p(k)


def exp_progress(t: float, k: float = 5.0, midpoint: float = 0.5) -> float:
    """Convex: the rank stays low for a long time and then grows quickly."""
    k = float(k)
    if abs(k) < 1e-12:
        return t
    return (math.expm1(k * t)) / (math.expm1(k))


def logistic_progress(t: float, k: float = 12.0, midpoint: float = 0.5) -> float:
    """
    Sigmoid, renormalized so that p(0) = 0 and p(1) = 1.

    Compared with cosine this holds the rank near rank_start longer, then
    transitions faster around `midpoint`.
    """
    k = float(k)
    if k <= 0.0:
        return t

    t0 = min(max(float(midpoint), 0.0), 1.0)

    def _sigmoid(x: float) -> float:
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    lo = _sigmoid(k * (0.0 - t0))
    hi = _sigmoid(k * (1.0 - t0))
    span = hi - lo
    if abs(span) < 1e-12:
        return t
    return (_sigmoid(k * (t - t0)) - lo) / span


PROGRESS_FNS: Dict[str, Callable[[float, float, float], float]] = {
    "cosine": cosine_progress,
    "linear": linear_progress,
    "log": log_progress,
    "exp": exp_progress,
    "logistic": logistic_progress,
}


def get_progress_fn(curve: str) -> Callable[[float, float, float], float]:
    curve = str(curve).lower()
    if curve not in PROGRESS_FNS:
        raise ValueError(
            f"Unknown rank growth curve '{curve}'. "
            f"Available: {sorted(PROGRESS_FNS)}"
        )
    return PROGRESS_FNS[curve]


def default_growth_k(curve: str) -> float:
    return float(DEFAULT_GROWTH_K.get(str(curve).lower(), 0.0))


def get_scheduled_rank(
    curve: str,
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    growth_k: float = None,
    growth_midpoint: float = DEFAULT_LOGISTIC_MIDPOINT,
) -> int:
    """
    Curve-generic replacement for get_cosine_rank().

    The step/horizon edge cases are identical to get_cosine_rank() so that
    curve="cosine" reproduces the existing schedule exactly.
    """
    step = int(step)
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if warmup_steps <= 1:
        return end_rank
    if step <= 1:
        return start_rank
    if step >= warmup_steps:
        return end_rank

    if growth_k is None:
        growth_k = default_growth_k(curve)

    t = (step - 1) / float(warmup_steps - 1)
    progress = get_progress_fn(curve)(t, float(growth_k), float(growth_midpoint))
    progress = min(max(progress, 0.0), 1.0)

    rank = start_rank + (end_rank - start_rank) * progress
    return int(round(rank))


def describe_schedule(
    curve: str,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    growth_k: float = None,
    growth_midpoint: float = DEFAULT_LOGISTIC_MIDPOINT,
    n_samples: int = 11,
) -> str:
    """Small helper used by the fitting scripts to print the resolved schedule."""
    if growth_k is None:
        growth_k = default_growth_k(curve)

    horizon = max(1, int(warmup_steps))
    samples = []
    for i in range(int(n_samples)):
        frac = i / float(max(1, int(n_samples) - 1))
        step = 1 + int(round(frac * (horizon - 1)))
        samples.append(
            get_scheduled_rank(
                curve,
                step,
                start_rank,
                end_rank,
                horizon,
                growth_k=growth_k,
                growth_midpoint=growth_midpoint,
            )
        )

    return (
        f"[RankGrowth] curve={curve} start={start_rank} end={end_rank} "
        f"horizon={horizon} k={growth_k:g} midpoint={growth_midpoint:g} "
        f"ranks={samples}"
    )
