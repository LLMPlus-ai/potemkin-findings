"""Loop-level randomization inference.

p_L = (1 + #{b : S(D^(b)) >= S(D)}) / (1 + B)

is valid (Pr(p_L <= alpha) <= alpha) for ANY deterministic statistic S of the
data, including the output of an adaptive search loop, provided the null
worlds D^(b) are exchangeable with D under the null (see the technical
appendix, Theorem 1). The +1 terms make the test exact rather than
approximately valid for finite B.
"""
from __future__ import annotations

import numpy as np


def rank_p(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    return (1.0 + np.sum(null >= obs)) / (1.0 + len(null))


def holm(pvals):
    """Holm step-down adjusted p-values (familywise error control)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def mc_band(p: float, B: int, z: float = 1.96):
    """Monte Carlo standard error band for a rejection rate estimated from B draws."""
    se = np.sqrt(p * (1 - p) / B)
    return p - z * se, p + z * se
