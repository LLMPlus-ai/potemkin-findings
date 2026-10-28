"""Null-world and planted-world generators.

Type A (treatment null): the firm-level treatment vector(s) are permuted across
firms within strata (e.g., NAICS-2 sectors). All treatment proxies move
together under one permutation so their mutual correlation is preserved.
Everything else (outcomes, moderators, controls, panel structure, missingness,
serial correlation) is untouched. Under the sharp null that outcomes do not
depend on the treatment, and within-stratum exchangeability of the treatment,
the permuted data are exchangeable with the observed data.

Type B (moderator null for moderator k): column k of the firm-level moderator
matrix is permuted within strata defined by sector x treatment quintile,
keeping the treatment and all other moderators fixed (a conditional
randomization test in the sense of Candes et al. 2018).

Planted worlds: known effects are added to outcomes in a Type A world.
"""
from __future__ import annotations

import numpy as np


def world_seed(b: int, stream: int = 0) -> int:
    """Registered seed rule: world b uses 1000003*b + 17 (+ stream offset)."""
    return 1000003 * b + 17 + 7919 * stream


def permute_within(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return an index array p such that x[p] permutes x within each stratum."""
    strata = np.asarray(strata)
    p = np.arange(len(strata))
    for s in np.unique(strata):
        ix = np.flatnonzero(strata == s)
        p[ix] = rng.permutation(ix)
    return p


def type_a(E: np.ndarray, strata: np.ndarray, b: int) -> np.ndarray:
    rng = np.random.default_rng(world_seed(b, 1))
    return E[permute_within(strata, rng)]


def type_b(M: np.ndarray, k: int, strata: np.ndarray, b: int) -> np.ndarray:
    rng = np.random.default_rng(world_seed(b, 2 + k))
    Mb = M.copy()
    Mb[:, k] = M[permute_within(strata, rng), k]
    return Mb


def planted_outcome_shift(firm_idx, post, E_world, M, outcome_sd, outcome_dir,
                          beta_star: float, treat_col: int = 0,
                          gamma_star: float = 0.0, mods=(), mod_signs=()):
    """Row-level additions to each outcome column (n x n_outcomes).

    Effect sizes are in within-firm SD units of each outcome: the planted
    effect of a 1-SD treatment is beta_star * sd_o in the registered direction.
    """
    e = E_world[firm_idx, treat_col] * post
    add = np.outer(e * beta_star, np.asarray(outcome_sd) * np.asarray(outcome_dir))
    for j, s in zip(mods, mod_signs):
        add += np.outer(e * M[firm_idx, j] * gamma_star * s, np.asarray(outcome_sd) * np.asarray(outcome_dir))
    return add
