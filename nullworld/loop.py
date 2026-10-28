"""A replayable research loop over a registered design space.

The loop is a deterministic function of (panel, firm-level treatment matrix,
firm-level moderator matrix). Null worlds and planted worlds are produced by
passing permuted or augmented inputs; the loop code is identical in every
world, which is what makes loop-level randomization inference valid.

Registry-driven grammar (registry/design_space.yaml):
  outcomes x treatments x controls x fixed effects x samples x transforms x
  clustering, visited lexicographically; moderator search inside each base
  specification.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fe import FEStructure, ClusterOLS

CRIT = 1.959963984540054


def winsorize_by_year(x: np.ndarray, year: np.ndarray, lo=0.01, hi=0.99) -> np.ndarray:
    out = x.copy()
    for y in np.unique(year):
        m = (year == y) & np.isfinite(x)
        if m.sum() < 20:
            continue
        a, b = np.quantile(x[m], [lo, hi])
        out[m] = np.clip(x[m], a, b)
    return out


@dataclass
class Grammar:
    outcomes: list            # names
    outcome_dir: list         # +1 / -1 (registered H1 sign on each outcome)
    treatments: list          # names of firm-level treatment columns
    moderators: list          # names of firm-level moderator columns
    mod_sign: list            # +1 amplifies, -1 attenuates
    controls: dict            # {'C0': [...], 'C1': [...], 'C2': [...]} row-level columns
    fes: dict                 # {'FE1': ['firm','year'], 'FE2': ['firm','naics3_year']}
    samples: dict             # {'S1': mask_col, ...} boolean row-level columns
    transforms: list          # ['W0','W1']
    clusters: dict            # {'CL1': 'firm', 'CL2': 'naics3'}
    max_triples: int = 10

    def base_specs(self):
        return list(itertools.product(range(len(self.outcomes)), range(len(self.treatments)),
                                      list(self.controls), list(self.fes), list(self.samples),
                                      self.transforms, list(self.clusters)))


class Frame:
    """Rows, fixed effects and cached demeaned columns for one (sample, FE)."""

    def __init__(self, df: pd.DataFrame, mask: np.ndarray, fe_cols, cluster_cols: dict, g: Grammar):
        self.idx = np.flatnonzero(mask)
        sub = df.iloc[self.idx]
        self.firm_idx = sub["firm_idx"].to_numpy()
        self.year = sub["year"].to_numpy()
        self.post = sub["post"].to_numpy().astype(float)
        self.fe = FEStructure([sub[c].to_numpy() for c in fe_cols])
        self.cl = {k: ClusterOLS(sub[v].to_numpy()) for k, v in cluster_cols.items()}
        self.n = len(self.idx)
        self.g = g
        # outcomes and controls (depend on transform)
        self.Y, self.Ctrl = {}, {}
        for w in g.transforms:
            Y = np.column_stack([sub[o].to_numpy(dtype=float) for o in g.outcomes])
            if w == "W1":
                Y = np.column_stack([winsorize_by_year(Y[:, j], self.year) for j in range(Y.shape[1])])
            self.Y[w] = self.fe.demean(Y)
            for cset, cols in g.controls.items():
                if not cols:
                    self.Ctrl[(w, cset)] = np.zeros((self.n, 0))
                    continue
                C = np.column_stack([sub[c].to_numpy(dtype=float) for c in cols])
                if w == "W1":
                    C = np.column_stack([winsorize_by_year(C[:, j], self.year)
                                         if np.unique(C[:, j]).size > 20 else C[:, j] for j in range(C.shape[1])])
                self.Ctrl[(w, cset)] = self.fe.demean(C)
        self.sub_outcomes_raw = np.column_stack([sub[o].to_numpy(dtype=float) for o in g.outcomes])
        # Frisch-Waugh-Lovell: orthonormal basis of the demeaned controls for each
        # (transform, control set); focal columns and outcomes are partialled out once.
        self.Q = {}
        for key, C in self.Ctrl.items():
            if C.shape[1] == 0:
                self.Q[key] = None
            else:
                q, r = np.linalg.qr(C)
                keep = np.abs(np.diag(r)) > 1e-10 * max(1.0, np.abs(np.diag(r)).max())
                self.Q[key] = q[:, keep]
        self.kc = {key: (0 if q is None else q.shape[1]) for key, q in self.Q.items()}
        self._resid_outcomes()

    def _resid(self, key, X):
        q = self.Q[key]
        return X if q is None else X - q @ (q.T @ X)

    def _resid_outcomes(self):
        self.Yr = {(w, c): self._resid((w, c), self.Y[w]) for (w, c) in self.Q}

    def set_worlds(self, E: np.ndarray, M: np.ndarray, Y_add: np.ndarray | None = None):
        """E: firms x n_treat, M: firms x n_mod (standardized firm-level values).
        Y_add: optional row-level additions to outcomes (planted worlds), aligned to
        the full panel; outcomes are re-demeaned when supplied."""
        e = E[self.firm_idx]            # n x T
        m = M[self.firm_idx]            # n x K
        ep = e * self.post[:, None]
        mp = m * self.post[:, None]
        T, K = e.shape[1], m.shape[1]
        epm = (ep[:, :, None] * m[:, None, :]).reshape(self.n, T * K)
        Z = self.fe.demean(np.column_stack([ep, mp, epm]))
        self.Zr = {}
        for key in self.Q:
            R = self._resid(key, Z)
            self.Zr[key] = (R[:, :T], R[:, T:T + K], R[:, T + K:].reshape(self.n, T, K))
        if Y_add is not None:
            add = Y_add[self.idx]
            for w in self.g.transforms:
                Y = self.sub_outcomes_raw + add
                if w == "W1":
                    Y = np.column_stack([winsorize_by_year(Y[:, j], self.year) for j in range(Y.shape[1])])
                self.Y[w] = self.fe.demean(Y)
            self._resid_outcomes()


@dataclass
class Report:
    success: bool = False
    first_success: dict | None = None
    best: dict | None = None
    T_first: float = -np.inf          # package statistic of the reported package (first success, else best)
    T_max: float = -np.inf            # best package statistic over the whole space
    h1_max: float = -np.inf           # max sign-consistent H1 t over base specs (reality-check statistic)
    h1_ts: list = field(default_factory=list)   # all H1 sign-consistent t (specification curve)
    h1_bs: list = field(default_factory=list)
    n_models: int = 0
    n_models_to_first: int = 0


class Loop:
    """Replayable loop. policy in {'U', 'Ufree'}; run_both evaluates both policies in one
    pass (identical results to two separate runs; shared H1 and single-moderator fits)."""

    def __init__(self, df: pd.DataFrame, g: Grammar):
        self.g = g
        self.df = df
        self.frames = {}
        for s, scol in g.samples.items():
            for f, fcols in g.fes.items():
                self.frames[(s, f)] = Frame(df, df[scol].to_numpy(bool), fcols, g.clusters, g)

    def set_world(self, E, M, Y_add=None):
        for fr in self.frames.values():
            fr.set_worlds(E, M, Y_add)

    def _fit(self, fr, w, cset, cl, y, cols):
        X = np.column_stack(cols)
        return fr.cl[cl].fit(y, X, k_extra=fr.kc[(w, cset)])

    def run(self, policy: str = "U", full: bool = True) -> Report:
        g, R = self.g, Report()
        signs = np.asarray(g.mod_sign)
        for (o, k, cset, fe, s, w, cl) in g.base_specs():
            fr = self.frames[(s, fe)]
            d = g.outcome_dir[o]
            y = fr.Yr[(w, cset)][:, o]
            EP, MP, EPM = fr.Zr[(w, cset)]
            res = self._fit(fr, w, cset, cl, y, [EP[:, [k]]])
            R.n_models += 1
            if res is None:
                continue
            b, se, _ = res
            t1 = d * b[0] / se[0]
            R.h1_ts.append(t1); R.h1_bs.append(b[0])
            R.h1_max = max(R.h1_max, t1)
            if t1 < CRIT or (R.success and not full):
                continue
            # single moderators
            singles = []
            for j in range(len(g.moderators)):
                r = self._fit(fr, w, cset, cl, y, [EP[:, [k]], EPM[:, k, [j]], MP[:, [j]]])
                R.n_models += 1
                if r is None:
                    continue
                tj = d * r[0][1] / r[1][1]
                if policy == "U":
                    tj_s = tj * signs[j]
                    if tj_s >= CRIT:
                        singles.append((j, tj_s, np.sign(signs[j])))
                else:
                    if abs(tj) >= CRIT:
                        singles.append((j, abs(tj), np.sign(tj)))
            if len(singles) < 3:
                continue
            triples = [c for c in itertools.combinations(singles, 3) if len({x[2] for x in c}) > 1]
            triples.sort(key=lambda c: -sum(x[1] for x in c))
            for c in triples[: g.max_triples]:
                js = [x[0] for x in c]
                cols = [EP[:, [k]]] + [EPM[:, k, [j]] for j in js] + [MP[:, [j]] for j in js]
                r = self._fit(fr, w, cset, cl, y, cols)
                R.n_models += 1
                if r is None:
                    continue
                bb, ss = r[0], r[1]
                th1 = d * bb[0] / ss[0]
                tm = []
                for q, (j, _, sg) in enumerate(c):
                    tq = d * bb[1 + q] / ss[1 + q]
                    if policy == "U":
                        tm.append(tq * signs[j])
                    else:
                        tm.append(tq * sg)   # must keep the sign seen singly
                T = min([th1] + tm)
                pkg = dict(outcome=g.outcomes[o], treatment=g.treatments[k], controls=cset, fe=fe,
                           sample=s, transform=w, cluster=cl, moderators=[g.moderators[j] for j in js],
                           t_h1=th1, t_mods=tm, b=bb[:4].tolist(), se=ss[:4].tolist(), T=T)
                if T > R.T_max:
                    R.T_max, R.best = T, pkg
                if T >= CRIT and not R.success:
                    R.success, R.first_success, R.T_first = True, pkg, T
                    R.n_models_to_first = R.n_models
                    if not full:
                        return R
        if not R.success:
            R.T_first = R.T_max
        return R

    def run_both(self, full: bool = True, cache: dict | None = None, record: dict | None = None,
                 recompute: tuple = ()):
        """Evaluate policies U and Ufree in one pass over the design space.

        cache/record: exact reuse of fits that cannot change across worlds. In a Type B
        world only moderator(s) `recompute` are permuted, so the H1 fits and the single-
        moderator fits of all other moderators equal those in the real data; `record`
        collects them on the real data and `cache` replays them. Triples are always
        refitted. Results are identical to an uncached run (validated in scripts/92)."""
        g = self.g
        signs = np.asarray(g.mod_sign)
        reps = {"U": Report(), "Ufree": Report()}
        n_models = 0
        for (o, k, cset, fe, s, w, cl) in g.base_specs():
            fr = self.frames[(s, fe)]
            d = g.outcome_dir[o]
            y = fr.Yr[(w, cset)][:, o]
            EP, MP, EPM = fr.Zr[(w, cset)]
            key = (o, k, cset, fe, s, w, cl)
            if cache is not None and key in cache:
                b0, t1 = cache[key]["b0"], cache[key]["t1"]
            else:
                res = self._fit(fr, w, cset, cl, y, [EP[:, [k]]])
                if res is None:
                    for R in reps.values():
                        R.n_models += 1
                    continue
                b0, t1 = res[0][0], d * res[0][0] / res[1][0]
            n_models += 1
            for R in reps.values():
                R.n_models += 1
            if record is not None:
                record[key] = {"b0": b0, "t1": t1, "tj": {}}
            b = [b0]
            for R in reps.values():
                R.h1_ts.append(t1); R.h1_bs.append(b[0]); R.h1_max = max(R.h1_max, t1)
            if t1 < CRIT:
                continue
            need = {pol: (full or not R.success) for pol, R in reps.items()}
            if not any(need.values()):
                continue
            tj_all = {}
            for j in range(len(g.moderators)):
                if cache is not None and key in cache and j not in recompute and j in cache[key]["tj"]:
                    tj_all[j] = cache[key]["tj"][j]
                    for pol in reps:
                        if need[pol]:
                            reps[pol].n_models += 1
                    continue
                r = self._fit(fr, w, cset, cl, y, [EP[:, [k]], EPM[:, k, [j]], MP[:, [j]]])
                for pol in reps:
                    if need[pol]:
                        reps[pol].n_models += 1
                if r is not None:
                    tj_all[j] = d * r[0][1] / r[1][1]
            if record is not None:
                record[key]["tj"] = dict(tj_all)
            for pol, R in reps.items():
                if not need[pol]:
                    continue
                if pol == "U":
                    singles = [(j, tj * signs[j], np.sign(signs[j])) for j, tj in tj_all.items() if tj * signs[j] >= CRIT]
                else:
                    singles = [(j, abs(tj), np.sign(tj)) for j, tj in tj_all.items() if abs(tj) >= CRIT]
                if len(singles) < 3:
                    continue
                triples = [c for c in itertools.combinations(singles, 3) if len({x[2] for x in c}) > 1]
                triples.sort(key=lambda c: -sum(x[1] for x in c))
                for c in triples[: g.max_triples]:
                    js = [x[0] for x in c]
                    cols = [EP[:, [k]]] + [EPM[:, k, [j]] for j in js] + [MP[:, [j]] for j in js]
                    r = self._fit(fr, w, cset, cl, y, cols)
                    R.n_models += 1
                    if r is None:
                        continue
                    bb, ss = r[0], r[1]
                    th1 = d * bb[0] / ss[0]
                    tm = []
                    for q, (j, _, sg) in enumerate(c):
                        tq = d * bb[1 + q] / ss[1 + q]
                        tm.append(tq * signs[j] if pol == "U" else tq * sg)
                    T = min([th1] + tm)
                    pkg = dict(outcome=g.outcomes[o], treatment=g.treatments[k], controls=cset, fe=fe,
                               sample=s, transform=w, cluster=cl, moderators=[g.moderators[j] for j in js],
                               t_h1=th1, t_mods=tm, b=bb[:4].tolist(), se=ss[:4].tolist(), T=T)
                    if T > R.T_max:
                        R.T_max, R.best = T, pkg
                    if T >= CRIT and not R.success:
                        R.success, R.first_success, R.T_first = True, pkg, T
                        R.n_models_to_first = R.n_models
                        if not full:
                            break
        for R in reps.values():
            if not R.success:
                R.T_first = R.T_max
        return reps

