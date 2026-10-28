"""Synthetic twin of a firm-year panel.

Purpose: (1) public reproducibility when the licensed microdata cannot be
shared (the call's FAQ and the OS transparency policy both name synthetic
data as the alternative); (2) Monte Carlo worlds with KNOWN truth for
validating the size and power of loop-level inference.

Generative model (fitted to the real panel; no real row is copied):
  firm level : Gaussian copula over treatments, moderators and pre-period
               covariates (normal-score correlation matrix + empirical
               marginal quantile functions evaluated at fresh uniforms);
               sector drawn from its empirical distribution conditional on the
               treatment quintile (preserves the treatment-industry link that
               stratified permutation relies on);
  observation: each synthetic firm borrows the year-coverage pattern of a
               randomly drawn real firm;
  outcomes / time-varying controls:
               x_ft = mu + sd_between * a_f + tau_t + s_{g(f),t} + e_ft,
               e_ft = rho * e_f,t-1 + sqrt(1-rho^2) * sd_within * z_ft,
               with rho, sd_between, sd_within, year effects and sector-year
               shock SD estimated from the real panel; firm effects a_f are
               correlated with pre-period size.
  effects    : optional planted H1 and moderator effects (beta, gammas) in
               within-SD units, in the registered direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _normal_scores(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return stats.norm.ppf(r / (len(x) + 1))


def fit(panel: pd.DataFrame, firms: pd.DataFrame, firm_cols, row_cols):
    F = firms.sort_values("firm_idx").reset_index(drop=True)
    Zs = np.column_stack([_normal_scores(F[c].to_numpy()) for c in firm_cols])
    corr = np.corrcoef(Zs, rowvar=False)
    quantiles = {c: np.sort(F[c].to_numpy()) for c in firm_cols}
    e0 = F[firm_cols[0]].to_numpy()
    quint = pd.qcut(pd.Series(e0).rank(method="first"), 5, labels=False).to_numpy()
    sector_given_q = {q: F.loc[quint == q, "naics3"].to_numpy() for q in range(5)}
    coverage = [np.sort(g.year.to_numpy()) for _, g in panel.groupby("firm_idx")]
    rowp = {}
    for c in row_cols:
        d = panel[["firm_idx", "year", "naics2", c]].dropna().sort_values(["firm_idx", "year"])
        fm = d.groupby("firm_idx")[c].transform("mean")
        w = d[c] - fm
        yr = w.groupby(d.year).mean()
        w2 = w - d.year.map(yr)
        sy = w2.groupby([d.naics2, d.year]).transform("mean")
        e = w2 - sy
        lag = e.groupby(d.firm_idx).shift(1)
        ok = lag.notna()
        rho = float(np.corrcoef(e[ok], lag[ok])[0, 1]) if ok.sum() > 100 else 0.0
        rowp[c] = dict(mu=float(d[c].mean()), sd_between=float(d.groupby("firm_idx")[c].mean().std()),
                       sd_within=float(e.std()), rho=max(min(rho, 0.95), -0.5),
                       year_fx=yr.to_dict(), sd_sector_year=float(sy.std()),
                       lo=float(d[c].quantile(0.001)), hi=float(d[c].quantile(0.999)))
    return dict(firm_cols=list(firm_cols), row_cols=list(row_cols), corr=corr, quantiles=quantiles,
                sector_given_q=sector_given_q, coverage=coverage, rowp=rowp, n_firms=len(F))


def generate(params: dict, seed: int, n_firms: int | None = None, effects: dict | None = None,
             outcome_dir: dict | None = None):
    rng = np.random.default_rng(seed)
    nf = n_firms or params["n_firms"]
    fc = params["firm_cols"]
    L = np.linalg.cholesky(params["corr"] + 1e-9 * np.eye(len(fc)))
    Z = rng.standard_normal((nf, len(fc))) @ L.T
    U = stats.norm.cdf(Z)
    firms = pd.DataFrame({c: np.quantile(params["quantiles"][c], U[:, j]) for j, c in enumerate(fc)})
    firms["firm_idx"] = np.arange(nf)
    q = pd.qcut(pd.Series(firms[fc[0]]).rank(method="first"), 5, labels=False).to_numpy()
    firms["naics3"] = [rng.choice(params["sector_given_q"][qq]) for qq in q]
    firms["naics2"] = firms.naics3 // 10
    cov = params["coverage"]
    rows = []
    for f in range(nf):
        for y in cov[rng.integers(len(cov))]:
            rows.append((f, y))
    p = pd.DataFrame(rows, columns=["firm_idx", "year"])
    p = p.merge(firms[["firm_idx", "naics2", "naics3"]], on="firm_idx")
    p["post"] = (p.year >= 2023).astype(int)
    p["naics3_year"] = p.naics3 * 10000 + p.year
    p = p.sort_values(["firm_idx", "year"]).reset_index(drop=True)
    size = firms.get("pre_log_hc")
    size_z = ((size - size.mean()) / size.std()).to_numpy() if size is not None else np.zeros(nf)
    sectors_years = {(s, y): None for s, y in zip(p.naics2, p.year)}
    for c, rp in params["rowp"].items():
        a = 0.5 * size_z + np.sqrt(0.75) * rng.standard_normal(nf)
        sy = {k: rng.normal(0, rp["sd_sector_year"]) for k in sectors_years}
        e = np.empty(len(p))
        prev_f, prev_e = -1, 0.0
        z = rng.standard_normal(len(p))
        for i, (f, y) in enumerate(zip(p.firm_idx.to_numpy(), p.year.to_numpy())):
            if f != prev_f:
                prev_e = z[i] * rp["sd_within"]
            else:
                prev_e = rp["rho"] * prev_e + np.sqrt(1 - rp["rho"] ** 2) * rp["sd_within"] * z[i]
            e[i] = prev_e
            prev_f = f
        yfx = p.year.map(rp["year_fx"]).fillna(0).to_numpy()
        syv = np.array([sy[(s, y)] for s, y in zip(p.naics2, p.year)])
        p[c] = rp["mu"] + rp["sd_between"] * a[p.firm_idx] + yfx + syv + e
    if effects:
        for c, eff in effects.items():
            rp = params["rowp"][c]
            d = (outcome_dir or {}).get(c, 1)
            ep = firms[eff["treatment"]].to_numpy()[p.firm_idx] * p.post.to_numpy()
            add = eff.get("beta", 0.0) * ep
            for mcol, gam in eff.get("gammas", {}).items():
                add = add + gam * ep * firms[mcol].to_numpy()[p.firm_idx]
            p[c] = p[c] + d * rp["sd_within"] * add
    return p, firms
