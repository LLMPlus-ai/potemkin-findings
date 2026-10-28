"""90 - Validate nullworld's FE/OLS engine against pyfixest on a simulated
unbalanced panel (coefficients and cluster-robust SEs must agree to 1e-8).
Output: intermediate/90_engine_validation.csv"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pyfixest as pf
from nullworld.fe import FEStructure, ClusterOLS

rng = np.random.default_rng(7)
nf, years = 1500, np.arange(2016, 2026)
rows = []
for i in range(nf):
    yrs = years[rng.random(len(years)) > 0.15]
    for t in yrs:
        rows.append((i, t))
df = pd.DataFrame(rows, columns=["firm", "year"])
df["ind"] = df.firm % 37
df["indyear"] = df.ind * 10000 + df.year
E = rng.normal(size=nf); M = rng.normal(size=(nf, 3))
df["post"] = (df.year >= 2023).astype(float)
df["ep"] = E[df.firm] * df.post
for j in range(3):
    df[f"epm{j}"] = df.ep * M[df.firm, j]; df[f"mp{j}"] = M[df.firm, j] * df.post
df["x1"] = rng.normal(size=len(df)); df["x2"] = rng.normal(size=len(df))
fe_f = rng.normal(size=nf)
df["y"] = fe_f[df.firm] + 0.1 * df.year + 0.3 * df.ep + 0.2 * df.epm0 + df.x1 + rng.normal(size=len(df))
out = []
for fes, clus in [(["firm", "year"], "firm"), (["firm", "indyear"], "firm"), (["firm", "year"], "ind")]:
    X = ["ep", "epm0", "epm1", "mp0", "mp1", "x1", "x2"]
    t0 = time.time()
    fe = FEStructure([df[c].to_numpy() for c in fes])
    Z = fe.demean(df[["y"] + X].to_numpy())
    b, se, e = ClusterOLS(df[clus].to_numpy()).fit(Z[:, 0], Z[:, 1:])
    t_ours = time.time() - t0
    m = pf.feols(f"y ~ {'+'.join(X)} | {'+'.join(fes)}", data=df, vcov={"CRV1": clus})
    bp, sep = m.coef().to_numpy(), m.se().to_numpy()
    out.append(dict(fe="+".join(fes), cluster=clus, max_abs_diff_b=np.abs(b - bp).max(),
                    max_abs_diff_se=np.abs(se - sep).max(), max_rel_diff_se=np.abs(se / sep - 1).max(),
                    seconds_ours=t_ours))
res = pd.DataFrame(out); print(res.to_string())
Path("intermediate").mkdir(exist_ok=True); res.to_csv("intermediate/90_engine_validation.csv", index=False)
