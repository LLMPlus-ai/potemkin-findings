"""23 - Monte Carlo: what happens to loop-level inference when exposure is NOT exchangeable within sector.

Every replication draws a synthetic twin (nullworld/synth.py) with NO effect of exposure, then adds a
firm trait u whose within-sector correlation with exposure is rho:
    u = rho * e_w + sqrt(1 - rho^2) * z,   e_w = within-NAICS-2 standardized exposure, z ~ N(0, 1).
The trait reaches the outcome (experienced-hire share) through one channel:
  level : + 1.0 * sd_between * u          (time-invariant; absorbed by firm fixed effects)
  post  : + delta * sd_within * u * Post  (a post-shock shift that exposure merely proxies for)
Reduced registered grammar as in scripts/13 (one outcome, 3 exposure proxies, C0/C1, W0/W1: 12 base
specifications); the statistic is the largest sign-consistent H1 t over the search.

Procedures (rejection at 5%):
  NWC        : loop-level p from B within-sector permutations of exposure (the application's Type A worlds)
  NWC-strat  : permutations within NAICS-2 x quintile of u (u observed)
  NWC-cov    : covariate-preserving worlds: exposure = projection on (sector, u) + permuted residual
  single     : one pre-specified regression (first proxy, C1, W0), t >= 1.96
  naive loop : largest t over the 12 specifications >= 1.96
  Bonferroni : largest t >= z(1 - 0.05/12)
usage: python3 scripts/23_mc_exchangeability.py <cell> R B [workers]
       cells: null | level | post_r{0.2,0.4,0.6}_d{0.025,0.05,0.1,0.2}
Output: intermediate/23_mc/<cell>_<r>.json
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats

_S = {}
C1 = ["log_hc_l1", "hire_rate_l1", "sep_rate_l1", "log_at_l1", "roa_l1", "lev_l1", "sales_growth_l1", "rd_at_l1"]
PRESPEC = 2          # base-spec index of (first proxy, C1, W0) in the lexicographic order


def _init():
    import pickle
    from nullworld import registry as R
    _S["reg"] = R.load(ROOT / "registry/design_space.yaml")
    _S["params"] = pickle.loads((ROOT / "synthetic/twin_params.pkl").read_bytes())


def grammar(reg):
    from nullworld.loop import Grammar
    return Grammar(outcomes=["tilt_exp"], outcome_dir=[1], treatments=[t["name"] + "_z" for t in reg["treatments"]],
                   moderators=[], mod_sign=[], controls={"C0": [], "C1": C1}, fes={"FE1": ["firm_idx", "year"]},
                   samples={"S1": "S1"}, transforms=["W0", "W1"], clusters={"CL1": "firm_idx"})


def parse(cell):
    if cell == "null":
        return "post", 0.0, 0.0
    if cell == "level":
        return "level", 0.6, 1.0
    _, r, d = cell.split("_")
    return "post", float(r[1:]), float(d[1:])


def within_std(x, g):
    s = pd.Series(x).groupby(g)
    w = x - s.transform("mean").to_numpy()
    return w / w.std()


def work(args):
    cell, r, B = args
    out = ROOT / "intermediate/23_mc" / f"{cell}_{r:04d}.json"
    if out.exists():
        return None
    from nullworld import synth
    from nullworld.loop import Loop
    from nullworld.worlds import permute_within
    from nullworld.infer import rank_p
    reg, params = _S["reg"], _S["params"]
    channel, rho, delta = parse(cell)
    p, firms = synth.generate(params, seed=200000 + r)
    firms = firms.sort_values("firm_idx").reset_index(drop=True)
    n2 = firms.naics2.to_numpy()
    rng = np.random.default_rng(900000 + r)
    ew = within_std(firms["expo_beta_z"].to_numpy(), n2)
    u = rho * ew + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(firms))
    rp = params["rowp"]["tilt_exp"]
    if channel == "level":
        p["tilt_exp"] = p.tilt_exp + delta * rp["sd_between"] * u[p.firm_idx]
    else:
        p["tilt_exp"] = p.tilt_exp + delta * rp["sd_within"] * u[p.firm_idx] * p.post
    p["S1"] = True
    g = grammar(reg)
    E = firms[g.treatments].to_numpy(float)
    M = np.zeros((len(firms), 0))
    loop = Loop(p, g); loop.set_world(E, M)
    rep = loop.run("U", full=True)
    h1, t_single, b_single = rep.h1_max, rep.h1_ts[PRESPEC], rep.h1_bs[PRESPEC]
    # null-world constructions
    uq = pd.qcut(pd.Series(u).rank(method="first"), 5, labels=False).to_numpy()
    strat_u = n2.astype(int) * 10 + uq
    D = pd.get_dummies(n2.astype(int)).to_numpy(float)
    Z = np.column_stack([D, u])
    fit = Z @ np.linalg.lstsq(Z, E, rcond=None)[0]
    res = E - fit
    null = {"NWC": [], "NWC_strat": [], "NWC_cov": []}
    for b in range(B):
        rb = np.random.default_rng(50000 + r * 1000 + b)
        for name, Eb in [("NWC", E[permute_within(n2, rb)]), ("NWC_strat", E[permute_within(strat_u, rb)]),
                         ("NWC_cov", fit + res[permute_within(n2, rb)])]:
            loop.set_world(Eb, M)
            null[name].append(loop.run("U", full=True).h1_max)
    crit_b = stats.norm.ppf(1 - 0.05 / len(rep.h1_ts))
    resd = dict(cell=cell, channel=channel, rho=rho, delta=delta, r=r, B=B, h1=h1, t_single=t_single,
                b_single_sdw=b_single / rp["sd_within"],
                p={k: rank_p(h1, np.array(v)) for k, v in null.items()},
                reject=dict(single=bool(t_single >= 1.96), naive=bool(h1 >= 1.96), bonf=bool(h1 >= crit_b)))
    for k in null:
        resd["reject"][k] = bool(resd["p"][k] <= 0.05)
    out.write_text(json.dumps(resd, default=float))
    return r


def main():
    cell, R_, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    (ROOT / "intermediate/23_mc").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(nw, initializer=_init) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, [(cell, r, B) for r in range(R_)])):
            if (i + 1) % 50 == 0:
                print(f"{cell}: {i+1}/{R_} in {time.time()-t0:.0f}s", flush=True)
    print("DONE", cell, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
