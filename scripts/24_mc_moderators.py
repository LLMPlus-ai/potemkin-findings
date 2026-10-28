"""24 - Monte Carlo: do moderator null worlds test "no moderation" or something sharper?

Every replication draws a synthetic twin with a planted main effect (H1, 0.10 within-firm SD per SD of
exposure, so the loop's moderator search is active, as in the application). The target moderator is
hierarchy shape (H2). Its TRUE interaction with exposure is zero in every scenario; what varies is the
other structure the data contain:
  clean     : nothing else
  trend     : the target has its own post-shock trend, + 0.10 sd_w * M_k * Post
  proxy     : a correlated moderator (digitizable work; corr. -0.60 with M_k, as in the real data) has a
              true interaction, - 0.06 sd_w * E * M_j * Post
  nonlinear : the exposure effect is concave, - 0.05 sd_w * (E^2 - 1) * Post (M_k corr. -0.60 with E)
The proxy and nonlinear signs are chosen adversarially: given the negative correlations, both leak into
the target's interaction in its REGISTERED direction (amplifies), the case in which a test can over-reject.
  all       : trend + proxy + nonlinear
Statistics for the target (registered sign): S_free = max over the 12 base specifications with a
significant H1 of the single-moderator t; S_fix = the same t in one pre-specified specification.
Null-world constructions (B worlds each):
  perm     : M_k permuted within NAICS-2 x exposure quintile (the application's Type B worlds)
  residperm: M_k = projection on (sector, three exposure proxies, exposure^2, the other nine moderators)
             + residual permuted within the same strata (conditional permutation)
  FL       : Freedman-Lane null-model worlds: outcomes rebuilt from the restricted model (everything but
             E x M_k x Post) with residuals sign-flipped by firm
usage: python3 scripts/24_mc_moderators.py <scenario> R B [workers]
Output: intermediate/24_mc/<scenario>_<r>.json
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from multiprocessing import Pool

_S = {}
C1 = ["log_hc_l1", "hire_rate_l1", "sep_rate_l1", "log_at_l1", "roa_l1", "lev_l1", "sales_growth_l1", "rd_at_l1"]
TARGET, PROXY = "pyramid_z", "remote_z"
PRESPEC = 2          # (first proxy, C1, W0)


def _init():
    import pickle
    from nullworld import registry as R
    _S["reg"] = R.load(ROOT / "registry/design_space.yaml")
    _S["params"] = pickle.loads((ROOT / "synthetic/twin_params.pkl").read_bytes())


def grammar(reg):
    from nullworld.loop import Grammar
    sign = {m["name"] + "_z": (1 if m["sign"] == "amplifies" else -1) for m in reg["moderators"]}
    return Grammar(outcomes=["tilt_exp"], outcome_dir=[1], treatments=[t["name"] + "_z" for t in reg["treatments"]],
                   moderators=[TARGET], mod_sign=[sign[TARGET]], controls={"C0": [], "C1": C1},
                   fes={"FE1": ["firm_idx", "year"]}, samples={"S1": "S1"}, transforms=["W0", "W1"],
                   clusters={"CL1": "firm_idx"})


def target_stats(loop, g):
    """(S_free, S_fix) for the single target moderator in the loop's current world."""
    from nullworld.loop import CRIT
    free, fix = -np.inf, None
    sg = g.mod_sign[0]
    for i, (o, kk, cset, fe, s, w, cl) in enumerate(g.base_specs()):
        fr = loop.frames[(s, fe)]
        d = g.outcome_dir[o]
        y = fr.Yr[(w, cset)][:, o]
        EP, MP, EPM = fr.Zr[(w, cset)]
        r = loop._fit(fr, w, cset, cl, y, [EP[:, [kk]], EPM[:, kk, [0]], MP[:, [0]]])
        if r is None:
            continue
        tm = float(d * r[0][1] / r[1][1] * sg)
        if i == PRESPEC:
            fix = tm
        r1 = loop._fit(fr, w, cset, cl, y, [EP[:, [kk]]])
        if r1 is not None and d * r1[0][0] / r1[1][0] >= CRIT:
            free = max(free, tm)
    return free, fix


def work(args):
    scen, r, B = args
    out = ROOT / "intermediate/24_mc" / f"{scen}_{r:04d}.json"
    if out.exists():
        return None
    from nullworld import synth
    from nullworld.loop import Loop
    from nullworld.fe import FEStructure
    from nullworld.worlds import permute_within
    from nullworld.infer import rank_p
    reg, params = _S["reg"], _S["params"]
    allm = [m["name"] + "_z" for m in reg["moderators"]]
    eff = {"tilt_exp": {"treatment": "expo_beta_z", "beta": 0.10,
                        "gammas": ({PROXY: -0.06} if scen in ("proxy", "all") else {})}}
    p, firms = synth.generate(params, seed=300000 + r, effects=eff, outcome_dir={"tilt_exp": 1})
    firms = firms.sort_values("firm_idx").reset_index(drop=True)
    sdw = params["rowp"]["tilt_exp"]["sd_within"]
    e0 = firms["expo_beta_z"].to_numpy()
    if scen in ("trend", "all"):
        p["tilt_exp"] = p.tilt_exp + 0.10 * sdw * firms[TARGET].to_numpy()[p.firm_idx] * p.post
    if scen in ("nonlinear", "all"):
        p["tilt_exp"] = p.tilt_exp - 0.05 * sdw * (e0 ** 2 - 1)[p.firm_idx] * p.post
    p["S1"] = True
    g = grammar(reg)
    E = firms[g.treatments].to_numpy(float)
    Mk = firms[[TARGET]].to_numpy(float)
    n2 = firms.naics2.to_numpy()
    q = pd.qcut(pd.Series(e0).rank(method="first"), 5, labels=False).to_numpy()
    strata = n2.astype(int) * 10 + q
    loop = Loop(p, g); loop.set_world(E, Mk)
    obs_free, obs_fix = target_stats(loop, g)
    # residualized moderator (conditional permutation)
    others = firms[[m for m in allm if m != TARGET]].to_numpy(float)
    D = pd.get_dummies(n2.astype(int)).to_numpy(float)
    Z = np.column_stack([D, E, e0 ** 2, others])
    fitm = Z @ np.linalg.lstsq(Z, Mk, rcond=None)[0]
    resm = Mk - fitm
    # Freedman-Lane restricted residuals (reference specification: firm + year effects, C1)
    fi = p.firm_idx.to_numpy(); post = p.post.to_numpy(float)
    Mall = firms[allm].to_numpy(float)
    X = np.column_stack([E[fi] * post[:, None], (e0[fi] ** 2 * post)[:, None], Mall[fi] * post[:, None]]
                        + [(e0[fi] * Mall[fi, j] * post)[:, None] for j, m in enumerate(allm) if m != TARGET]
                        + [p[c].to_numpy(float)[:, None] for c in C1])
    fe = FEStructure([fi, p.year.to_numpy()])
    Zd = fe.demean(np.column_stack([p.tilt_exp.to_numpy(float), X]))
    beta = np.linalg.lstsq(Zd[:, 1:], Zd[:, 0], rcond=None)[0]
    eps = Zd[:, 0] - Zd[:, 1:] @ beta
    null = {c: {"free": [], "fix": []} for c in ("perm", "residperm", "FL")}
    for b in range(B):
        rb = np.random.default_rng(70000 + r * 1000 + b)
        for c in ("perm", "residperm"):
            Mb = Mk[permute_within(strata, rb)] if c == "perm" else fitm + resm[permute_within(strata, rb)]
            loop.set_world(E, Mb)
            f_, x_ = target_stats(loop, g)
            null[c]["free"].append(f_); null[c]["fix"].append(x_)
        v = rb.choice([-1.0, 1.0], size=len(firms))
        loop.set_world(E, Mk, ((v[fi] - 1.0) * eps)[:, None])
        f_, x_ = target_stats(loop, g)
        null["FL"]["free"].append(f_); null["FL"]["fix"].append(x_)
        loop.set_world(E, Mk, np.zeros((len(p), 1)))      # restore the observed outcomes
    res = dict(scenario=scen, r=r, B=B, obs_free=obs_free, obs_fix=obs_fix,
               corr_target_proxy=float(np.corrcoef(Mk[:, 0], firms[PROXY])[0, 1]),
               corr_target_expo=float(np.corrcoef(Mk[:, 0], e0)[0, 1]), p={}, reject={})
    for c, d in null.items():
        for s_, obs in (("free", obs_free), ("fix", obs_fix)):
            pv = rank_p(obs, np.array(d[s_], dtype=float))
            res["p"][f"{c}_{s_}"] = pv
            res["reject"][f"{c}_{s_}"] = bool(pv <= 0.05)
    res["reject"]["nominal_fix"] = bool(obs_fix >= 1.96)
    res["reject"]["nominal_free"] = bool(obs_free >= 1.96)
    out.write_text(json.dumps(res, default=float))
    return r


def main():
    scen, R_, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    (ROOT / "intermediate/24_mc").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(nw, initializer=_init) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, [(scen, r, B) for r in range(R_)])):
            if (i + 1) % 50 == 0:
                print(f"{scen}: {i+1}/{R_} in {time.time()-t0:.0f}s", flush=True)
    print("DONE", scen, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
