"""13 - Monte Carlo validation of loop-level inference on synthetic twins with
KNOWN truth.

Scenarios (R replications each; B inner null worlds per replication):
  null        : no treatment or moderator effects; treatment exchangeable within sector.
  confounded  : no causal effect, but post-period outcome drift is correlated with the
                treatment through an unobserved firm trait (exchangeability violated) -
                shows that NWC prices SEARCH, not CONFOUNDING.
  leakage     : null world; after peeking at the H1 results of all 12 base
                specifications, the analyst prunes the design space to the single
                best-looking specification and "registers" it; the pruning is NOT
                replayed in the null worlds (size distortion the registry prevents).
  leakage_replayed : the same pruning replayed inside every null world (valid again).
  power       : planted H1 (beta) and moderator (gamma) effects.
Reduced grammar for speed: outcome tilt_exp; 3 treatments; controls C0/C1;
FE1; sample S1; W0/W1; CL1 (12 base specifications) and the 10-moderator search.
usage: python3 scripts/13_monte_carlo.py <scenario> R B [workers]
Output: intermediate/13_mc/<scenario>_<r>.json
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


def _init():
    import pickle
    from nullworld import registry as R
    reg = R.load(ROOT / "registry/design_space.yaml")
    _S["reg"] = reg
    _S["params"] = pickle.loads((ROOT / "synthetic/twin_params.pkl").read_bytes())


def _grammar(reg, mods=None):
    from nullworld.loop import Grammar
    allm = [m["name"] + "_z" for m in reg["moderators"]]
    sign = {m["name"] + "_z": (1 if m["sign"] == "amplifies" else -1) for m in reg["moderators"]}
    mods = mods or allm
    return Grammar(outcomes=["tilt_exp"], outcome_dir=[1],
                   treatments=[t["name"] + "_z" for t in reg["treatments"]], moderators=mods,
                   mod_sign=[sign[m] for m in mods], controls={"C0": [], "C1": C1},
                   fes={"FE1": ["firm_idx", "year"]}, samples={"S1": "S1"}, transforms=["W0", "W1"],
                   clusters={"CL1": "firm_idx"})


def _prep(p, firms):
    p = p.copy(); p["S1"] = True
    return p


def _peek_mods(p, firms, allm, k=3):
    """Proposal leakage: choose moderators most associated with the outcome change."""
    ch = p[p.post == 1].groupby("firm_idx").tilt_exp.mean() - p[p.post == 0].groupby("firm_idx").tilt_exp.mean()
    f = firms.set_index("firm_idx").loc[ch.index]
    e = f["expo_beta_z"]
    score = {m: abs(np.corrcoef(ch * e, f[m])[0, 1]) for m in allm}
    return sorted(allm, key=lambda m: -score[m])[:k]


def work(args):
    scen, r, B = args
    out = ROOT / "intermediate/13_mc" / f"{scen}_{r:04d}.json"
    if out.exists():
        return None
    from nullworld import synth, worlds as W
    from nullworld.loop import Loop
    from nullworld.infer import rank_p
    reg, params = _S["reg"], _S["params"]
    allm = [m["name"] + "_z" for m in reg["moderators"]]
    eff = None
    if scen == "power":
        eff = {"tilt_exp": {"treatment": "expo_beta_z", "beta": 0.10,
                            "gammas": {"analysts_z": 0.06, "slack_cash_z": -0.06, "remote_z": 0.06}}}
    p, firms = synth.generate(params, seed=100000 + r, effects=eff, outcome_dir={"tilt_exp": 1})
    if scen == "confounded":
        u = 0.6 * firms["expo_beta_z"].to_numpy() + 0.8 * np.random.default_rng(r).standard_normal(len(firms))
        sd = params["rowp"]["tilt_exp"]["sd_within"]
        p["tilt_exp"] = p.tilt_exp + 0.10 * sd * u[p.firm_idx] * p.post
    p = _prep(p, firms)
    g = _grammar(reg)
    E = firms.sort_values("firm_idx")[g.treatments].to_numpy(float)
    M = firms.sort_values("firm_idx")[g.moderators].to_numpy(float)
    n2 = firms.sort_values("firm_idx").naics2.to_numpy()
    loop = Loop(p, g); loop.set_world(E, M)
    rep = loop.run("U", full=True)
    obs = dict(success=rep.success, T=rep.T_first, h1=rep.h1_max, n_models=rep.n_models,
               spec_share=float(np.mean(np.array(rep.h1_ts) >= 1.96)), spec_med=float(np.median(rep.h1_ts)))
    if scen == "leakage":
        from nullworld.loop import Grammar
        best = g.base_specs()[int(np.argmax(rep.h1_ts))]
        o, k, cset, fe, s_, w, cl = best
        gl = Grammar(outcomes=[g.outcomes[o]], outcome_dir=[g.outcome_dir[o]], treatments=g.treatments,
                     moderators=g.moderators, mod_sign=g.mod_sign, controls={cset: g.controls[cset]},
                     fes={fe: g.fes[fe]}, samples={s_: g.samples[s_]}, transforms=[w], clusters={cl: g.clusters[cl]})
        leak_loop = Loop(p, gl)
    nullT, nullH = [], []
    for b in range(B):
        Eb = W.type_a(E, n2, 50000 + r * 1000 + b)
        if scen == "leakage":
            # pruned design space applied to the null world WITHOUT replaying the pruning;
            # the treatment proxy chosen on the real data is kept fixed
            Eb_leak = Eb.copy()
            leak_loop.set_world(Eb_leak, M); rl = leak_loop.run("U", full=True)
            nullH.append(rl.h1_ts[k] if len(rl.h1_ts) > k else rl.h1_max); nullT.append(rl.T_first)
        else:
            loop.set_world(Eb, M); rb = loop.run("U", full=True)
            nullT.append(rb.T_first); nullH.append(rb.h1_max)
    crit_bonf = stats.norm.ppf(1 - 0.05 / max(rep.n_models, 1))
    res = dict(scenario=scen, r=r, B=B, obs=obs,
               p_nwc_package=rank_p(rep.T_first, np.array(nullT)), p_nwc_h1=rank_p(rep.h1_max, np.array(nullH)),
               U_success=rep.success, BONF_success=bool(rep.success and rep.T_first >= crit_bonf),
               SPEC_success=obs["spec_med"] >= 1.96)
    out.write_text(json.dumps(res, default=float))
    return r


def main():
    scen, R_, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    (ROOT / "intermediate/13_mc").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(nw, initializer=_init) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, [(scen, r, B) for r in range(R_)])):
            if (i + 1) % 10 == 0:
                print(f"{scen}: {i+1}/{R_} in {time.time()-t0:.0f}s", flush=True)
    print("DONE", scen, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
