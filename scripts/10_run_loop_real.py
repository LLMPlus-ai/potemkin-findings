"""10 - Run the replayable loop on the real panel (policies U and Ufree), the
exploration/confirmation guard, and save the full forking-paths ledger.

Every run first verifies the registry hash.
Outputs:
  intermediate/10_loop_real.json      reports of U and Ufree on the full sample
  intermediate/10_spec_curve.csv      H1 estimate and t for every base specification
  intermediate/10_guard.json          exploration package and its confirmation estimates
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld import registry as R
from nullworld.loop import Loop, CRIT
from nullworld.fe import FEStructure, ClusterOLS
from nullworld.infer import holm
from scipy import stats


def as_json(rep):
    return dict(success=rep.success, T_first=rep.T_first, T_max=rep.T_max, h1_max=rep.h1_max,
                first_success=rep.first_success, best=rep.best, n_models=rep.n_models,
                n_models_to_first=rep.n_models_to_first,
                share_h1_sig=float(np.mean(np.array(rep.h1_ts) >= CRIT)),
                median_h1_t=float(np.median(rep.h1_ts)))


def confirm(panel, firms, g, pkg, keep_firms):
    """Estimate the selected package once on the given firms."""
    sub = panel[panel.gvkey.isin(keep_firms) & panel[pkg["sample"]]].copy()
    fe_cols = g.fes[pkg["fe"]]
    fe = FEStructure([sub[c].to_numpy() for c in fe_cols])
    k = g.treatments.index(pkg["treatment"])
    o = g.outcomes.index(pkg["outcome"])
    d = g.outcome_dir[o]
    F = firms.set_index("firm_idx")
    e = F[pkg["treatment"]].reindex(sub.firm_idx).to_numpy() * sub.post.to_numpy()
    cols = [e]
    for m in pkg["moderators"]:
        mv = F[m].reindex(sub.firm_idx).to_numpy()
        cols.append(e * mv)
    for m in pkg["moderators"]:
        mv = F[m].reindex(sub.firm_idx).to_numpy()
        cols.append(mv * sub.post.to_numpy())
    y = sub[pkg["outcome"]].to_numpy(float)
    if pkg["transform"] == "W1":
        from nullworld.loop import winsorize_by_year
        y = winsorize_by_year(y, sub.year.to_numpy())
    C = sub[g.controls[pkg["controls"]]].to_numpy(float) if g.controls[pkg["controls"]] else np.zeros((len(sub), 0))
    Z = fe.demean(np.column_stack([y] + cols + ([C] if C.shape[1] else [])))
    b, se, _ = ClusterOLS(sub[g.clusters[pkg["cluster"]]].to_numpy()).fit(Z[:, 0], Z[:, 1:])
    signs = [1] + [g.mod_sign[g.moderators.index(m)] for m in pkg["moderators"]]
    t = [d * s * b[i] / se[i] for i, s in enumerate(signs)]
    p = [1 - stats.norm.cdf(x) for x in t]           # one-sided, registered direction
    return dict(n=len(sub), firms=int(sub.gvkey.nunique()), b=b[:4].tolist(), se=se[:4].tolist(),
                t_signed=t, p_one_sided=p, p_holm=holm(p).tolist())


def main():
    h = R.verify(ROOT / "registry/design_space.yaml", ROOT / "registry/design_space.sha256")
    reg = R.load(ROOT / "registry/design_space.yaml")
    pp, fp, tag = R.data_paths(ROOT)
    panel = pd.read_parquet(pp)
    firms = pd.read_parquet(fp)
    g = R.grammar_from_registry(reg, panel)
    E, M, n2 = R.firm_matrices(firms, g)
    out = {"registry_sha256": h, "data": str(pp)}
    t0 = time.time()
    loop = Loop(panel, g)
    loop.set_world(E, M)
    print(f"frames built in {time.time()-t0:.1f}s", flush=True)
    for pol in ["U", "Ufree"]:
        t0 = time.time()
        rep = loop.run(pol, full=True)
        out[pol] = as_json(rep)
        out[pol]["seconds"] = time.time() - t0
        print(pol, json.dumps({k: out[pol][k] for k in ["success", "T_first", "T_max", "h1_max", "n_models",
                                                         "n_models_to_first", "share_h1_sig", "median_h1_t"]}),
              flush=True)
        if pol == "U":
            specs = g.base_specs()
            sc = pd.DataFrame([dict(outcome=g.outcomes[s[0]], treatment=g.treatments[s[1]], controls=s[2], fe=s[3],
                                    sample=s[4], transform=s[5], cluster=s[6]) for s in specs])
            sc["t_h1"] = rep.h1_ts; sc["b_h1"] = rep.h1_bs
            sc.to_csv(ROOT / f"intermediate/{tag}10_spec_curve.csv", index=False)
    (ROOT / f"intermediate/{tag}10_loop_real.json").write_text(json.dumps(out, indent=1, default=float))

    # ---- guard: exploration / confirmation split within NAICS-2 strata ----
    rng = np.random.default_rng(reg["guard"]["seed"])
    fr = firms[["gvkey", "naics2"]].sort_values("gvkey").reset_index(drop=True)
    expl = np.zeros(len(fr), bool)
    for s, ix in fr.groupby("naics2").groups.items():
        ix = np.array(sorted(ix))
        pick = rng.permutation(ix)[: len(ix) // 2 + (rng.random() < 0.5 if len(ix) % 2 else 0)]
        expl[pick] = True
    ex_firms, cf_firms = set(fr.gvkey[expl]), set(fr.gvkey[~expl])
    pe = panel[panel.gvkey.isin(ex_firms)].copy()
    fe_ = firms[firms.gvkey.isin(ex_firms)].copy()
    remap = {old: i for i, old in enumerate(sorted(fe_.firm_idx))}
    pe["firm_idx"] = pe.firm_idx.map(remap); fe_["firm_idx"] = fe_.firm_idx.map(remap)
    Ee, Me, _ = R.firm_matrices(fe_, g)
    le = Loop(pe, g); le.set_world(Ee, Me)
    rep_e = le.run("U", full=True)
    guard = {"n_exploration_firms": len(ex_firms), "n_confirmation_firms": len(cf_firms),
             "exploration": as_json(rep_e)}
    pkg = rep_e.first_success or rep_e.best
    if pkg:
        guard["confirmation"] = confirm(panel, firms, g, pkg, cf_firms)
        guard["full_sample_same_spec"] = confirm(panel, firms, g, pkg, set(firms.gvkey))
    (ROOT / f"intermediate/{tag}10_guard.json").write_text(json.dumps(guard, indent=1, default=float))
    print(json.dumps(guard, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
