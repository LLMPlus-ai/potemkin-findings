"""11 - Replay the loop on null worlds (Type A: treatment permuted within NAICS-2;
Type B: one moderator permuted within NAICS-2 x exposure quintile) and on
planted worlds (known effects added to a Type A world).

usage:
  python3 scripts/11_null_worlds.py typeA  B_start B_end [workers]
  python3 scripts/11_null_worlds.py typeB  <moderator_name> B_start B_end [workers]
  python3 scripts/11_null_worlds.py planted <beta_star> <gamma_star> B_start B_end [workers]
Results are appended world by world to intermediate/11_<kind>/<tag>_<b>.json
(resumable; existing worlds are skipped).
"""
import sys, json, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats

_G = {}


def _init():
    from nullworld import registry as R
    from nullworld.loop import Loop
    R.verify(ROOT / "registry/design_space.yaml", ROOT / "registry/design_space.sha256")
    reg = R.load(ROOT / "registry/design_space.yaml")
    pp, fp, tag = R.data_paths(ROOT)
    panel = pd.read_parquet(pp)
    firms = pd.read_parquet(fp)
    _G["tag"] = tag
    g = R.grammar_from_registry(reg, panel)
    E, M, n2 = R.firm_matrices(firms, g)
    _G.update(dict(R=R, g=g, E=E, M=M, n2=n2, panel=panel, loop=Loop(panel, g),
                   strataB=R.exposure_strata(E[:, 0], n2)))
    # within-firm SD of each outcome (planted-effect units)
    sd = []
    for o in g.outcomes:
        w = panel[o] - panel.groupby("gvkey")[o].transform("mean")
        sd.append(float(w.std()))
    _G["sd"] = np.array(sd)
    real = json.loads((ROOT / f"intermediate/{tag}10_loop_real.json").read_text())
    pkg = real["U"]["first_success"] or real["U"]["best"]
    _G["planted_mods"] = [g.moderators.index(m) for m in pkg["moderators"]] if pkg else [0, 1, 2]


def _summ(rep, g):
    crit_bonf = stats.norm.ppf(1 - 0.05 / max(rep.n_models, 1))
    return dict(success=rep.success, T_first=rep.T_first, T_max=rep.T_max, h1_max=rep.h1_max,
                n_models=rep.n_models, n_to_first=rep.n_models_to_first,
                bonf_success=bool(rep.success and rep.T_first >= crit_bonf),
                share_h1_sig=float(np.mean(np.array(rep.h1_ts) >= 1.96)),
                median_h1_t=float(np.median(rep.h1_ts)),
                spec_curve_success=bool(np.median(rep.h1_ts) >= 1.96),
                pkg=rep.first_success or rep.best)


def _guard_setup():
    """Fixed exploration/confirmation split (registry seed) and an exploration-half loop."""
    if "gl" in _G:
        return
    from nullworld.loop import Loop
    reg = _G["R"].load(ROOT / "registry/design_space.yaml")
    firms = pd.read_parquet(R.data_paths(ROOT)[1])
    rng = np.random.default_rng(reg["guard"]["seed"])
    fr = firms[["gvkey", "naics2", "firm_idx"]].sort_values("gvkey").reset_index(drop=True)
    expl = np.zeros(len(fr), bool)
    for s_, ix in fr.groupby("naics2").groups.items():
        ix = np.array(sorted(ix))
        pick = rng.permutation(ix)[: len(ix) // 2 + (rng.random() < 0.5 if len(ix) % 2 else 0)]
        expl[pick] = True
    ex_idx = np.sort(fr.firm_idx[expl].to_numpy()); cf_idx = np.sort(fr.firm_idx[~expl].to_numpy())
    pe = _G["panel"][_G["panel"].firm_idx.isin(ex_idx)].copy()
    remap = {old: i for i, old in enumerate(ex_idx)}
    pe["firm_idx"] = pe.firm_idx.map(remap)
    _G["gl"] = Loop(pe, _G["g"]); _G["ex_idx"] = ex_idx
    _G["pc"] = _G["panel"][_G["panel"].firm_idx.isin(cf_idx)].copy()


def _confirm(pkg, Eb, Mb):
    """Single pre-specified confirmation fit of the exploration package on the held-out half."""
    from nullworld.fe import FEStructure, ClusterOLS
    from nullworld.loop import winsorize_by_year
    from nullworld.infer import holm
    g, sub = _G["g"], _G["pc"]
    sub = sub[sub[pkg["sample"]]]
    k = g.treatments.index(pkg["treatment"]); o = g.outcomes.index(pkg["outcome"]); d = g.outcome_dir[o]
    fi = sub.firm_idx.to_numpy(); post = sub.post.to_numpy(float)
    e = Eb[fi, k] * post
    js = [g.moderators.index(m) for m in pkg["moderators"]]
    cols = [e] + [e * Mb[fi, j] for j in js] + [Mb[fi, j] * post for j in js]
    y = sub[pkg["outcome"]].to_numpy(float)
    if pkg["transform"] == "W1":
        y = winsorize_by_year(y, sub.year.to_numpy())
    C = [sub[c].to_numpy(float) for c in g.controls[pkg["controls"]]]
    fe = FEStructure([sub[c].to_numpy() for c in g.fes[pkg["fe"]]])
    Z = fe.demean(np.column_stack([y] + cols + C))
    r = ClusterOLS(sub[g.clusters[pkg["cluster"]]].to_numpy()).fit(Z[:, 0], Z[:, 1:])
    if r is None:
        return dict(success=False)
    b, se, _ = r
    signs = [1] + [g.mod_sign[j] for j in js]
    t = [d * sg * b[i] / se[i] for i, sg in enumerate(signs)]
    p = [1 - stats.norm.cdf(x) for x in t]
    ph = holm(p)
    return dict(success=bool(np.all(ph < 0.05)), t=t, p_holm=ph.tolist(), h1_success=bool(p[0] < 0.05))


def work(args):
    kind, tag, b, params = args
    out = ROOT / f"intermediate/{_G.get('tag', '')}11_{kind}" / f"{tag}_{b:04d}.json"
    if out.exists():
        return None
    from nullworld import worlds as W
    g, E, M = _G["g"], _G["E"], _G["M"]
    t0 = time.time()
    Y_add = None
    if kind == "typeA":
        Eb, Mb = W.type_a(E, _G["n2"], b), M
    elif kind == "typeB":
        k = g.moderators.index(params["moderator"])
        Eb, Mb = E, W.type_b(M, k, _G["strataB"], b)
    elif kind == "guardA":
        Eb, Mb = W.type_a(E, _G["n2"], b), M
    else:  # planted
        Eb, Mb = W.type_a(E, _G["n2"], b), M
        p = _G["panel"]
        mods = _G["planted_mods"]
        Y_add = W.planted_outcome_shift(p.firm_idx.to_numpy(), p.post.to_numpy(float), Eb, Mb, _G["sd"],
                                        g.outcome_dir, params["beta"], 0, params["gamma"], mods,
                                        [g.mod_sign[j] for j in mods])
    if kind == "guardA":
        _guard_setup()
        Eb, Mb = W.type_a(E, _G["n2"], b), M
        gl = _G["gl"]; gl.set_world(Eb[_G["ex_idx"]], Mb[_G["ex_idx"]])
        rep = gl.run("U", full=False)
        pkg = rep.first_success or rep.best
        res = {"b": b, "kind": kind, "tag": tag, "explore_success": rep.success,
               "confirm": (_confirm(pkg, Eb, Mb) if pkg else dict(success=False)), "pkg": pkg,
               "seconds": time.time() - t0}
        out.write_text(json.dumps(res, default=float))
        return b
    loop = _G["loop"]
    loop.set_world(Eb, Mb, Y_add)
    res = {"b": b, "kind": kind, "tag": tag}
    if kind == "typeB":
        if "record" not in _G:          # exact cache of fits unaffected by permuting moderator k
            loop.set_world(E, M); rec = {}
            loop.run_both(full=True, record=rec); _G["record"] = rec
            loop.set_world(Eb, Mb, Y_add)
        reps = loop.run_both(full=False, cache=_G["record"], recompute=(g.moderators.index(params["moderator"]),))
    else:
        reps = loop.run_both(full=(kind == "typeA"))
    for pol in ["U", "Ufree"]:
        res[pol] = _summ(reps[pol], g)
    if kind == "planted":
        loop.set_world(Eb, Mb, np.zeros_like(Y_add))  # reset outcomes for the next world
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, default=float))
    return b


def main():
    kind = sys.argv[1]
    if kind in ("typeA", "guardA"):
        tag, params = ("A" if kind == "typeA" else "G"), {}
        b0, b1 = int(sys.argv[2]), int(sys.argv[3]); nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    elif kind == "typeB":
        tag, params = f"B_{sys.argv[2]}", {"moderator": sys.argv[2]}
        b0, b1 = int(sys.argv[3]), int(sys.argv[4]); nw = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    else:
        params = {"beta": float(sys.argv[2]), "gamma": float(sys.argv[3])}
        tag = f"P_b{params['beta']}_g{params['gamma']}"
        b0, b1 = int(sys.argv[4]), int(sys.argv[5]); nw = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    import os
    pre = "synthetic_" if os.environ.get("NULLWORLD_DATA", "").lower() == "synthetic" else ""
    (ROOT / f"intermediate/{pre}11_{kind}").mkdir(parents=True, exist_ok=True)
    jobs = [(kind, tag, b, params) for b in range(b0, b1)]
    t0 = time.time()
    with Pool(nw, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs)):
            if (i + 1) % 20 == 0:
                print(f"{kind} {tag}: {i+1}/{len(jobs)} worlds, {time.time()-t0:.0f}s", flush=True)
    print("DONE", kind, tag, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
