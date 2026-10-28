"""22 - Validity worlds requested in the 20 October 2026 review (plan_10_20.tex).

Four constructions, each replaying the SAME registered loop (policy U) on the real panel:

  joint   : all ten registered moderators permuted independently within NAICS-2 x exposure-quintile
            strata in the same world (the registered Type B design: "permute all moderator
            candidates"). Gives the familywise null distribution of the largest moderator statistic,
            so the 10 -> 3 selection of moderators is priced (single-step and step-down max-T).
  permB   : the existing one-moderator Type B worlds (same seeds as scripts/11), re-evaluated with two
            further statistics so the moderator p-value can be decomposed (see `mod_stats`).
  fl      : Freedman-Lane null-model worlds for one moderator k. Outcomes are rebuilt under the
            restricted model that contains everything except E x M_k x Post (firm and NAICS-3 x year
            effects, the eight C1 controls, the three exposure proxies x Post, exposure^2 x Post, all ten
            M_j x Post, and E x M_j x Post for j != k); residuals are sign-flipped by firm (wild cluster,
            Rademacher). M_k and all its relations with exposure, the other moderators and outcome
            trends are kept exactly; only the target interaction is removed.
  residB  : conditional-permutation moderator worlds for one moderator k: M_k is split into its projection
            on (NAICS-2, the three exposure proxies, exposure^2, the other nine moderators) and a residual;
            only the residual is permuted within NAICS-2 x exposure-quintile strata, so M_k keeps its linear
            relations with exposure and with the other moderators.
  covA    : covariate-preserving treatment worlds. Each exposure proxy is split, within NAICS-2, into
            its linear projection on twelve predetermined firm traits (the ten moderators, pre-period
            size, pre-period hiring growth) and a residual; the residual vector (all three proxies
            jointly) is permuted within NAICS-2. Exposure keeps its within-sector correlation with the
            observed traits, so trait-driven outcome trends are present in every null world.

Moderator statistics (sign-consistent, registered signs):
  S_sel  : t of M_k in the package the loop reports (-inf if the loop does not report M_k) - the
           statistic of scripts/11 and the proposal figure;
  S_free : max over base specifications with a significant H1 of the single-moderator t of M_k (the
           statistic the loop screens on; defined in every world, so it measures strength, not selection);
  S_fix  : t of M_k in the reported package specification with its two partner moderators (no search).

usage:
  python3 scripts/22_validity_worlds.py joint B_start B_end [workers]
  python3 scripts/22_validity_worlds.py permB <moderator> B_start B_end [workers]
  python3 scripts/22_validity_worlds.py fl    <moderator> B_start B_end [workers]
  python3 scripts/22_validity_worlds.py residB <moderator> B_start B_end [workers]
  python3 scripts/22_validity_worlds.py covA  B_start B_end [workers]
  python3 scripts/22_validity_worlds.py real            (observed S_free / S_fix and the covA R^2)
Outputs: intermediate/22_<kind>/<tag>_<b>.json (resumable), intermediate/22_real.json
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from multiprocessing import Pool

_G = {}
TRAITS = ["ilm_share_z", "slack_cash_z", "analysts_z", "firm_age_z", "remote_z", "ai_talent_pre_z", "pyramid_z",
          "entry_wage_z", "hhi_ind_z", "rd_int_z", "pre_log_hc", "pre_hire_growth"]


def _init():
    from nullworld import registry as R
    from nullworld.loop import Loop
    R.verify(ROOT / "registry/design_space.yaml", ROOT / "registry/design_space.sha256")
    reg = R.load(ROOT / "registry/design_space.yaml")
    panel = pd.read_parquet(ROOT / "data/analysis/panel.parquet")
    firms = pd.read_parquet(ROOT / "data/analysis/firms.parquet")
    g = R.grammar_from_registry(reg, panel)
    E, M, n2 = R.firm_matrices(firms, g)
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    loop = Loop(panel, g)
    loop.set_world(E, M)
    rec = {}
    loop.run_both(full=True, record=rec)          # exact cache of fits that do not change when only M changes
    _G.update(dict(R=R, g=g, E=E, M=M, n2=n2, panel=panel, firms=firms.sort_values("firm_idx").reset_index(drop=True),
                   loop=loop, record=rec, pkg=real["U"]["first_success"], strataB=R.exposure_strata(E[:, 0], n2),
                   h1_pass=[key for key, v in rec.items() if v["t1"] >= 1.959963984540054]))


# ---------------------------------------------------------------- statistics
def mod_stats(ks, h1_pass=None):
    """S_free and S_fix for moderators ks (indices) in the loop's current world."""
    from nullworld.loop import CRIT
    g, loop, pkg = _G["g"], _G["loop"], _G["pkg"]
    signs = np.asarray(g.mod_sign)
    free = {k: -np.inf for k in ks}
    specs = h1_pass if h1_pass is not None else g.base_specs()
    for (o, kk, cset, fe, s, w, cl) in specs:
        fr = loop.frames[(s, fe)]
        d = g.outcome_dir[o]
        y = fr.Yr[(w, cset)][:, o]
        EP, MP, EPM = fr.Zr[(w, cset)]
        if h1_pass is None:
            r = loop._fit(fr, w, cset, cl, y, [EP[:, [kk]]])
            if r is None or d * r[0][0] / r[1][0] < CRIT:
                continue
        for k in ks:
            r = loop._fit(fr, w, cset, cl, y, [EP[:, [kk]], EPM[:, kk, [k]], MP[:, [k]]])
            if r is not None:
                free[k] = max(free[k], float(d * r[0][1] / r[1][1] * signs[k]))
    # fixed specification of the reported package
    o, kk = g.outcomes.index(pkg["outcome"]), g.treatments.index(pkg["treatment"])
    fr = loop.frames[(pkg["sample"], pkg["fe"])]
    w, cset, cl, d = pkg["transform"], pkg["controls"], pkg["cluster"], g.outcome_dir[o]
    y = fr.Yr[(w, cset)][:, o]
    EP, MP, EPM = fr.Zr[(w, cset)]
    js = [g.moderators.index(m) for m in pkg["moderators"]]
    cols = [EP[:, [kk]]] + [EPM[:, kk, [j]] for j in js] + [MP[:, [j]] for j in js]
    r = loop._fit(fr, w, cset, cl, y, cols)
    fix = {}
    for k in ks:
        if k in js and r is not None:
            q = js.index(k)
            fix[k] = float(d * r[0][1 + q] / r[1][1 + q] * signs[k])
        else:
            fix[k] = None
    return free, fix


def s_sel(rep, g):
    """Sign-consistent t of every moderator in the reported package (absent moderators omitted)."""
    pk = rep.first_success if rep.success else None
    if not pk:
        return {}
    return {m: float(t) for m, t in zip(pk["moderators"], pk["t_mods"])}


# ---------------------------------------------------------------- world constructions
def joint_world(M, strata, b):
    from nullworld.worlds import world_seed, permute_within
    Mb = M.copy()
    for j in range(M.shape[1]):
        rng = np.random.default_rng(world_seed(b, 40 + j))
        Mb[:, j] = M[permute_within(strata, rng), j]
    return Mb


def covA_parts():
    """Within-NAICS-2 projection of the three exposure proxies on the twelve traits."""
    if "covA" in _G:
        return _G["covA"]
    f, E = _G["firms"], _G["E"]
    X = f[TRAITS].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    D = pd.get_dummies(f.naics2.astype(int)).to_numpy(float)
    Z = np.column_stack([D, X])
    coef, *_ = np.linalg.lstsq(Z, E, rcond=None)
    fit = Z @ coef
    res = E - fit
    # within-sector R^2 of the trait projection (share of within-sector exposure variance it preserves)
    Ew = E - D @ np.linalg.lstsq(D, E, rcond=None)[0]
    r2 = 1 - (res ** 2).sum(0) / (Ew ** 2).sum(0)
    _G["covA"] = (fit, res, r2)
    return _G["covA"]


def residB_parts(k):
    key = ("residB", k)
    if key not in _G:
        f, E, M = _G["firms"], _G["E"], _G["M"]
        D = pd.get_dummies(f.naics2.astype(int)).to_numpy(float)
        Z = np.column_stack([D, E, E[:, 0] ** 2, np.delete(M, k, axis=1)])
        fit = Z @ np.linalg.lstsq(Z, M[:, k], rcond=None)[0]
        _G[key] = (fit, M[:, k] - fit)
    return _G[key]


def covA_world(b):
    from nullworld.worlds import world_seed, permute_within
    fit, res, _ = covA_parts()
    rng = np.random.default_rng(world_seed(b, 80))
    return fit + res[permute_within(_G["n2"], rng)]


def fl_residuals(k):
    """Restricted-model residuals (rows x outcomes) for the Freedman-Lane worlds of moderator k."""
    key = ("fl", k)
    if key in _G:
        return _G[key]
    from nullworld.fe import FEStructure
    g, p, E, M = _G["g"], _G["panel"], _G["E"], _G["M"]
    fi = p.firm_idx.to_numpy()
    post = p.post.to_numpy(float)
    e = E[fi] * post[:, None]
    m = M[fi]
    e0 = E[fi, 0]
    cols = [e, (e0 ** 2 * post)[:, None], m * post[:, None]]
    cols += [(e0 * m[:, j] * post)[:, None] for j in range(M.shape[1]) if j != k]
    cols += [p[c].to_numpy(float)[:, None] for c in g.controls["C1"]]
    X = np.column_stack(cols)
    Y = p[g.outcomes].to_numpy(float)
    fe = FEStructure([p.firm_idx.to_numpy(), p.naics3_year.to_numpy()])
    Zd = fe.demean(np.column_stack([Y, X]))
    Yd, Xd = Zd[:, :Y.shape[1]], Zd[:, Y.shape[1]:]
    beta, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
    res = Yd - Xd @ beta
    _G[key] = res
    return res


# ---------------------------------------------------------------- workers
def work(args):
    kind, tag, b, params = args
    out = ROOT / f"intermediate/22_{kind}" / f"{tag}_{b:04d}.json"
    if out.exists():
        return None
    from nullworld import worlds as W
    g, E, M, loop = _G["g"], _G["E"], _G["M"], _G["loop"]
    t0 = time.time()
    res = {"b": b, "kind": kind, "tag": tag}
    if kind == "joint":
        Mb = joint_world(M, _G["strataB"], b)
        loop.set_world(E, Mb)
        reps = loop.run_both(full=False, cache=_G["record"], recompute=tuple(range(len(g.moderators))))
        free, _ = mod_stats(list(range(len(g.moderators))), h1_pass=_G["h1_pass"])
        res.update(sel=s_sel(reps["U"], g), success=reps["U"].success,
                   free={g.moderators[k]: v for k, v in free.items()})
        loop.set_world(E, M)
    elif kind == "permB":
        k = g.moderators.index(params["moderator"])
        Mb = W.type_b(M, k, _G["strataB"], b)                     # identical to scripts/11
        loop.set_world(E, Mb)
        free, fix = mod_stats([k], h1_pass=_G["h1_pass"])
        old = json.loads((ROOT / f"intermediate/11_typeB/B_{params['moderator']}_{b:04d}.json").read_text())
        pk = old["U"]["pkg"] if old["U"]["success"] else None
        sel = (pk["t_mods"][pk["moderators"].index(params["moderator"])]
               if pk and params["moderator"] in pk["moderators"] else None)
        res.update(sel=sel, free=free[k], fix=fix[k])
        loop.set_world(E, M)
    elif kind == "fl":
        k = g.moderators.index(params["moderator"])
        eps = fl_residuals(k)
        rng = np.random.default_rng(W.world_seed(b, 60 + k))
        v = rng.choice([-1.0, 1.0], size=E.shape[0])
        Y_add = (v[_G["panel"].firm_idx.to_numpy()] - 1.0)[:, None] * eps
        loop.set_world(E, M, Y_add)
        reps = loop.run_both(full=False)
        free, fix = mod_stats([k])
        sel = s_sel(reps["U"], g).get(params["moderator"])
        res.update(sel=sel, free=free[k], fix=fix[k], success=reps["U"].success, h1_max=reps["U"].h1_max)
        loop.set_world(E, M, np.zeros_like(Y_add))
    elif kind == "residB":
        k = g.moderators.index(params["moderator"])
        fit, rk = residB_parts(k)
        rng = np.random.default_rng(W.world_seed(b, 100 + k))
        Mb = M.copy()
        Mb[:, k] = fit + rk[W.permute_within(_G["strataB"], rng)]
        loop.set_world(E, Mb)
        reps = loop.run_both(full=False, cache=_G["record"], recompute=(k,))
        free, fix = mod_stats([k], h1_pass=_G["h1_pass"])
        res.update(sel=s_sel(reps["U"], g).get(params["moderator"]), free=free[k], fix=fix[k],
                   success=reps["U"].success)
        loop.set_world(E, M)
    elif kind == "covA":
        Eb = covA_world(b)
        loop.set_world(Eb, M)
        reps = loop.run_both(full=True)
        for pol in ["U", "Ufree"]:
            R_ = reps[pol]
            res[pol] = dict(success=R_.success, T_first=R_.T_first, h1_max=R_.h1_max, n_models=R_.n_models,
                            n_to_first=R_.n_models_to_first)
        loop.set_world(E, M)
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, default=float))
    return b


def real():
    _init()
    g = _G["g"]
    ks = [g.moderators.index(m) for m in _G["pkg"]["moderators"]]
    free, fix = mod_stats(list(range(len(g.moderators))), h1_pass=_G["h1_pass"])
    _, _, r2 = covA_parts()
    out = dict(free={g.moderators[k]: v for k, v in free.items()},
               fix={g.moderators[k]: fix[k] for k in ks},
               sel={m: t for m, t in zip(_G["pkg"]["moderators"], _G["pkg"]["t_mods"])},
               n_h1_pass=len(_G["h1_pass"]), covA_within_r2=dict(zip(g.treatments, map(float, r2))))
    (ROOT / "intermediate/22_real.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    kind = sys.argv[1]
    if kind == "real":
        return real()
    if kind in ("joint", "covA"):
        tag, params = kind, {}
        b0, b1 = int(sys.argv[2]), int(sys.argv[3]); nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    else:
        tag, params = f"{kind}_{sys.argv[2]}", {"moderator": sys.argv[2]}
        b0, b1 = int(sys.argv[3]), int(sys.argv[4]); nw = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    (ROOT / f"intermediate/22_{kind}").mkdir(parents=True, exist_ok=True)
    jobs = [(kind, tag, b, params) for b in range(b0, b1)]
    t0 = time.time()
    with Pool(nw, initializer=_init) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, jobs)):
            if (i + 1) % 20 == 0:
                print(f"{kind} {tag}: {i+1}/{len(jobs)} worlds, {time.time()-t0:.0f}s", flush=True)
    print("DONE", kind, tag, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
