"""14 - Aggregate null-world, planted-world and guard results into the numbers the
paper reports.

Potemkin rate of a procedure = share of Type A null worlds (treatment permuted
within NAICS-2; no effect of the treatment by construction) in which the
procedure reports a fully significant H1-H4 package.
Search-adjusted p-values (Theorem 1): rank of the real statistic among the
null-world statistics produced by the SAME loop.
Outputs: tables/nwc.json, intermediate/14_null_summary.csv
"""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld.infer import rank_p, holm, mc_band


def load(kind, tag=None):
    fs = sorted(glob.glob(str(ROOT / f"intermediate/11_{kind}" / f"{tag or ''}*.json")))
    return [json.loads(Path(f).read_text()) for f in fs]


def mod_stat(pkg, mod):
    """Sign-consistent t of moderator `mod` in a reported package (-inf if absent)."""
    if not pkg or mod not in pkg["moderators"]:
        return -np.inf
    return pkg["t_mods"][pkg["moderators"].index(mod)]


def main():
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    out = {"registry_sha256": real["registry_sha256"]}
    A = load("typeA", "A_")
    B = len(A)
    out["B_typeA"] = B
    if B:
        df = pd.DataFrame([dict(b=w["b"], U_success=w["U"]["success"], U_T=w["U"]["T_first"], U_Tmax=w["U"]["T_max"],
                                h1_max=w["U"]["h1_max"], U_n=w["U"]["n_models"], U_bonf=w["U"]["bonf_success"],
                                spec_share=w["U"]["share_h1_sig"], spec_med=w["U"]["median_h1_t"],
                                spec_success=w["U"]["spec_curve_success"],
                                F_success=w["Ufree"]["success"], F_T=w["Ufree"]["T_first"], sec=w["seconds"]) for w in A])
        df.to_csv(ROOT / "intermediate/14_null_summary.csv", index=False)
        rates = {}
        for name, col in [("U (registered signs)", "U_success"), ("U-free (signs rationalized)", "F_success"),
                          ("Bonferroni over models visited", "U_bonf"), ("Specification curve (median t >= 1.96)", "spec_success")]:
            r = float(df[col].mean()); lo, hi = mc_band(r, B)
            rates[name] = dict(rate=r, lo=max(lo, 0), hi=hi)
        # NWC by construction: leave-one-out rank p of each null world among the others
        T = df.U_T.to_numpy()
        loo = np.array([rank_p(T[i], np.delete(T, i)) for i in range(B)])
        r = float(np.mean(loo <= 0.05)); lo, hi = mc_band(r, B)
        rates["NWC (loop-level p <= .05)"] = dict(rate=r, lo=max(lo, 0), hi=hi)
        out["potemkin_rates"] = rates
        out["null_h1_share_sig_mean"] = float(df.spec_share.mean())
        out["null_models_mean"] = float(df.U_n.mean())
        out["null_seconds_mean"] = float(df.sec.mean())
        # search-adjusted p-values for the real report
        out["p_nwc_h1_max"] = rank_p(real["U"]["h1_max"], df.h1_max.to_numpy())
        out["p_nwc_package_U"] = rank_p(real["U"]["T_first"], df.U_T.to_numpy())
        out["p_nwc_package_Ufree"] = rank_p(real["Ufree"]["T_first"], df.F_T.to_numpy())
        out["null_h1_max_q95"] = float(np.quantile(df.h1_max, 0.95))
        from scipy.stats import norm as _n
        crit_h1 = _n.ppf(1 - 0.05 / 648)
        out["bonf_h1_crit"] = float(crit_h1)
        out["bonf_h1_potemkin"] = float(np.mean(df.h1_max >= crit_h1))
        out["null_h1_max_median"] = float(df.h1_max.median())
        out["null_U_T_q95"] = float(np.quantile(df.U_T[np.isfinite(df.U_T)], 0.95)) if np.isfinite(df.U_T).any() else None
        out["real"] = dict(h1_max=real["U"]["h1_max"], U_T=real["U"]["T_first"], Ufree_T=real["Ufree"]["T_first"],
                           U_models_to_first=real["U"]["n_models_to_first"], U_models=real["U"]["n_models"],
                           Ufree_models_to_first=real["Ufree"]["n_models_to_first"],
                           share_h1_sig=real["U"]["share_h1_sig"], package=real["U"]["first_success"])
    # Type B: per-moderator search-adjusted p-values for the moderators in the real package
    pkg = real["U"]["first_success"] or real["U"]["best"]
    tb = {}
    if pkg:
        for m in pkg["moderators"]:
            W = load("typeB", f"B_{m}_")
            if not W:
                continue
            null = np.array([mod_stat(w["U"]["pkg"] if w["U"]["success"] else None, m) for w in W])
            tb[m] = dict(B=len(W), t_real=mod_stat(pkg, m), p=rank_p(mod_stat(pkg, m), null),
                         share_null_selected=float(np.mean(np.isfinite(null))))
        if tb:
            ps = [tb[m]["p"] for m in tb]
            for m, ph in zip(tb, holm(ps)):
                tb[m]["p_holm_package"] = float(ph)
            npool = 10
            order = np.argsort(ps)
            for rank_i, j in enumerate(order):
                m = list(tb)[j]
                tb[m]["p_holm_pool10"] = float(min(1.0, max((npool - r_) * ps[order[r_]] for r_ in range(rank_i + 1))))
    out["typeB"] = tb
    # guard in null worlds
    G = load("guardA", "G_")
    if G:
        s = np.array([g_["confirm"].get("success", False) for g_ in G])
        h = np.array([g_["confirm"].get("h1_success", False) for g_ in G])
        lo, hi = mc_band(float(s.mean()), len(G))
        out["guard_null"] = dict(B=len(G), package_rate=float(s.mean()), lo=max(lo, 0), hi=hi,
                                 h1_rate=float(h.mean()), explore_success_rate=float(np.mean([g_["explore_success"] for g_ in G])))
    # planted worlds: power, NWC power against the Type A null distribution, winner's curse, moderator recovery
    panel = pd.read_parquet(ROOT / "data/analysis/panel.parquet")
    sdw = {o: float((panel[o] - panel.groupby("gvkey")[o].transform("mean")).std())
           for o in ["tilt_exp", "hire_sen_mean", "entry_rate"]}
    dirs = {"tilt_exp": 1, "hire_sen_mean": 1, "entry_rate": -1}
    nullT = np.array([w["U"]["T_first"] for w in A]) if A else np.array([])
    planted_mods = set(pkg["moderators"]) if pkg else set()
    P = {}
    for f in sorted(glob.glob(str(ROOT / "intermediate/11_planted/P_*_*.json"))):
        w = json.loads(Path(f).read_text()); P.setdefault(w["tag"], []).append(w)
    pl = {}
    for tag, ws in P.items():
        beta = float(tag.split("_b")[1].split("_g")[0]); gam = float(tag.split("_g")[1])
        ratios, rec_mods = [], []
        for w in ws:
            pk = w["U"]["pkg"] if w["U"]["success"] else None
            if pk and pk["treatment"] == "expo_beta_z":
                truth = beta * sdw[pk["outcome"]] * dirs[pk["outcome"]]
                ratios.append(pk["b"][0] / truth)
            if pk:
                rec_mods.append(len(set(pk["moderators"]) & planted_mods) / 3.0)
        pl[tag] = dict(beta_star=beta, gamma_star=gam, worlds=len(ws),
                       U_power=float(np.mean([w["U"]["success"] for w in ws])),
                       Ufree_power=float(np.mean([w["Ufree"]["success"] for w in ws])),
                       bonf_power=float(np.mean([w["U"]["bonf_success"] for w in ws])),
                       nwc_power=float(np.mean([rank_p(w["U"]["T_first"], nullT) <= 0.05 for w in ws])) if len(nullT) else None,
                       winner_ratio_median=float(np.median(ratios)) if ratios else None,
                       winner_ratio_n=len(ratios),
                       moderator_recovery=float(np.mean(rec_mods)) if rec_mods else None)
    out["planted"] = pl
    (ROOT / "tables/nwc.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("planted",)}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
