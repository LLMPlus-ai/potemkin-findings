"""25 - Aggregate the validity analyses of the 20 October 2026 review (scripts 22-24) into the numbers,
tables, and macros cited by plan_10_20.tex, the proposal, and the technical appendix.

Inputs : intermediate/22_{permB,residB,fl,joint,covA}/, intermediate/22_real.json, intermediate/23_mc/,
         intermediate/24_mc/, intermediate/11_typeA/, intermediate/11_typeB/, intermediate/11_guardA/,
         intermediate/10_loop_real.json, tables/nwc.json, tables/09_testbed.json, synthetic/twin_params.pkl
Outputs: tables/validity.json, manuscript/numbers_validity.tex,
         tables/tab_v_exch.tex, tables/tab_v_modmc.tex, tables/tab_v_modreal.tex, tables/tab_v_covA.tex
"""
import sys, json, glob, re, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld.infer import rank_p, holm, mc_band

MODS3 = ["pyramid_z", "remote_z", "firm_age_z"]
HYP = {"pyramid_z": "H2", "remote_z": "H3", "firm_age_z": "H4"}
LAB = {"pyramid_z": "hierarchy shape", "remote_z": "digitizable work", "firm_age_z": "organizational age"}


def load(pattern):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / pattern)))]


def arr(x):
    return np.array([-np.inf if v is None else v for v in x], dtype=float)


def minp_adjust(obs, null):
    """Westfall-Young min-P adjusted p-values, single-step and step-down.
    obs: dict j -> observed statistic; null: dict j -> array over the same B joint worlds."""
    keys = list(obs)
    B = len(next(iter(null.values())))
    # marginal p of the observed value and of every null world, each among the B+1 values
    P0, Pb = {}, {}
    for j in keys:
        allv = np.concatenate([[obs[j]], null[j]])
        P0[j] = rank_p(obs[j], null[j])
        srt = np.sort(allv)
        Pb[j] = (len(allv) - np.searchsorted(srt, null[j], side="left")) / (B + 1)
    order = sorted(keys, key=lambda j: P0[j])
    single = {j: float((1 + np.sum(np.min(np.column_stack([Pb[i] for i in keys]), 1) <= P0[j])) / (B + 1)) for j in keys}
    step, run = {}, 0.0
    for i, j in enumerate(order):
        rest = order[i:]
        mp = np.min(np.column_stack([Pb[r] for r in rest]), 1)
        run = max(run, (1 + np.sum(mp <= P0[j])) / (B + 1))
        step[j] = float(min(1.0, run))
    return {j: dict(p=float(P0[j]), single=single[j], stepdown=step[j]) for j in keys}


def maxt_stepdown(obs, null):
    """Westfall-Young step-down max-T (the design log's planned familywise procedure)."""
    order = sorted(obs, key=lambda j: -obs[j])
    run, adj = 0.0, {}
    for i, j in enumerate(order):
        mx = np.max(np.column_stack([null[r] for r in order[i:]]), 1)
        run = max(run, rank_p(obs[j], mx)); adj[j] = float(min(1.0, run))
    return adj


def main():
    out = {}
    nwc = json.loads((ROOT / "tables/nwc.json").read_text())
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    obs = json.loads((ROOT / "intermediate/22_real.json").read_text())
    pkg = real["U"]["first_success"]
    allmods = list(obs["free"])

    # ------------------------------------------------------------ C4: statistic decomposition (perm worlds)
    dec = {}
    for m in MODS3:
        W = load(f"intermediate/22_permB/permB_{m}_*.json")
        sel, free, fix = arr([w["sel"] for w in W]), arr([w["free"] for w in W]), arr([w["fix"] for w in W])
        t = pkg["t_mods"][pkg["moderators"].index(m)]
        selected = np.isfinite(sel)
        n_sel = int(selected.sum())
        dec[m] = dict(B=len(W), t_obs=t, free_obs=obs["free"][m], share_selected=float(selected.mean()),
                      n_selected=n_sel, n_sel_exceed=int(np.sum(sel >= t)),
                      cond_tail=(float(np.mean(sel[selected] >= t)) if n_sel else None),
                      p_sel=rank_p(t, sel), p_free=rank_p(obs["free"][m], free), p_fix=rank_p(t, fix),
                      null_free_q95=float(np.quantile(free, 0.95)), null_fix_q95=float(np.quantile(fix, 0.95)))
        assert abs(dec[m]["p_sel"] - nwc["typeB"][m]["p"]) < 1e-12, "permB S_sel does not reproduce scripts/14"
    out["decomposition"] = dec

    # ------------------------------------------------------------ C2: constructions on the real data
    cons = {}
    for c in ["permB", "residB", "fl"]:
        cons[c] = {}
        for m in MODS3:
            W = load(f"intermediate/22_{c}/{c}_{m}_*.json")
            if not W:
                continue
            sel, free, fix = arr([w["sel"] for w in W]), arr([w["free"] for w in W]), arr([w["fix"] for w in W])
            t = pkg["t_mods"][pkg["moderators"].index(m)]
            cons[c][m] = dict(B=len(W), p_sel=rank_p(t, sel), p_free=rank_p(obs["free"][m], free), p_fix=rank_p(t, fix),
                              share_selected=float(np.isfinite(sel).mean()))
        ps = [cons[c][m]["p_sel"] for m in MODS3 if cons[c].get(m, {}).get("B", 0) >= 500]
        if len(ps) == 3:
            h3, h10 = holm(ps), holm(ps + [1.0] * 7)
            for m, a, b in zip(MODS3, h3, h10):
                cons[c][m]["holm3"], cons[c][m]["holm10"] = float(a), float(b)
    out["constructions"] = cons

    # ------------------------------------------------------------ C3: joint worlds, familywise over ten
    J = load("intermediate/22_joint/joint_*.json")
    if J:
        Bj = len(J)
        nsel = {j: arr([w["sel"].get(j) for w in J]) for j in allmods}
        nfree = {j: arr([w["free"][j] for w in J]) for j in allmods}
        osel = {j: (pkg["t_mods"][pkg["moderators"].index(j)] if j in pkg["moderators"] else -np.inf) for j in allmods}
        mx = np.max(np.column_stack([nsel[j] for j in allmods]), 1)
        fam = dict(B=Bj, success_rate=float(np.mean([w["success"] for w in J])),
                   any_mod_reported_t196=float(np.mean(mx >= 1.96)),
                   maxT_sel={j: rank_p(osel[j], mx) for j in MODS3},
                   maxT_stepdown_sel=maxt_stepdown(osel, nsel),
                   minP_sel=minp_adjust(osel, nsel), minP_free=minp_adjust(obs["free"], nfree),
                   holm10_permB={m: cons["permB"][m]["holm10"] for m in MODS3},
                   holm3_permB={m: cons["permB"][m]["holm3"] for m in MODS3},
                   null_free_q95={j: float(np.quantile(nfree[j], 0.95)) for j in allmods},
                   obs_free=obs["free"])
        # empirical familywise error under the global moderator null: each joint world is scored against the other
        # joint worlds (leave-one-out rank p for every moderator it reports), then corrected as the proposal did
        # (Holm over the three reported) and as it should (Holm over all ten).
        M_ = np.column_stack([nsel[j] for j in allmods])
        rej = {"unadjusted": [], "holm3": [], "holm10": []}
        for b in range(Bj):
            others = np.delete(M_, b, axis=0)
            ps = [(1 + np.sum(others[:, i] >= M_[b, i])) / Bj for i in range(len(allmods)) if np.isfinite(M_[b, i])]
            ps = sorted(ps) or [1.0]
            rej["unadjusted"].append(ps[0] <= 0.05)
            rej["holm3"].append(ps[0] * 3 <= 0.05)
            rej["holm10"].append(ps[0] * 10 <= 0.05)
        fam["empirical_fwer"] = {k: float(np.mean(v)) for k, v in rej.items()}
        out["familywise"] = fam

    # ------------------------------------------------------------ C1: covariate-preserving treatment worlds
    CA = load("intermediate/22_covA/covA_*.json")
    A = load("intermediate/11_typeA/A_*.json")
    if CA:
        def summ(ws):
            h = np.array([w["U"]["h1_max"] for w in ws]); T = np.array([w["U"]["T_first"] for w in ws])
            return dict(B=len(ws), p_h1=rank_p(real["U"]["h1_max"], h), p_pkg=rank_p(real["U"]["T_first"], T),
                        potemkin_U=float(np.mean([w["U"]["success"] for w in ws])),
                        potemkin_Ufree=float(np.mean([w["Ufree"]["success"] for w in ws])),
                        h1_q95=float(np.quantile(h, 0.95)), h1_max=float(h.max()), h1_median=float(np.median(h)),
                        sec_median=float(np.median([w["seconds"] for w in ws])))
        out["covA"] = dict(typeA=summ(A), covA=summ(CA), within_r2=obs["covA_within_r2"])

    # ------------------------------------------------------------ C6: risk by search budget; C5: random family
    nU = np.array([w["U"]["n_to_first"] if w["U"]["success"] else np.inf for w in A])
    nF = np.array([w["Ufree"]["n_to_first"] if w["Ufree"]["success"] else np.inf for w in A])
    grid = np.unique(np.round(np.logspace(0, np.log10(6000), 400)).astype(int))
    out["budget"] = dict(grid=grid.tolist(), U=[float(np.mean(nU <= m)) for m in grid], Ufree=[float(np.mean(nF <= m)) for m in grid],
                         U_at={str(m): float(np.mean(nU <= m)) for m in [12, 100, 1000, 3483]},
                         Ufree_at={str(m): float(np.mean(nF <= m)) for m in [12, 100, 1000, 3483]},
                         real_U=real["U"]["n_models_to_first"], real_Ufree=real["Ufree"]["n_models_to_first"])
    nmod = np.array([w["U"]["n_models"] for w in A])
    pk = [w["U"]["pkg"] for w in A if w["U"]["success"]]
    out["n_h1_pass"] = obs["n_h1_pass"]
    out["family"] = dict(models_min=int(nmod.min()), models_median=float(np.median(nmod)), models_max=int(nmod.max()),
                         distinct_triples=len({tuple(sorted(p["moderators"])) for p in pk}),
                         distinct_base=len({(p["outcome"], p["treatment"], p["controls"], p["fe"], p["sample"], p["transform"], p["cluster"]) for p in pk}),
                         n_success=len(pk))

    # ------------------------------------------------------------ C1/C2: what the permutations break (real firms)
    firms = pd.read_parquet(ROOT / "data/analysis/firms.parquet")
    def wres(x, g):
        x = pd.Series(np.asarray(x, float)); return (x - x.groupby(np.asarray(g)).transform("mean")).to_numpy()
    n2 = firms.naics2.to_numpy()
    e_w = wres(firms.expo_beta_z, n2)
    corr_sector = {c: float(np.corrcoef(e_w, wres(firms[c], n2))[0, 1]) for c in
                   ["remote_z", "pyramid_z", "ai_talent_pre_z", "pre_hire_growth", "firm_age_z", "entry_wage_z", "slack_cash_z"]}
    q = pd.qcut(pd.Series(firms.expo_beta_z.to_numpy()).rank(method="first"), 5, labels=False).to_numpy()
    cell = n2.astype(int) * 10 + q
    p_w = wres(firms.pyramid_z, cell)
    corr_cell = {c: float(np.corrcoef(p_w, wres(firms[c], cell))[0, 1]) for c in ["remote_z", "entry_wage_z", "ilm_share_z"]}
    out["broken_links"] = dict(exposure_within_sector=corr_sector, pyramid_within_typeB_cell=corr_cell)

    # ------------------------------------------------------------ C1: exchangeability Monte Carlo
    tb = json.loads((ROOT / "tables/09_testbed.json").read_text())
    b_real = [h for h in tb["h1"] if h["label"] == "FE1-C1"][0]["b"]
    sdw = pickle.loads((ROOT / "synthetic/twin_params.pkl").read_bytes())["rowp"]["tilt_exp"]["sd_within"]
    rows = []
    for f in sorted({re.sub(r"_\d+\.json$", "", Path(x).name) for x in glob.glob(str(ROOT / "intermediate/23_mc/*.json"))}):
        W = load(f"intermediate/23_mc/{f}_[0-9]*.json")
        r = dict(cell=f, channel=W[0]["channel"], rho=W[0]["rho"], delta=W[0]["delta"], R=len(W), B=W[0]["B"],
                 spurious_pct=float(100 * np.mean([w["b_single_sdw"] for w in W]) * sdw / b_real))
        for k in ["NWC", "NWC_strat", "NWC_cov", "single", "naive", "bonf"]:
            r[k] = float(np.mean([w["reject"][k] for w in W]))
        rows.append(r)
    ex = pd.DataFrame(rows)
    out["mc_exch"] = dict(rows=rows, b_real=b_real, sd_within=sdw)

    # ------------------------------------------------------------ C2: moderator Monte Carlo
    rows = []
    for s in ["clean", "trend", "proxy", "nonlinear", "all"]:
        W = load(f"intermediate/24_mc/{s}_[0-9]*.json")
        if not W:
            continue
        r = dict(scenario=s, R=len(W), B=W[0]["B"])
        for k in W[0]["reject"]:
            r[k] = float(np.mean([w["reject"][k] for w in W]))
        rows.append(r)
    out["mc_mod"] = rows

    (ROOT / "tables/validity.json").write_text(json.dumps(out, indent=1, default=float))
    write_macros(out)
    write_tables(out, ex)
    print(json.dumps({k: v for k, v in out.items() if k not in ("budget", "mc_exch")}, indent=1, default=float)[:6000])


def fp(p):
    """p-value as printed: three decimals, with the Monte Carlo floor shown as <= (1/(B+1))."""
    return f"{p:.3f}"


def write_macros(o):
    L = ["% generated by scripts/25_aggregate_validity.py -- do not edit"]
    def mac(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}\\xspace}}")
    d = o["decomposition"]
    for m, tag in [("pyramid_z", "Pyr"), ("remote_z", "Rem"), ("firm_age_z", "Age")]:
        mac(f"VSel{tag}", f"{100 * d[m]['share_selected']:.1f}")
        mac(f"VNsel{tag}", d[m]["n_selected"])
        mac(f"VpSel{tag}", fp(d[m]["p_sel"])); mac(f"VpFree{tag}", fp(d[m]["p_free"])); mac(f"VpFix{tag}", fp(d[m]["p_fix"]))
        mac(f"VFreeObs{tag}", f"{d[m]['free_obs']:.1f}")
        for c, ct in [("residB", "Res"), ("fl", "FL")]:
            cc = o["constructions"].get(c, {}).get(m, {})
            full = cc.get("B", 0) >= 500
            for st, sk in [("Sel", "p_sel"), ("Free", "p_free"), ("Fix", "p_fix"), ("HolmTen", "holm10")]:
                name = f"V{ct}HolmTen{tag}" if st == "HolmTen" else f"Vp{ct}{st}{tag}"
                mac(name, fp(cc[sk]) if (full and sk in cc) else "[pending]")
        cb = o["constructions"]["permB"][m]
        mac(f"VHolmThree{tag}", fp(cb["holm3"])); mac(f"VHolmTen{tag}", fp(cb["holm10"]))
        f = o.get("familywise")
        mac(f"VMinP{tag}", fp(f["minP_sel"][m]["stepdown"]) if f else "[pending]")
        mac(f"VMaxT{tag}", fp(f["maxT_stepdown_sel"][m]) if f else "[pending]")
        mac(f"VMinPFree{tag}", fp(f["minP_free"][m]["stepdown"]) if f else "[pending]")
    f = o.get("familywise")
    mac("VBjoint", f"{f['B']:,}" if f else "[pending]"); mac("VJointSuccess", f"{100 * f['success_rate']:.1f}" if f else "[pending]")
    for k, kt in [("unadjusted", "Unadj"), ("holm3", "HolmThree"), ("holm10", "HolmTen")]:
        mac(f"VFwer{kt}", f"{100 * f['empirical_fwer'][k]:.1f}" if f else "[pending]")
    if "covA" in o:
        c = o["covA"]
        mac("VBcovA", f"{c['covA']['B']:,}"); mac("VcovAph", fp(c["covA"]["p_h1"])); mac("VcovApkg", fp(c["covA"]["p_pkg"]))
        mac("VcovAPotU", f"{100 * c['covA']['potemkin_U']:.1f}"); mac("VcovAPotFree", f"{100 * c['covA']['potemkin_Ufree']:.1f}")
        mac("VcovAhQ", f"{c['covA']['h1_q95']:.1f}"); mac("VcovAhMax", f"{c['covA']['h1_max']:.1f}")
        mac("VcovARsq", f"{100 * c['within_r2']['expo_beta_z']:.0f}")
        mac("VTypeAhMax", f"{c['typeA']['h1_max']:.1f}")
    bl = o["broken_links"]
    for k, c in [("VcorrExpRem", "remote_z"), ("VcorrExpPyr", "pyramid_z"), ("VcorrExpAI", "ai_talent_pre_z"),
                 ("VcorrExpGrowth", "pre_hire_growth")]:
        mac(k, f"{bl['exposure_within_sector'][c]:.2f}".replace("-", "$-$"))
    for k, c in [("VcorrPyrRemCell", "remote_z"), ("VcorrPyrWageCell", "entry_wage_z"), ("VcorrPyrIlmCell", "ilm_share_z")]:
        mac(k, f"{bl['pyramid_within_typeB_cell'][c]:.2f}".replace("-", "$-$"))
    bm = ROOT / "tables/benchmark.json"
    if bm.exists():
        k = json.loads(bm.read_text())
        mac("VpTmax", fp(k["p_T_max"])); mac("VSecWorld", f"{k['engine_world_s_median']:.1f}")
        mac("VSpeedup", f"{k['speedup']:.0f}"); mac("VFitsWorld", f"{k['fits_per_world_median']:,.0f}")
        mac("VPyfixestWorld", f"{k['pyfixest_world_s']:.0f}"); mac("VPyfixestModel", f"{k['pyfixest_per_model_s']:.3f}")
    mac("VNhPass", o["n_h1_pass"])
    rows0 = {r["cell"]: r for r in o["mc_exch"]["rows"]}
    if "null" in rows0:
        mac("VExSingleNull", f"{100 * rows0['null']['single']:.1f}"); mac("VExNaiveNull", f"{100 * rows0['null']['naive']:.1f}")
    b = o["budget"]
    mac("VBudUThousand", f"{100 * b['U_at']['1000']:.1f}"); mac("VBudUReal", f"{100 * b['U_at']['3483']:.1f}")
    mac("VBudFreeTwelve", f"{100 * b['Ufree_at']['12']:.1f}"); mac("VBudFreeHundred", f"{100 * b['Ufree_at']['100']:.1f}")
    mac("VBudUTwelve", f"{100 * b['U_at']['12']:.1f}")
    fa = o["family"]
    mac("VModelsMin", f"{fa['models_min']:,}"); mac("VModelsMax", f"{fa['models_max']:,}")
    mac("VModelsMed", f"{fa['models_median']:,.0f}"); mac("VTriples", fa["distinct_triples"]); mac("VBaseDistinct", fa["distinct_base"])
    # exchangeability Monte Carlo: selected cells
    rows = {r["cell"]: r for r in o["mc_exch"]["rows"]}
    def pc(cell, k):
        return f"{100 * rows[cell][k]:.1f}" if cell in rows else "[pending]"
    mac("VExNull", pc("null", "NWC")); mac("VExLevel", pc("level", "NWC"))
    post = [r for r in o["mc_exch"]["rows"] if r["channel"] == "post" and r["cell"] != "null"]
    if post:
        mac("VExCovMax", f"{100 * max(r['NWC_cov'] for r in post):.1f}"); mac("VExStratMax", f"{100 * max(r['NWC_strat'] for r in post):.1f}")
        mac("VExNcells", len(post))
    else:
        mac("VExCovMax", "[pending]"); mac("VExStratMax", "[pending]"); mac("VExNcells", "[pending]")
    for cell, tag in [("post_r0.6_d0.1", "SixTen"), ("post_r0.6_d0.2", "SixTwenty"), ("post_r0.2_d0.025", "TwoQuarter"),
                      ("post_r0.4_d0.05", "FourFive"), ("post_r0.2_d0.05", "TwoFive")]:
        for k, kt in [("NWC", "NWC"), ("single", "Single"), ("naive", "Naive"), ("NWC_cov", "Cov"), ("NWC_strat", "Strat")]:
            mac(f"VEx{kt}{tag}", pc(cell, k))
        mac(f"VExSpur{tag}", f"{rows[cell]['spurious_pct']:.0f}" if cell in rows else "[pending]")
    mm = {r["scenario"]: r for r in o["mc_mod"]}
    for s, tag in [("clean", "Clean"), ("trend", "Trend"), ("proxy", "Proxy"), ("nonlinear", "Nonlin"), ("all", "All")]:
        for k, kt in [("perm_free", "PermFree"), ("perm_fix", "PermFix"), ("residperm_free", "ResFree"), ("residperm_fix", "ResFix"),
                      ("FL_free", "FLFree"), ("FL_fix", "FLFix"), ("nominal_fix", "NomFix"), ("nominal_free", "NomFree")]:
            mac(f"VMm{kt}{tag}", f"{100 * mm[s][k]:.1f}" if s in mm else "[pending]")
    have = {l.split("{")[1].split("}")[0].lstrip("\\") for l in L[1:]}
    for k in ["VBcovA", "VcovAph", "VcovApkg", "VcovAPotU", "VcovAPotFree", "VcovAhQ", "VcovAhMax", "VcovARsq", "VTypeAhMax",
              "VpTmax", "VSecWorld", "VSpeedup", "VFitsWorld", "VPyfixestWorld", "VPyfixestModel", "VExSingleNull", "VExNaiveNull"]:
        if k not in have:
            mac(k, "[pending]")
    (ROOT / "manuscript/numbers_validity.tex").write_text("\n".join(L) + "\n")


def write_tables(o, ex):
    # exchangeability Monte Carlo
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize",
         "\\caption{Monte Carlo: rejection rates (\\%) when exposure is not exchangeable within sector}\\label{tab:v_exch}",
         "\\setlength{\\tabcolsep}{4.5pt}\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         " & & & Spurious & \\multicolumn{3}{c}{NWC worlds} & \\multicolumn{2}{c}{Nominal} \\\\",
         "\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
         "Trait channel & $\\rho$ & $\\delta$ & effect (\\%) & within sector & strata on $u$ & $u$-preserving & one regression & naive loop \\\\",
         "\\midrule"]
    ex = ex.copy(); ex["ord"] = ex.channel.map({"post": 1, "level": 0}); ex = ex.sort_values(["ord", "rho", "delta"])
    for _, r in ex.iterrows():
        ch = "none (null)" if r.cell == "null" else ("level only" if r.channel == "level" else "post-shock shift")
        L.append(f"{ch} & {r.rho:.1f} & {r.delta:g} & {r.spurious_pct:.0f} & {100*r.NWC:.1f} & {100*r.NWC_strat:.1f} & "
                 f"{100*r.NWC_cov:.1f} & {100*r.single:.1f} & {100*r.naive:.1f} \\\\")
    R_, B_ = int(ex.R.max()), int(ex.B.max())
    L += ["\\bottomrule", "\\end{tabular}",
          "\\begin{minipage}{0.97\\linewidth}\\vspace{2pt}\\scriptsize\\textit{Notes.} "
          f"$R={R_}$ synthetic twins per row, $B={B_}$ null worlds per twin; Monte Carlo standard error about 1.5 points at 5\\%. "
          "Exposure has no effect. A firm trait $u$ with within-sector correlation $\\rho$ with exposure shifts the outcome either in levels "
          "(1 between-firm SD) or after the shock ($\\delta$ within-firm SDs). Spurious effect: the mean estimate of the pre-specified "
          "regression as a share of the real H1 estimate. NWC: within-sector permutation (the application's Type A worlds); strata on $u$: "
          "within sector $\\times$ quintile of $u$; $u$-preserving: exposure = projection on (sector, $u$) + permuted residual.\\end{minipage}",
          "\\end{table}"]
    (ROOT / "tables/tab_v_exch.tex").write_text("\n".join(L))
    # moderator Monte Carlo
    lab = {"clean": "No other structure", "trend": "Own post-shock trend of $M_k$", "proxy": "Correlated moderator, true interaction",
           "nonlinear": "Concave exposure effect", "all": "All three"}
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize",
         "\\caption{Monte Carlo: rejection rates (\\%) of moderator tests when the true interaction is zero}\\label{tab:v_modmc}",
         "\\setlength{\\tabcolsep}{4.5pt}\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         " & \\multicolumn{4}{c}{With search ($S^{\\text{free}}$)} & \\multicolumn{4}{c}{One specification ($S^{\\text{fix}}$)} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
         "Scenario & Nominal & Perm. & Cond. & Null m. & Nominal & Perm. & Cond. & Null m. \\\\", "\\midrule"]
    for r in o["mc_mod"]:
        L.append(f"{lab[r['scenario']]} & {100*r['nominal_free']:.1f} & {100*r['perm_free']:.1f} & {100*r['residperm_free']:.1f} & "
                 f"{100*r['FL_free']:.1f} & {100*r['nominal_fix']:.1f} & {100*r['perm_fix']:.1f} & {100*r['residperm_fix']:.1f} & {100*r['FL_fix']:.1f} \\\\")
    if o["mc_mod"]:
        R_, B_ = o["mc_mod"][0]["R"], o["mc_mod"][0]["B"]
    else:
        R_, B_ = 0, 0
    L += ["\\bottomrule", "\\end{tabular}",
          "\\begin{minipage}{0.97\\linewidth}\\vspace{2pt}\\scriptsize\\textit{Notes.} "
          f"$R={R_}$ synthetic twins per row with a planted main effect (0.10 within-firm SD), $B={B_}$ worlds per construction. Target: "
          "hierarchy shape, whose true interaction is zero. Own trend: $+0.10$ SD $\\times M_k\\times$Post; correlated moderator: digitizable work "
          "($r=-0.60$ with $M_k$) with interaction $-0.06$ SD; concave exposure: $-0.05$ SD $\\times(E^2-1)\\times$Post. Signs are adversarial: "
          "the leakage runs in $M_k$'s registered direction. Perm.: within sector $\\times$ exposure quintile (Type B); Cond.: conditional permutation, "
          "only the part of $M_k$ not explained by sector, exposure, exposure$^2$ and the other nine moderators is permuted; Null m.: null-model "
          "Freedman--Lane worlds with residuals sign-flipped by firm.\\end{minipage}", "\\end{table}"]
    (ROOT / "tables/tab_v_modmc.tex").write_text("\n".join(L))
    # real-data moderator p-values
    c, d = o["constructions"], o["decomposition"]
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize",
         "\\caption{Search-adjusted $p$-values of the reported moderators, by world construction and statistic}\\label{tab:v_modreal}",
         "\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{5.3cm}rrr}", "\\toprule",
         " & & H2 hierarchy & H3 digitizable & H4 organiz. \\\\", " & & shape & work & age \\\\", "\\midrule",
         "\\multicolumn{2}{l}{Observed $t$ in the package ($S^{\\text{sel}}=S^{\\text{fix}}$)} & " + " & ".join(f"{d[m]['t_obs']:.2f}" for m in MODS3) + " \\\\",
         "\\multicolumn{2}{l}{Observed largest screening $t$ ($S^{\\text{free}}$)} & " + " & ".join(f"{d[m]['free_obs']:.2f}" for m in MODS3) + " \\\\",
         "\\multicolumn{2}{l}{Share of permuted worlds reporting the moderator} & " + " & ".join(f"{100*d[m]['share_selected']:.1f}\\%" for m in MODS3) + " \\\\",
         "\\midrule"]
    names = {"permB": "Permute (Type B)", "residB": "Conditional permute", "fl": "Null model (Freedman--Lane)"}
    for cc in ["permB", "residB", "fl"]:
        if not all(c.get(cc, {}).get(m, {}).get("B", 0) >= 500 for m in MODS3):
            continue
        for s, sl in [("p_sel", "$S^{\\text{sel}}$ (reported package)"), ("p_free", "$S^{\\text{free}}$ (screening, all worlds)"),
                      ("p_fix", "$S^{\\text{fix}}$ (reported specification)")]:
            L.append(f"{names[cc] if s == 'p_sel' else ''} & {sl} & " + " & ".join(fp(c[cc][m][s]) for m in MODS3) + " \\\\")
        L.append(f" & Holm over ten candidates ($S^{{\\text{{sel}}}}$) & " + " & ".join(fp(c[cc][m]["holm10"]) for m in MODS3) + " \\\\")
        L.append("\\addlinespace")
    if "familywise" in o:
        f = o["familywise"]
        L.append(f"Joint (all ten permuted) & max-$T$ step-down (planned), $S^{{\\text{{sel}}}}$ & " + " & ".join(fp(f["maxT_stepdown_sel"][m]) for m in MODS3) + " \\\\")
        L.append(f" & min-$P$ step-down, $S^{{\\text{{sel}}}}$ & " + " & ".join(fp(f["minP_sel"][m]["stepdown"]) for m in MODS3) + " \\\\")
        L.append(f" & min-$P$ step-down, $S^{{\\text{{free}}}}$ (all ten tested) & " + " & ".join(fp(f["minP_free"][m]["stepdown"]) for m in MODS3) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\begin{minipage}{0.97\\linewidth}\\vspace{2pt}\\scriptsize\\textit{Notes.} $p=(1+\\#\\{b: S_b\\ge S_{\\text{obs}}\\})/(B+1)$ with $B=500$ "
          "worlds per moderator and construction ($B=" + (f"{o['familywise']['B']:,}" if 'familywise' in o else '') + "$ joint worlds); the smallest attainable $p$ is "
          "0.002 (0.001 for joint worlds). $S^{\\text{sel}}$ is $-\\infty$ when the loop does not report the moderator, so its $p$ combines strength and "
          "selection. Holm over ten treats the seven unreported candidates as $p=1$.\\end{minipage}", "\\end{table}"]
    (ROOT / "tables/tab_v_modreal.tex").write_text("\n".join(L))
    # covariate-preserving worlds on the real data
    if "covA" in o:
        a, cv = o["covA"]["typeA"], o["covA"]["covA"]
        L = ["\\begin{table}[!htbp]\\centering\\footnotesize",
             "\\caption{Real data: Type A worlds versus covariate-preserving worlds}\\label{tab:v_covA}",
             "\\begin{tabular}{lrr}", "\\toprule", " & Type A (within sector) & Covariate-preserving \\\\", "\\midrule",
             f"Worlds & {a['B']:,} & {cv['B']:,} \\\\",
             f"Largest null H1 $t$: median / 95th pct.\\ / max & {a['h1_median']:.1f} / {a['h1_q95']:.1f} / {a['h1_max']:.1f} & "
             f"{cv['h1_median']:.1f} / {cv['h1_q95']:.1f} / {cv['h1_max']:.1f} \\\\",
             f"Search-adjusted $p$, H1 (real $t={json.loads((ROOT / 'intermediate/10_loop_real.json').read_text())['U']['h1_max']:.1f}$) & {fp(a['p_h1'])} & {fp(cv['p_h1'])} \\\\",
             f"Search-adjusted $p$, full package & {fp(a['p_pkg'])} & {fp(cv['p_pkg'])} \\\\",
             f"Potemkin rate, U (\\%) & {100*a['potemkin_U']:.1f} & {100*cv['potemkin_U']:.1f} \\\\",
             f"Potemkin rate, U-free (\\%) & {100*a['potemkin_Ufree']:.1f} & {100*cv['potemkin_Ufree']:.1f} \\\\",
             "\\bottomrule", "\\end{tabular}",
             "\\begin{minipage}{0.97\\linewidth}\\vspace{2pt}\\scriptsize\\textit{Notes.} Covariate-preserving worlds keep each exposure proxy's "
             f"within-sector projection on twelve predetermined traits (the ten moderators, pre-period size and hiring growth; within-sector $R^2$ "
             f"{100*o['covA']['within_r2']['expo_beta_z']:.0f}\\% for the main proxy) and permute only the residual within sector, so trait-driven "
             "outcome trends are present in every null world.\\end{minipage}", "\\end{table}"]
        (ROOT / "tables/tab_v_covA.tex").write_text("\n".join(L))
    else:
        (ROOT / "tables/tab_v_covA.tex").write_text("\\begin{table}[!htbp]\\centering\\footnotesize\\caption{Real data: Type A worlds "
                                                     "versus covariate-preserving worlds [pending]}\\label{tab:v_covA}\\end{table}\n")


if __name__ == "__main__":
    main()
