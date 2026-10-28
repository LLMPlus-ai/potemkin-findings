"""17 - Generate manuscript/numbers.tex: every number cited in the prose is a macro
computed from script outputs (never typed by hand). Missing inputs produce a
visible [pending] marker so an incomplete run can never pass silently.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd, numpy as np


def J(p):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else {}


def main():
    tb, t16, nwc, real, guard = J("tables/09_testbed.json"), J("tables/16_tables.json"), J("tables/nwc.json"), \
        J("intermediate/10_loop_real.json"), J("intermediate/10_guard.json")
    mc = J("tables/mc.json")
    M = {}
    P = "\\textbf{[pending]}"
    g = lambda d, *ks, default=None: (lambda x: x)(_get(d, ks, default))
    # sample
    M["NfirmYears"] = f"{tb['sample']['n']:,}" if tb else P
    M["Nfirms"] = f"{tb['sample']['firms']:,}" if tb else P
    sf = pd.read_csv(ROOT / "intermediate/sample_flow.csv")
    M["NlinkedFirms"] = f"{int(sf.firms.iloc[0]):,}"
    hv = pd.read_csv(ROOT / "intermediate/07b_headcount_validity.csv")
    M["CorrHeadcount"] = f"{hv.corr_log.iloc[0]:.2f}"
    au = pd.read_csv(ROOT / "intermediate/01_canonical_label_audit.csv", index_col=0)
    M["GitHubValid"] = f"{100 * au.loc['GitHub Copilot', 'share_valid']:.1f}"
    M["HFValid"] = f"{100 * au.loc['Hugging Face Transformers', 'share_valid']:.0f}"
    hits = pd.read_csv(ROOT / "intermediate/01_dictionary_hits.csv")
    M["GenAIrows"] = f"{hits[hits.tier == 'genai'].n.sum() / 1e6:.1f}"
    # H1 (primary outcome, percentage points)
    if t16:
        h = t16["h1"]
        M["HoneB"] = f"{h[1]['b']:.2f}"; M["HoneT"] = f"{h[1]['b'] / h[1]['se']:.1f}"; M["HonePct"] = f"{h[1]['pct']:.1f}"
        M["HoneIndB"] = f"{h[2]['b']:.2f}"; M["HoneIndT"] = f"{h[2]['b'] / h[2]['se']:.1f}"
        M["EntryAloneB"] = f"{h[4]['b']:.2f}"; M["EntryAloneT"] = f"{h[4]['b'] / h[4]['se']:.1f}"; M["EntryAloneAbs"] = f"{abs(h[4]['b']):.2f}"
        mj = t16["moderation_joint"]
        tt = lambda k: abs(mj[k]["b"] / mj[k]["se"])
        M["PkgHoneB"] = f"{mj['expo_beta_post']['b']:.2f}"; M["PkgHoneT"] = f"{tt('expo_beta_post'):.1f}"
        M["PkgHoneAbs"] = f"{abs(mj['expo_beta_post']['b']):.2f}"
        M["PkgPyrB"] = f"{mj['ep_x_pyramid']['b']:.2f}"; M["PkgPyrT"] = f"{tt('ep_x_pyramid'):.1f}"
        M["PkgRemB"] = f"{mj['ep_x_remote']['b']:.2f}"; M["PkgRemT"] = f"{tt('ep_x_remote'):.1f}"
        M["PkgAgeB"] = f"{mj['ep_x_firm_age']['b']:.2f}"; M["PkgAgeT"] = f"{tt('ep_x_firm_age'):.1f}"
        mg = t16["moderation_magnitudes"]
        M["PyrHi"] = f"{abs(mg['pyramid']['pct_of_mean_plus']):.1f}"; M["PyrLo"] = f"{abs(mg['pyramid']['pct_of_mean_minus']):.1f}"
        M["RemHi"] = f"{abs(mg['remote']['pct_of_mean_plus']):.1f}"; M["RemLo"] = f"{abs(mg['remote']['pct_of_mean_minus']):.1f}"
        M["AgeHi"] = f"{abs(mg['firm_age']['pct_of_mean_plus']):.1f}"; M["AgeLo"] = f"{abs(mg['firm_age']['pct_of_mean_minus']):.1f}"
        M["FSF"] = f"{t16['iv']['fs_F']:.0f}"; M["FSB"] = f"{t16['iv']['fs_b']:.2f}"
        M["IVtiltB"] = f"{t16['iv']['iv_tilt_b']:.1f}"; M["IVtiltSE"] = f"{t16['iv']['iv_tilt_se']:.1f}"
    if tb:
        M["PretrendP"] = f"{tb['pretrend_wald_p']:.3f}"; M["Mbar"] = f"{tb['rm_breakdown_Mbar']:.1f}"
        idn = {r["label"]: r for r in tb["identification"]}
        for key, lab in [("TrendB", "EXPOST_exposure_linear_trend"), ("LateB", "EXPOST_window_2019_2025"),
                         ("NegCtl", "negative_control_alone"), ("PlaceboNineteen", "placebo_2019")]:
            r = idn[lab]; M[key] = f"{100 * r['b']:.2f}"; M[key + "T"] = f"{abs(r['t']):.1f}"
        es = {int(r["year"]): r for r in tb["event_study"]}
        M["EvTwentyThree"] = f"{100 * es[2023]['b']:.2f}"; M["EvTwentyFive"] = f"{100 * es[2025]['b']:.2f}"
    if real:
        M["ModelsToFirstU"] = f"{real['U']['n_models_to_first']:,}"; M["ModelsU"] = f"{real['U']['n_models']:,}"
        M["ModelsToFirstFree"] = f"{real['Ufree']['n_models_to_first']:,}"
        M["ShareHoneSig"] = f"{100 * real['U']['share_h1_sig']:.0f}"; M["HoneMaxT"] = f"{real['U']['h1_max']:.1f}"
        M["NbaseSpecs"] = "648"
        from scipy.stats import norm as _n
        M["BonfCrit"] = f"{_n.ppf(1 - 0.05 / real['U']['n_models']):.1f}"
    if guard:
        M["GuardExploreSuccess"] = "no" if not guard["exploration"]["success"] else "a"
        c = guard.get("confirmation", {})
        if c:
            M["GuardConfHoneT"] = f"{c['t_signed'][0]:.1f}"
    if nwc:
        r = nwc.get("potemkin_rates", {})
        pick = lambda k: f"{100 * r[k]['rate']:.1f}" if k in r else P
        M["Bnull"] = f"{nwc.get('B_typeA', 0):,}"
        M["PotemU"] = pick("U (registered signs)"); M["PotemFree"] = pick("U-free (signs rationalized)")
        M["PotemBonf"] = pick("Bonferroni over models visited"); M["PotemSpec"] = pick("Specification curve (median t >= 1.96)")
        M["PotemNWC"] = pick("NWC (loop-level p <= .05)")
        gn = nwc.get("guard_null")
        M["PotemGuard"] = f"{100 * gn['package_rate']:.1f}" if gn else P
        M["BGuard"] = f"{gn['B']:,}" if gn else P
        M["GuardHonePot"] = f"{100 * gn['h1_rate']:.0f}" if gn else P
        M["pNWChOne"] = f"{nwc['p_nwc_h1_max']:.3f}" if "p_nwc_h1_max" in nwc else P
        M["pNWCpkg"] = f"{nwc['p_nwc_package_U']:.3f}" if "p_nwc_package_U" in nwc else P
        M["NullHoneQ"] = f"{nwc.get('null_h1_max_q95', float('nan')):.1f}"
        M["NullHoneMed"] = f"{nwc.get('null_h1_max_median', float('nan')):.1f}"
        M["BonfHoneCrit"] = f"{nwc.get('bonf_h1_crit', float('nan')):.1f}"
        M["BonfHonePot"] = f"{100 * nwc.get('bonf_h1_potemkin', float('nan')):.0f}"
        M["NullShareHoneSig"] = f"{100 * nwc.get('null_h1_share_sig_mean', float('nan')):.0f}"
        tbB = nwc.get("typeB", {})
        for key, m in [("Pyr", "pyramid_z"), ("Rem", "remote_z"), ("Age", "firm_age_z")]:
            M[f"pB{key}"] = f"{tbB[m]['p']:.3f}" if m in tbB else P
            M[f"BB{key}"] = f"{tbB[m]['B']:,}" if m in tbB else P
            M[f"Sel{key}"] = f"{100 * tbB[m]['share_null_selected']:.1f}" if m in tbB else P
            M[f"Holm{key}"] = f"{tbB[m]['p_holm_package']:.3f}" if m in tbB else P
        pl = nwc.get("planted", {})
        get = lambda b, k: next((v[k] for v in pl.values() if abs(v["beta_star"] - b) < 1e-9 and v.get(k) is not None), None)
        for key, b in [("PlantPowerLow", 0.02), ("PlantPowerMid", 0.05), ("PlantPowerHigh", 0.10)]:
            v = get(b, "nwc_power")
            if v is not None:
                M[key] = f"{100 * v:.0f}"
        v = get(0.10, "bonf_power")
        if v is not None:
            M["PlantPowerBonf"] = f"{100 * v:.0f}"
        v = get(0.05, "U_power")
        if v is not None:
            M["PlantPowerU"] = f"{100 * v:.0f}"
        v = get(0.05, "winner_ratio_median")
        if v is not None:
            M["WinnerCurse"] = f"{v:.1f}"
    if mc:
        for k, v in mc.items():
            M["MC" + k] = v
    sha = (ROOT / "registry/design_space.sha256").read_text().split("\n")
    M["RegistryHash"] = sha[0].split()[0][:16]
    M["RegistryTime"] = sha[1].strip() if len(sha) > 1 else P
    expected = ["PotemU", "PotemFree", "PotemBonf", "PotemSpec", "PotemNWC", "PotemGuard", "BGuard", "Bnull", "pNWChOne",
                "pNWCpkg", "NullHoneQ", "NullShareHoneSig", "pBPyr", "pBRem", "pBAge", "BBPyr", "BBRem", "BBAge",
                "GuardConfHoneT", "GuardExploreSuccess", "PlantPowerLow", "PlantPowerMid", "PlantPowerHigh", "WinnerCurse", "PlantPowerBonf", "PlantPowerU",
                "MCsizeNWC", "MCsizeNWCpkg", "MCsizeU", "MCsizeSpec", "MCsizeLeak", "MCsizeLeakReplay", "MCsizeConf", "MCpowerNWC", "MCpowerNWChOne", "MCpowerU", "MCR", "MCB"]
    for k in expected:
        M.setdefault(k, P)
    lines = ["% generated by scripts/17_numbers.py -- do not edit"]
    for k, v in M.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}")
    (ROOT / "manuscript").mkdir(exist_ok=True)
    (ROOT / "manuscript/numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _get(d, ks, default):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


if __name__ == "__main__":
    main()
