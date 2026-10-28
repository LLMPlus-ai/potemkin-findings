"""19 - Remaining LaTeX tables: descriptive statistics, balance, identification
checks, event study, moderator pool (single estimates), and the procedure
comparison (from tables/nwc.json when available; Holm over all ten registered moderators from tables/validity.json).
Outputs: tables/tab_desc.tex, tab_balance.tex, tab_ident.tex, tab_pool.tex, tab_procedures.tex
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd

T = ROOT / "tables"
NAMES = {"tilt_exp": "Experienced-hire share", "hire_sen_mean": "Mean hire seniority", "entry_rate": "Entry-level hires per 100 employees",
         "expo_beta": "LLM exposure (GPT-4 $\\beta$)", "genai_share": "GenAI adoption (\\% of workforce)",
         "ilm_share": "Internal labor market", "slack_cash": "Financial slack", "analysts": "Analyst coverage (log)",
         "firm_age": "Organizational age (log)", "remote": "Digitizable work", "ai_talent_pre": "Prior AI talent",
         "pyramid": "Hierarchy shape (log ratio)", "entry_wage": "Junior labor cost (log)", "hhi_ind": "Industry concentration",
         "rd_int": "Knowledge intensity", "log_hc_l1": "Log headcount$_{t-1}$", "hire_rate_l1": "Hiring rate$_{t-1}$",
         "sep_rate_l1": "Separation rate$_{t-1}$", "log_at_l1": "Log assets$_{t-1}$", "roa_l1": "ROA$_{t-1}$",
         "lev_l1": "Leverage$_{t-1}$", "sales_growth_l1": "Sales growth$_{t-1}$", "rd_at_l1": "R\\&D/assets$_{t-1}$"}


def wrap(caption, label, header, rows, note, colspec):
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize", f"\\caption{{{caption}}}\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"] + [r + " \\\\" for r in rows] + \
        ["\\bottomrule", "\\end{tabular}", f"\\begin{{minipage}}{{0.95\\linewidth}}\\vspace{{2pt}}\\scriptsize\\textit{{Notes.}} {note}\\end{{minipage}}", "\\end{table}"]
    return "\n".join(L)


def main():
    ss = pd.read_csv(T / "desc_summary.csv", index_col=0)
    rows = [f"{NAMES.get(v, v)} & {int(r['count']):,} & {r['mean']:.3f} & {r['std']:.3f} & {r['25%']:.3f} & {r['50%']:.3f} & {r['75%']:.3f}"
            for v, r in ss.iterrows()]
    (T / "tab_desc.tex").write_text(wrap("Descriptive statistics (firm-years, 2016--2025)", "tab:desc",
                                         "Variable & N & Mean & SD & P25 & Median & P75", rows,
                                         "Firm-level variables (exposure and moderators) are repeated across a firm's years. Moderators are shown before standardization.",
                                         "lrrrrrr"))
    b = pd.read_csv(T / "desc_balance.csv")
    rows = [f"{NAMES.get(r.variable, r.variable)} & {r.low:.3f} & {r.middle:.3f} & {r.high:.3f} & {r.norm_diff:+.2f}" for r in b.itertuples()]
    (T / "tab_balance.tex").write_text(wrap("Pre-period characteristics by terciles of LLM exposure", "tab:balance",
                                            "Variable (2019--2021 mean) & Low & Middle & High & Norm.\\ diff.\\ (High$-$Low)", rows,
                                            "Normalized difference = difference in means divided by the square root of the average of the two variances. Level differences are absorbed by firm fixed effects; identification rests on parallel trends (Table~\\ref{tab:ident}, Figure~\\ref{fig:es}).",
                                            "lrrrr"))
    tb = json.loads((T / "09_testbed.json").read_text())
    lab = {"placebo_2019": "Placebo shock in 2019 (pre-period data only)", "placebo_2020": "Placebo shock in 2020 (pre-period data only)",
           "negative_control_alone": "Negative control: non-language AI exposure $\\times$ Post",
           "negative_control_joint": "Negative control, jointly with LLM exposure",
           "expo_beta_joint_with_negative_control": "LLM exposure, jointly with negative control",
           "drop_information_sector": "Excluding the information sector (NAICS 51)",
           "remote_x_year_and_naics3_x_year": "Remote suitability $\\times$ year and NAICS-3 $\\times$ year FE",
           "drop_2025": "Excluding 2025 (new data delivery)", "drop_2020_2021": "Excluding 2020--2021 (COVID-19)",
           "us_hq_only": "U.S.-headquartered firms", "us_hires_only": "Outcome from U.S. hires only",
           "cluster_naics3": "Standard errors clustered by NAICS-3", "min_30_hires": "Firm-years with $\\ge$30 hires",
           "EXPOST_exposure_linear_trend": "\\textit{Ex post:} exposure-specific linear trend",
           "EXPOST_window_2019_2025": "\\textit{Ex post:} 2019--2025 window",
           "EXPOST_trend_and_naics3_x_year": "\\textit{Ex post:} trend and NAICS-3 $\\times$ year FE"}
    rows = []
    for r in tb["identification"]:
        rows.append(f"{lab.get(r['label'], r['label'])} & {100*r['b']:.2f} & ({100*r['se']:.2f}) & {r['t']:.2f} & {int(r['n']):,}")
    note = ("Outcome: experienced-hire share in percentage points; coefficient on the (placebo or control) exposure $\\times$ post term "
            "unless stated. All models include firm and year fixed effects and the lagged controls; standard errors clustered by firm. "
            f"Joint test of no pre-trends (2016--2021 event-study coefficients): $\\chi^2({tb['pretrend_wald_df']})={tb['pretrend_wald_chi2']:.1f}$, "
            f"$p={tb['pretrend_wald_p']:.3f}$; approximate relative-magnitudes breakdown value $\\bar M^*={tb['rm_breakdown_Mbar']:.2f}$ "
            "(Rambachan and Roth 2023). Rows marked \\textit{ex post} were added after the event study was inspected.")
    (T / "tab_ident.tex").write_text(wrap("Identification checks for H1", "tab:ident", "Check & Coef.\\ (pp) & (SE) & $t$ & N", rows, note, "lrrrr"))
    es = pd.DataFrame(tb["event_study"]).sort_values("year")
    rows = [f"{int(r.year)} & {100*r.b:.2f} & ({100*r.se:.2f})" for r in es.itertuples()]
    (T / "tab_es.tex").write_text(wrap("Event-study coefficients (experienced-hire share, pp)", "tab:es", "Year & Coef. & (SE)", rows,
                                       "Coefficients on exposure $\\times$ year indicators; 2022 is the reference year. Firm and year fixed effects, lagged controls, SE clustered by firm.", "lrr"))
    rows = []
    for r in tb["moderators_single"]:
        rows.append(f"{NAMES.get(r['label'], r['label'])} & {'amplifies' if r['registered_sign'] > 0 else 'attenuates'} & "
                    f"{100*r['b']:.3f} & ({100*r['se']:.3f}) & {r['t_signed']:+.2f}")
    (T / "tab_pool.tex").write_text(wrap("The registered moderator pool: single-moderator estimates on the primary outcome", "tab:pool",
                                         "Moderator & Registered sign & Coef.\\ (pp) & (SE) & Sign-consistent $t$", rows,
                                         "Outcome: experienced-hire share (pp). Each row adds one moderator's interaction with exposure $\\times$ post and its "
                                         "moderator $\\times$ post trend to the H1 model (firm and year fixed effects, lagged controls, SE clustered by firm). "
                                         "A positive sign-consistent $t$ means the estimate has the registered direction; negative values are "
                                         "significant \\emph{against} the registered theory, which an unconstrained loop could rationalize ex post.", "llrrr"))
    nf = T / "nwc.json"
    h10 = json.loads((T / "validity.json").read_text())["constructions"]["permB"]      # Holm over ten (scripts/25)
    if nf.exists():
        nwc = json.loads(nf.read_text())
        rows = []
        for k, v in nwc.get("potemkin_rates", {}).items():
            rows.append(f"{k} & {100*v['rate']:.1f} & [{100*v['lo']:.1f}, {100*v['hi']:.1f}]")
        if "guard_null" in nwc:
            g = nwc["guard_null"]
            rows.append(f"Guard: split sample, one confirmation, Holm & {100*g['package_rate']:.1f} & [{100*g['lo']:.1f}, {100*g['hi']:.1f}]")
        rows.append("\\midrule \\multicolumn{3}{l}{\\textit{Search-adjusted $p$-values for the real report}}")
        rows.append(f"H1 (maximum $t$ over the search) & {nwc.get('p_nwc_h1_max', float('nan')):.3f} & ")
        rows.append(f"Package reported by loop U & {nwc.get('p_nwc_package_U', float('nan')):.3f} & ")
        for m, v in nwc.get("typeB", {}).items():
            rows.append(f"Moderator {NAMES.get(m.replace('_z',''), m)} (Type B, B={v['B']}) & {v['p']:.3f} & Holm over ten: {h10[m]['holm10']:.3f}")
        note = (f"Top panel: share (\\%) of {nwc.get('B_typeA', 0):,} Type A null worlds (LLM exposure permuted across firms within 2-digit NAICS sectors) "
                "in which the procedure reports H1 and three moderators all significant at the 5\\% level with mixed signs; 95\\% Monte Carlo intervals. "
                "Bottom panel: rank of the real statistic among the null-world statistics produced by the same loop (Theorem 1). "
                "Holm over ten: step-down adjustment over all ten registered moderators, the seven the loop did not report "
                "counted as $p=1$ (Table~\\ref{tab:v_modreal}).")
        (T / "tab_procedures.tex").write_text(wrap("Potemkin rates and search-adjusted inference", "tab:procedures",
                                                   "Procedure & Rate / $p$ & Interval / adjusted", rows, note, "lrr"))
    # ---- article Table 1: nominal versus search-adjusted evidence for H1-H4 ----
    t16 = json.loads((T / "16_tables.json").read_text()) if (T / "16_tables.json").exists() else {}
    if nf.exists() and t16:
        nwc = json.loads(nf.read_text()); mj = t16["moderation_joint"]
        from scipy import stats as st
        tb_ = nwc.get("typeB", {})
        rows = []
        spec = [("H1", "Exposure $\\times$ Post (fewer entry-level hires)", "expo_beta_post", None, "--"),
                ("H2", "$\\times$ Hierarchy shape (amplifies)", "ep_x_pyramid", "pyramid_z", "amplifies"),
                ("H3", "$\\times$ Digitizable work (amplifies)", "ep_x_remote", "remote_z", "amplifies"),
                ("H4", "$\\times$ Organizational age (attenuates)", "ep_x_firm_age", "firm_age_z", "attenuates")]
        for h, lab_, v, m, sign in spec:
            b, se = mj[v]["b"], mj[v]["se"]; t = b / se; pn = 2 * (1 - st.norm.cdf(abs(t)))
            if m is None:
                padj = f"{nwc.get('p_nwc_h1_max', float('nan')):.3f}"; hol = "--"
            else:
                padj = f"{tb_[m]['p']:.3f}" if m in tb_ else "--"
                hol = f"{tb_[m].get('p_holm_package', float('nan')):.3f}" if m in tb_ else "--"
            pn_s = f"{pn:.3f}" if pn >= 0.001 else "<0.001"
            rows.append(f"{h} & {lab_} & {b:.3f} & ({se:.3f}) & {t:.2f} & {pn_s} & {padj} & {hol}")
        note = ("Estimates from the package the registered loop (policy U) reported on the full sample: entry-level hires per 100 employees, "
                "firm and NAICS-3 $\\times$ year fixed effects, lagged firm controls, moderator $\\times$ Post trends, SE clustered by firm "
                f"({t16['h1'][0]['n']:,} firm-years). Nominal $p$: two-sided, from the reported model. Search-adjusted $p$: rank of the real "
                "statistic among the statistics the same loop produced in null worlds---for H1 the largest H1 $t$ over the search "
                f"(Type~A, B={nwc.get('B_typeA', 0):,}); for H2--H4 the moderator's $t$ in the reported package when the moderator is permuted "
                "within sector $\\times$ exposure-quintile strata (Type~B, B=500 each). Holm: familywise adjustment across H2--H4.")
        (T / "tab_summary.tex").write_text(wrap("What the loop reported, and what survives the price of its search", "tab:summary",
                                               "& Hypothesis & Coef. & (SE) & $t$ & Nominal $p$ & Search-adj.\\ $p$ & Holm", rows, note, "llrrrrrr"))
    print("tables written")


if __name__ == "__main__":
    main()
