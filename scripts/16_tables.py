"""16 - LaTeX tables for the article and technical appendix, estimated on the
full sample with all controls and fixed effects (pyfixest; SE clustered by firm).
The moderation table estimates the package the registered loop (policy U)
reported on the full sample; its search-adjusted p-values come from
tables/nwc.json (script 14). Every table reports N, firms, fixed effects,
clustering, within-R2, the pre-period outcome mean, and the economic magnitude
(effect of a 1-SD exposure difference as % of the pre-period mean).
Outputs: tables/tab_*.tex, tables/16_tables.json
"""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf, yaml
import importlib
tb = importlib.import_module("09_testbed")
from nullworld.loop import winsorize_by_year

T = ROOT / "tables"
REG = yaml.safe_load((ROOT / "registry/design_space.yaml").read_text())
C1, C2X = tb.C1, tb.C2X
LAB = {"expo_beta_post": "Exposure $\\times$ Post (H1)", "genai_share": "GenAI adoption (\\%)",
       "ep_x_pyramid": "Exposure $\\times$ Post $\\times$ Hierarchy shape (H2)",
       "ep_x_remote": "Exposure $\\times$ Post $\\times$ Digitizable work (H3)",
       "ep_x_firm_age": "Exposure $\\times$ Post $\\times$ Organizational age (H4)",
       "pyramid_post": "Hierarchy shape $\\times$ Post", "remote_post": "Digitizable work $\\times$ Post",
       "firm_age_post": "Organizational age $\\times$ Post"}
HYP = {"pyramid": "H2", "remote": "H3", "firm_age": "H4"}
CL = {"log_hc_l1": "Log headcount$_{t-1}$", "hire_rate_l1": "Hiring rate$_{t-1}$", "sep_rate_l1": "Separation rate$_{t-1}$",
      "log_at_l1": "Log assets$_{t-1}$", "roa_l1": "ROA$_{t-1}$", "lev_l1": "Leverage$_{t-1}$",
      "sales_growth_l1": "Sales growth$_{t-1}$", "rd_at_l1": "R\\&D/assets$_{t-1}$"}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(x, d=3):
    return f"{x:.{d}f}"


def reg_table(models, focal, pre_means, colnames, fe_rows, caption, label, note, magnitude_var="expo_beta_post",
              show_controls=True):
    lines = ["\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + "c" * len(models) + "@{}}", "\\toprule",
             " & " + " & ".join(f"({i+1})" for i in range(len(models))) + " \\\\",
             " & " + " & ".join(colnames) + " \\\\", "\\midrule"]
    for v in focal:
        if not any(v in m.coef().index for m in models):
            continue
        r1, r2, r3 = [LAB.get(v, v)], [""], [""]
        for m in models:
            if v in m.coef().index:
                b, se, p = m.coef()[v], m.se()[v], m.pvalue()[v]
                r1.append(fmt(b, 4) + stars(p)); r2.append(f"({fmt(se, 4)})"); r3.append(f"[$p$={p:.3g}]" if p >= 0.001 else "[$p$<0.001]")
            else:
                r1.append(""); r2.append(""); r3.append("")
        lines += [" & ".join(r1) + " \\\\", " & ".join(r2) + " \\\\", " & ".join(r3) + " \\\\"]
    lines.append("\\midrule")
    if show_controls:
        lines.append("Firm controls (lagged) & " + " & ".join("Yes" if any(c in m.coef().index or c + "_w" in m.coef().index for c in C1) else "No" for m in models) + " \\\\")
        lines.append("Pre-period size, hiring growth $\\times$ year & " + " & ".join("Yes" if any(c in m.coef().index or c + "_w" in m.coef().index for c in C2X) else "No" for m in models) + " \\\\")
    for k, vals in fe_rows.items():
        lines.append(f"{k} & " + " & ".join(vals) + " \\\\")
    lines.append("Clustered SE & " + " & ".join(["Firm"] * len(models)) + " \\\\")
    lines.append("Observations & " + " & ".join(f"{int(m._N):,}" for m in models) + " \\\\")
    lines.append("Firms & " + " & ".join(f"{m._data.gvkey.nunique():,}" for m in models) + " \\\\")
    lines.append("Within $R^2$ & " + " & ".join(fmt(m._r2_within, 3) if getattr(m, "_r2_within", None) is not None else "--" for m in models) + " \\\\")
    lines.append("Pre-period mean of outcome & " + " & ".join(fmt(x, 3) for x in pre_means) + " \\\\")
    mag = []
    for m, pm in zip(models, pre_means):
        if magnitude_var in m.coef().index and abs(pm) >= 0.0005:      # no % of a zero pre-period mean
            mag.append(f"{100 * m.coef()[magnitude_var] / pm:+.1f}\\%")
        else:
            mag.append("--")
    lines.append("1-SD exposure effect, \\% of pre-period mean & " + " & ".join(mag) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", f"\\begin{{minipage}}{{0.97\\linewidth}}\\vspace{{2pt}}\\scriptsize\\textit{{Notes.}} {note}\\end{{minipage}}",
              "\\end{table}"]
    return "\n".join(lines)


def main():
    p, f = tb.load()
    p["entry_rate_w"] = winsorize_by_year(p.entry_rate.to_numpy(float), p.year.to_numpy())
    # registered transform W1 winsorizes the outcome AND the continuous controls by year (as in nullworld.loop.Frame)
    for c in C1 + C2X:
        x = p[c].to_numpy(float)
        p[c + "_w"] = winsorize_by_year(x, p.year.to_numpy()) if np.unique(x).size > 20 else x
    p["tilt_exp_pct"] = 100 * p.tilt_exp
    pre = p[p.year <= 2022]
    out = {}
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    pkg = real["U"]["first_success"]
    mods = [m.replace("_z", "") for m in pkg["moderators"]]
    V = lambda m: {"CRV1": "gvkey"}
    fit = lambda fml, d=p: pf.feols(fml, data=d, vcov={"CRV1": "gvkey"})
    c1, c2 = "+".join(C1), "+".join(C1 + C2X)
    c1w, c2w = "+".join(c + "_w" for c in C1), "+".join(c + "_w" for c in C1 + C2X)
    # ---- Table: H1 main effect ----
    ms = [fit(f"tilt_exp_pct ~ expo_beta_post | gvkey + year"),
          fit(f"tilt_exp_pct ~ expo_beta_post + {c1} | gvkey + year"),
          fit(f"tilt_exp_pct ~ expo_beta_post + {c2} | gvkey + naics3_year"),
          fit(f"hire_sen_mean ~ expo_beta_post + {c1} | gvkey + year"),
          fit(f"entry_rate_w ~ expo_beta_post + {c1w} | gvkey + naics3_year"),
          fit(f"entry_rate_w ~ expo_beta_post + {c2w} | gvkey + naics3_year")]
    pms = [100 * pre.tilt_exp.mean()] * 3 + [pre.hire_sen_mean.mean()] + [pre.entry_rate_w.mean()] * 2
    fe = {"Firm FE": ["Yes"] * 6, "Year FE": ["Yes", "Yes", "--", "Yes", "--", "--"],
          "Industry (NAICS-3) $\\times$ year FE": ["No", "No", "Yes", "No", "Yes", "Yes"]}
    cols = ["Exp. share", "Exp. share", "Exp. share", "Mean sen.", "Entry rate", "Entry rate"]
    note = ("Unit: firm-year, 2016--2025. Exposure is the firm's 2019--2021 role-mix-weighted GPT-4 $\\beta$ exposure "
            "(Eloundou et al. 2024), standardized across firms; Post = 1 for 2023--2025. Outcomes: share of hires at associate "
            "level or above (\\%, columns 1--3); mean seniority of hires (1--7, column 4); entry-level hires per 100 employees "
            "(winsorized at 1/99 by year, columns 5--6; the registered H1 sign is negative). Standard errors clustered by firm in "
            "parentheses; two-sided $p$-values in brackets. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.")
    (T / "tab_h1.tex").write_text(reg_table(ms, ["expo_beta_post"], pms, cols, fe,
                                  "Generative-AI exposure and the experience tilt of hiring (H1)", "tab:h1", note))
    out["h1"] = [dict(b=float(m.coef()["expo_beta_post"]), se=float(m.se()["expo_beta_post"]), p=float(m.pvalue()["expo_beta_post"]),
                      n=int(m._N), pct=float(100 * m.coef()["expo_beta_post"] / pm)) for m, pm in zip(ms, pms)]
    # ---- Table: moderation (the loop's reported package, full sample) ----
    base = f"entry_rate_w ~ expo_beta_post"
    mm = [fit(f"{base} + {c1w} | gvkey + naics3_year")]
    for m in mods:
        mm.append(fit(f"{base} + ep_x_{m} + {m}_post + {c1w} | gvkey + naics3_year"))
    mm.append(fit(f"{base} + {'+'.join('ep_x_' + m for m in mods)} + {'+'.join(m + '_post' for m in mods)} + {c1w} | gvkey + naics3_year"))
    mm.append(fit(f"{base} + {'+'.join('ep_x_' + m for m in mods)} + {'+'.join(m + '_post' for m in mods)} + {c2w} | gvkey + naics3_year"))
    focal = ["expo_beta_post"] + [f"ep_x_{m}" for m in mods] + [f"{m}_post" for m in mods]
    fe = {"Firm FE": ["Yes"] * len(mm), "Industry (NAICS-3) $\\times$ year FE": ["Yes"] * len(mm)}
    note = ("Outcome: entry-level hires per 100 employees (winsorized at 1/99 by year); a negative coefficient on Exposure $\\times$ Post "
            "means fewer entry-level hires. Moderators are measured before the shock (2016--2021) and standardized across firms; "
            "their main effects are absorbed by firm fixed effects, and Moderator $\\times$ Post terms absorb differential trends. "
            "Registered signs: hierarchy shape amplifies (more negative), digitizable work amplifies, organizational age attenuates "
            "(less negative). Column 5 is the package reported by the registered loop (policy U) after "
            f"{real['U']['n_models_to_first']:,} models; search-adjusted $p$-values are in Table~\\ref{{tab:procedures}}. "
            "Standard errors clustered by firm. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.")
    (T / "tab_moderation.tex").write_text(reg_table(mm, focal, [pre.entry_rate_w.mean()] * len(mm),
                                          ["H1"] + ["+" + HYP[m] for m in mods] + ["Joint", "Joint+C2"], fe,
                                          "Organizational moderators of the exposure effect on entry-level hiring (H2--H4)", "tab:moderation", note))
    out["moderation_joint"] = {v: dict(b=float(mm[-2].coef()[v]), se=float(mm[-2].se()[v]), p=float(mm[-2].pvalue()[v])) for v in focal}
    out["moderation_joint_C2"] = {v: dict(b=float(mm[-1].coef()[v]), se=float(mm[-1].se()[v]), p=float(mm[-1].pvalue()[v])) for v in focal}
    # economic magnitudes of moderation: effect of exposure at +/-1 SD of each moderator
    b0 = mm[-2].coef()["expo_beta_post"]
    out["moderation_magnitudes"] = {m: dict(at_minus1sd=float(b0 - mm[-2].coef()[f"ep_x_{m}"]),
                                            at_plus1sd=float(b0 + mm[-2].coef()[f"ep_x_{m}"]),
                                            pct_of_mean_minus=float(100 * (b0 - mm[-2].coef()[f"ep_x_{m}"]) / pre.entry_rate_w.mean()),
                                            pct_of_mean_plus=float(100 * (b0 + mm[-2].coef()[f"ep_x_{m}"]) / pre.entry_rate_w.mean()))
                                    for m in mods}
    # ---- Table: first stage and 2SLS ----
    d = p.dropna(subset=["genai_share"])
    fs = fit(f"genai_share ~ expo_beta_post + {c1} | gvkey + year", d)
    rf1 = fit(f"tilt_exp_pct ~ expo_beta_post + {c1} | gvkey + year", d)
    iv1 = pf.feols(f"tilt_exp_pct ~ {c1} | gvkey + year | genai_share ~ expo_beta_post", data=d, vcov={"CRV1": "gvkey"})
    rf2 = fit(f"entry_rate_w ~ expo_beta_post + {c1} | gvkey + year", d)
    iv2 = pf.feols(f"entry_rate_w ~ {c1} | gvkey + year | genai_share ~ expo_beta_post", data=d, vcov={"CRV1": "gvkey"})
    ivm = [fs, rf1, iv1, rf2, iv2]
    note = ("Column 1: first stage; the outcome is the percentage of the year-$t$ workforce with a dated generative-AI skill "
            f"(first-stage $F$ = {(fs.coef()['expo_beta_post'] / fs.se()['expo_beta_post'])**2:.0f}). Columns 2 and 4: reduced forms; "
            "columns 3 and 5: two-stage least squares instrumenting adoption with Exposure $\\times$ Post. The exclusion restriction "
            "requires exposure to affect hiring only through workforce adoption; the reduced forms remain intent-to-treat estimates "
            "if it fails. Firm and year fixed effects; standard errors clustered by firm.")
    (T / "tab_iv.tex").write_text(reg_table(ivm, ["expo_beta_post", "genai_share"],
                                  [max(pre.genai_share.mean(), 1e-9), 100 * pre.tilt_exp.mean(), 100 * pre.tilt_exp.mean(), pre.entry_rate_w.mean(), pre.entry_rate_w.mean()],
                                  ["GenAI adopt.", "Exp. share", "Exp. share", "Entry rate", "Entry rate"],
                                  {"Firm FE": ["Yes"] * 5, "Year FE": ["Yes"] * 5, "Estimator": ["OLS", "OLS", "2SLS", "OLS", "2SLS"]},
                                  "Workforce generative-AI adoption: first stage and two-stage least squares", "tab:iv", note))
    out["iv"] = dict(fs_b=float(fs.coef()["expo_beta_post"]), fs_F=float((fs.coef()["expo_beta_post"] / fs.se()["expo_beta_post"]) ** 2),
                     iv_tilt_b=float(iv1.coef()["genai_share"]), iv_tilt_se=float(iv1.se()["genai_share"]),
                     iv_entry_b=float(iv2.coef()["genai_share"]), iv_entry_se=float(iv2.se()["genai_share"]))
    # ---- Table: sample flow ----
    sf = pd.read_csv(ROOT / "intermediate/sample_flow.csv")
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize", "\\caption{Construction of the analysis sample}\\label{tab:flow}",
         "\\begin{tabular}{lrr}", "\\toprule", "Step & Firm-years & Firms \\\\", "\\midrule"]
    for _, r in sf.iterrows():
        L.append(f"{r.step.strip()} & {int(r.rows):,} & {int(r.firms):,} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (T / "tab_flow.tex").write_text("\n".join(L))
    (T / "16_tables.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
