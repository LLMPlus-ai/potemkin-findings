"""09 - Testbed estimates: H1 (difference-in-differences), event study, placebos,
negative-control exposure, confounding-shock checks, H2-H4 moderation,
two-stage least squares, and an approximate parallel-trends breakdown value.

All models: firm and year fixed effects (or firm and NAICS-3 x year), standard
errors clustered by firm, estimated with pyfixest (validated against the
nullworld engine in script 90). Economic magnitudes: effect of a 1-SD change
in exposure as a percent of the pre-period outcome mean.
Outputs: tables/*.json (numbers pulled into the manuscript) and
         intermediate/09_*.csv
"""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf, yaml

REG = yaml.safe_load((ROOT / "registry/design_space.yaml").read_text())
C1 = REG["controls"]["C1"]
C2X = [f"c2_size_{y}" for y in range(2017, 2026)] + [f"c2_growth_{y}" for y in range(2017, 2026)]
MODS = [m["name"] for m in REG["moderators"]]
SIGN = {m["name"]: (1 if m["sign"] == "amplifies" else -1) for m in REG["moderators"]}


def load():
    p = pd.read_parquet(ROOT / "data/analysis/panel.parquet")
    f = pd.read_parquet(ROOT / "data/analysis/firms.parquet")
    zc = [c for c in f.columns if c.endswith("_z")]
    p = p.merge(f[["gvkey"] + zc], on="gvkey")
    for t in ["expo_beta", "expo_human", "expo_lm", "expo_nonlang"]:
        p[f"{t}_post"] = p[f"{t}_z"] * p.post
    for m in MODS:
        p[f"{m}_post"] = p[f"{m}_z"] * p.post
        p[f"ep_x_{m}"] = p["expo_beta_post"] * p[f"{m}_z"]
    for y in range(2016, 2026):
        p[f"ev_{y}"] = p.expo_beta_z * (p.year == y)
    return p, f


def fit(fml, data, vcov=None):
    return pf.feols(fml, data=data, vcov=vcov or {"CRV1": "gvkey"})


def row(m, var, pre_mean, label, extra=None):
    b, se, pv = m.coef()[var], m.se()[var], m.pvalue()[var]
    d = dict(label=label, b=b, se=se, t=b / se, p=pv, n=int(m._N), firms=int(m._data.gvkey.nunique()) if "gvkey" in m._data else None,
             r2_within=(float(m._r2_within) if getattr(m, "_r2_within", None) is not None else None), pre_mean=pre_mean,
             pct_of_mean_per_sd=100 * b / pre_mean)
    if extra:
        d.update(extra)
    return d


def main():
    p, f = load()
    out = {}
    pre = p[p.year <= 2022]
    pm = {o: float(pre[o].mean()) for o in ["tilt_exp", "hire_sen_mean", "entry_rate"]}
    out["pre_means"] = pm
    out["sample"] = dict(n=len(p), firms=int(p.gvkey.nunique()), years=[int(p.year.min()), int(p.year.max())])
    # ---- H1 across specifications (Table A6) ----
    h1 = []
    specs = [("FE1-C0", "tilt_exp ~ expo_beta_post | gvkey + year"),
             ("FE1-C1", f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year"),
             ("FE1-C2", f"tilt_exp ~ expo_beta_post + {'+'.join(C1 + C2X)} | gvkey + year"),
             ("FE2-C1", f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + naics3_year"),
             ("FE2-C2", f"tilt_exp ~ expo_beta_post + {'+'.join(C1 + C2X)} | gvkey + naics3_year")]
    for lab, fml in specs:
        h1.append(row(fit(fml, p), "expo_beta_post", pm["tilt_exp"], lab))
    for o in ["hire_sen_mean", "entry_rate"]:
        h1.append(row(fit(f"{o} ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p), "expo_beta_post", pm[o], f"{o}-FE1-C1"))
    for t in ["expo_human", "expo_lm"]:
        h1.append(row(fit(f"tilt_exp ~ {t}_post + {'+'.join(C1)} | gvkey + year", p), f"{t}_post", pm["tilt_exp"], f"{t}-FE1-C1"))
    out["h1"] = h1
    pd.DataFrame(h1).to_csv(ROOT / "intermediate/09_h1.csv", index=False)
    print(pd.DataFrame(h1)[["label", "b", "se", "t", "p", "n", "pct_of_mean_per_sd"]].to_string(), flush=True)

    # ---- event study ----
    ev = [f"ev_{y}" for y in range(2016, 2026) if y != 2022]
    m = fit(f"tilt_exp ~ {'+'.join(ev)} + {'+'.join(C1)} | gvkey + year", p)
    es = pd.DataFrame({"year": [int(v[3:]) for v in ev], "b": m.coef()[ev].to_numpy(), "se": m.se()[ev].to_numpy()})
    es = pd.concat([es, pd.DataFrame({"year": [2022], "b": [0.0], "se": [0.0]})]).sort_values("year")
    es.to_csv(ROOT / "intermediate/09_event_study.csv", index=False)
    pre_ev = [f"ev_{y}" for y in range(2016, 2022)]
    names = list(m._coefnames); V = np.asarray(m._vcov)
    ix = [names.index(v) for v in pre_ev]
    bp = m.coef()[pre_ev].to_numpy(); Vp = V[np.ix_(ix, ix)]
    from scipy import stats as _st
    W = float(bp @ np.linalg.solve(Vp, bp))
    out["pretrend_wald_chi2"] = W; out["pretrend_wald_df"] = len(ix)
    out["pretrend_wald_p"] = float(1 - _st.chi2.cdf(W, len(ix)))
    # approximate relative-magnitudes breakdown (Rambachan & Roth 2023 spirit)
    pre_b = es[es.year <= 2022].sort_values("year").b.to_numpy()
    maxd = np.max(np.abs(np.diff(pre_b)))
    post_avg = h1[1]["b"]; post_se = h1[1]["se"]
    out["rm_breakdown_Mbar"] = float((post_avg - 1.96 * post_se) / (2 * maxd)) if maxd > 0 else None
    out["event_study"] = es.to_dict("records")
    print(es.to_string(), "\npretrend p:", out["pretrend_wald_p"], "Mbar*:", out["rm_breakdown_Mbar"], flush=True)

    # ---- placebos and negative control ----
    plc = []
    for yr in (2019, 2020):
        q = pre.copy(); q["fake"] = q.expo_beta_z * (q.year >= yr)
        plc.append(row(fit(f"tilt_exp ~ fake + {'+'.join(C1)} | gvkey + year", q), "fake", pm["tilt_exp"], f"placebo_{yr}"))
    plc.append(row(fit(f"tilt_exp ~ expo_nonlang_post + {'+'.join(C1)} | gvkey + year", p), "expo_nonlang_post", pm["tilt_exp"], "negative_control_alone"))
    mm = fit(f"tilt_exp ~ expo_beta_post + expo_nonlang_post + {'+'.join(C1)} | gvkey + year", p)
    plc.append(row(mm, "expo_nonlang_post", pm["tilt_exp"], "negative_control_joint"))
    plc.append(row(mm, "expo_beta_post", pm["tilt_exp"], "expo_beta_joint_with_negative_control"))
    # confounding shocks
    q = p[p.naics2 != 51]
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", q), "expo_beta_post", pm["tilt_exp"], "drop_information_sector"))
    q = p.copy()
    for y in range(2017, 2026):
        q[f"rem_{y}"] = q.remote_z * (q.year == y)
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1 + [f'rem_{y}' for y in range(2017, 2026)])} | gvkey + naics3_year", q),
                   "expo_beta_post", pm["tilt_exp"], "remote_x_year_and_naics3_x_year"))
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p[p.year <= 2024]), "expo_beta_post", pm["tilt_exp"], "drop_2025"))
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p[~p.year.isin([2020, 2021])]), "expo_beta_post", pm["tilt_exp"], "drop_2020_2021"))
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p[p.hq_country == "United States"]), "expo_beta_post", pm["tilt_exp"], "us_hq_only"))
    plc.append(row(fit(f"tilt_exp_us ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p[p.hires_us_all >= 10]), "expo_beta_post", float(pre.tilt_exp_us.mean()), "us_hires_only"))
    plc.append(row(pf.feols(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", data=p, vcov={"CRV1": "naics3"}), "expo_beta_post", pm["tilt_exp"], "cluster_naics3"))
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p[p.S2]), "expo_beta_post", pm["tilt_exp"], "min_30_hires"))
    # ---- EX POST checks (added after observing the 2016-2018 pre-period coefficients;
    #      see intermediate/00_design_decisions.md) ----
    q = p.copy(); q["expo_trend"] = q.expo_beta_z * (q.year - 2022)
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + expo_trend + {'+'.join(C1)} | gvkey + year", q), "expo_beta_post", pm["tilt_exp"], "EXPOST_exposure_linear_trend"))
    q2 = q[q.year >= 2019]
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", q2), "expo_beta_post", pm["tilt_exp"], "EXPOST_window_2019_2025"))
    plc.append(row(fit(f"tilt_exp ~ expo_beta_post + expo_trend + {'+'.join(C1)} | gvkey + naics3_year", q), "expo_beta_post", pm["tilt_exp"], "EXPOST_trend_and_naics3_x_year"))
    out["identification"] = plc
    pd.DataFrame(plc).to_csv(ROOT / "intermediate/09_identification.csv", index=False)
    print(pd.DataFrame(plc)[["label", "b", "se", "t", "p", "n"]].to_string(), flush=True)

    # ---- first stage and 2SLS ----
    fs = fit(f"genai_share ~ expo_beta_post + {'+'.join(C1)} | gvkey + year", p.dropna(subset=["genai_share"]))
    out["first_stage"] = row(fs, "expo_beta_post", float(pre.genai_share.mean()) if pre.genai_share.mean() > 0 else 1.0, "first_stage")
    out["first_stage"]["F"] = float(out["first_stage"]["t"] ** 2)
    iv = pf.feols(f"tilt_exp ~ {'+'.join(C1)} | gvkey + year | genai_share ~ expo_beta_post", data=p.dropna(subset=["genai_share"]), vcov={"CRV1": "gvkey"})
    out["iv"] = row(iv, "genai_share", pm["tilt_exp"], "2sls_genai_share")
    print("first stage:", out["first_stage"], "\n2SLS:", out["iv"], flush=True)

    # ---- moderation: each registered moderator singly (full sample, FE1-C1) ----
    mod = []
    for mname in MODS:
        mm = fit(f"tilt_exp ~ expo_beta_post + ep_x_{mname} + {mname}_post + {'+'.join(C1)} | gvkey + year", p)
        r = row(mm, f"ep_x_{mname}", pm["tilt_exp"], mname, dict(registered_sign=SIGN[mname]))
        r["t_signed"] = r["t"] * SIGN[mname]
        mod.append(r)
    out["moderators_single"] = mod
    # joint model with the registered-sign-consistent moderators (reported descriptively here;
    # the certified package comes from the loop and the guard, scripts 10-11)
    cand = [r["label"] for r in mod if r["t_signed"] >= 1.96]
    out["moderators_registered_sig_single"] = cand
    if len(cand) >= 3:
        jj = cand[:3]
        mj = fit(f"tilt_exp ~ expo_beta_post + {'+'.join('ep_x_'+m for m in jj)} + {'+'.join(m+'_post' for m in jj)} + {'+'.join(C1)} | gvkey + year", p)
        out["moderators_joint"] = [row(mj, "expo_beta_post", pm["tilt_exp"], "H1_in_joint")] + \
            [dict(row(mj, f"ep_x_{m}", pm["tilt_exp"], m), t_signed=float(mj.coef()[f"ep_x_{m}"] / mj.se()[f"ep_x_{m}"] * SIGN[m])) for m in jj]
        print(pd.DataFrame(out["moderators_joint"])[["label", "b", "se", "t", "p"]].to_string(), flush=True)
    pd.DataFrame(mod).to_csv(ROOT / "intermediate/09_moderators_single.csv", index=False)
    print(pd.DataFrame(mod)[["label", "b", "se", "t", "p", "registered_sign", "t_signed"]].to_string(), flush=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/09_testbed.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
