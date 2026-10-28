"""07 - Build the firm-year analysis panel (2016-2025) and firm-level
treatment and moderator vectors, following registry/design_space.yaml.

Inputs : data/analysis/{revelio_firm_year, firm_role_pre, firm_ilm_pre,
         firm_year_ai, role_exposure, compustat_fy, firm_link, ibes_coverage}.parquet
Outputs: data/analysis/panel.parquet          (row = firm-year)
         data/analysis/firms.parquet          (row = firm; E and M, raw and standardized)
         intermediate/sample_flow.csv
         intermediate/moderator_imputation.csv
         intermediate/leakage_audit.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from config import ANALYSIS, INTERMEDIATE

A = ANALYSIS
YEARS = range(2016, 2026)
flow = []


def step(name, df, firm="gvkey"):
    flow.append(dict(step=name, rows=len(df), firms=df[firm].nunique()))
    print(f"{name:60s} rows={len(df):>9,} firms={df[firm].nunique():>7,}", flush=True)


def main():
    link = pd.read_parquet(A / "firm_link.parquet")
    rfy = pd.read_parquet(A / "revelio_firm_year.parquet")
    rfy = rfy.merge(link[["firm_rcid", "gvkey", "year_founded", "naics_code", "hq_country"]], on="firm_rcid")
    step("Revelio firm-years at Compustat-linked firms (2015-2025)", rfy)
    S = [f"hires_s{k}" for k in range(1, 8)]
    rfy["hires_all"] = rfy[S].sum(axis=1)
    rfy["hires_exp"] = rfy[[f"hires_s{k}" for k in range(3, 8)]].sum(axis=1)
    rfy["hires_entry"] = rfy.hires_s1 + rfy.hires_s2
    rfy["tilt_exp"] = rfy.hires_exp / rfy.hires_all.replace(0, np.nan)
    rfy["hire_sen_mean"] = sum(k * rfy[f"hires_s{k}"] for k in range(1, 8)) / rfy.hires_all.replace(0, np.nan)
    rfy["hires_us_all"] = rfy[[f"hiresus_s{k}" for k in range(1, 8)]].sum(axis=1)
    rfy["tilt_exp_us"] = rfy[[f"hiresus_s{k}" for k in range(3, 8)]].sum(axis=1) / rfy.hires_us_all.replace(0, np.nan)
    rfy = rfy.sort_values(["gvkey", "year"])
    g = rfy.groupby("gvkey")
    for c in ["hc", "hires", "seps", "hires_all"]:
        rfy[c + "_l1"] = g[c].shift(1).where(g.year.shift(1) == rfy.year - 1)
    rfy["entry_rate"] = 100 * rfy.hires_entry / rfy.hc_l1
    rfy["log_hc_l1"] = np.log(rfy.hc_l1)
    rfy["hire_rate_l1"] = rfy.hires_l1 / rfy.hc_l1
    rfy["sep_rate_l1"] = rfy.seps_l1 / rfy.hc_l1

    # ---- Compustat (calendar year = fiscal year, Compustat convention) ----
    c = pd.read_parquet(A / "compustat_fy.parquet").rename(columns={"fyear": "year"})
    c = c.sort_values(["gvkey", "year"]).drop_duplicates(["gvkey", "year"], keep="last")
    c["roa"] = c.oibdp / c["at"].where(c["at"] > 0)
    c["lev"] = (c.dltt.fillna(0) + c.dlc.fillna(0)) / c["at"].where(c["at"] > 0)
    c["rd_at"] = c.xrd.fillna(0).clip(lower=0) / c["at"].where(c["at"] > 0)
    c["cash_at"] = c.che / c["at"].where(c["at"] > 0)
    c["log_at"] = np.log(c["at"].where(c["at"] > 0))
    c["log_sale"] = np.log(c.sale.where(c.sale > 0))
    cg = c.groupby("gvkey")
    c["sales_growth"] = c.log_sale - cg.log_sale.shift(1).where(cg.year.shift(1) == c.year - 1)
    lag = c[["gvkey", "year", "log_at", "roa", "lev", "sales_growth", "rd_at"]].copy()
    lag["year"] = lag.year + 1
    lag = lag.rename(columns={k: k + "_l1" for k in ["log_at", "roa", "lev", "sales_growth", "rd_at"]})
    p = rfy[(rfy.year >= 2016) & (rfy.year <= 2025)].merge(lag, on=["gvkey", "year"], how="left")
    step("Firm-years 2016-2025", p)

    # industry: Compustat NAICS (latest non-missing), else Revelio NAICS
    cn = c.dropna(subset=["naics"]).sort_values("year").groupby("gvkey").naics.last().astype(str)
    p["naics"] = p.gvkey.map(cn)
    p["naics"] = p.naics.fillna(p.naics_code.dropna().astype("Int64").astype(str))
    p = p[p.naics.notna() & (p.naics.str.len() >= 3)]
    p["naics2"] = p.naics.str[:2].astype(int)
    p["naics3"] = p.naics.str[:3].astype(int)
    p["naics3_year"] = p.naics3 * 10000 + p.year
    step("  with industry code", p)

    # ---- firm-level treatment: predetermined LLM exposure ----
    fr = pd.read_parquet(A / "firm_role_pre.parquet").merge(link[["firm_rcid", "gvkey"]], on="firm_rcid")
    re = pd.read_parquet(A / "role_exposure.parquet")
    # negative control: non-language AI exposure = residual of general AIOE on LM-AIOE across roles
    ok = re[["aioe", "lm_aioe"]].notna().all(axis=1)
    X = np.column_stack([np.ones(ok.sum()), re.loc[ok, "lm_aioe"]])
    bcoef = np.linalg.lstsq(X, re.loc[ok, "aioe"], rcond=None)[0]
    re["nonlang"] = re.aioe - (bcoef[0] + bcoef[1] * re.lm_aioe)
    fr = fr.merge(re[["role", "dv_rating_beta", "human_rating_beta", "lm_aioe", "nonlang"]], on="role", how="left")
    E = {}
    for name, col in [("expo_beta", "dv_rating_beta"), ("expo_human", "human_rating_beta"),
                      ("expo_lm", "lm_aioe"), ("expo_nonlang", "nonlang")]:
        m = fr[col].notna()
        E[name] = (fr.n[m] * fr[col][m]).groupby(fr.gvkey[m]).sum() / fr.n[m].groupby(fr.gvkey[m]).sum()
    E["expo_role_coverage"] = fr.n[fr.dv_rating_beta.notna()].groupby(fr.gvkey).sum() / fr.n.groupby(fr.gvkey).sum()
    E["pre_hc_roles"] = fr.n.groupby(fr.gvkey).sum()
    firms = pd.DataFrame(E)

    # ---- moderators (predetermined) ----
    ilm = pd.read_parquet(A / "firm_ilm_pre.parquet").merge(link[["firm_rcid", "gvkey"]], on="firm_rcid")
    ilm = ilm.groupby("gvkey")[["mgr_hires", "mgr_internal"]].sum()
    firms["ilm_share"] = (ilm.mgr_internal / ilm.mgr_hires).where(ilm.mgr_hires >= 5)
    cp = c[c.year.between(2019, 2021)].groupby("gvkey")
    firms["slack_cash"] = cp.cash_at.mean()
    firms["rd_int"] = cp.rd_at.mean()
    ib = pd.read_parquet(A / "ibes_coverage.parquet")
    ibm = ib[ib.year.between(2019, 2021)].groupby("gvkey").analysts.sum() / 3.0
    firms["analysts"] = np.log1p(ibm.reindex(firms.index).fillna(0.0))
    yf = link.set_index("gvkey").year_founded
    yf = yf.where((yf >= 1600) & (yf <= 2021))
    firms["firm_age"] = np.log1p(2022 - yf.reindex(firms.index))
    pre = rfy[rfy.year.between(2019, 2021)].groupby("gvkey")
    firms["remote"] = pre.remote.mean()
    bottom = pre.hc_s1.sum() + pre.hc_s2.sum()
    top = sum(pre[f"hc_s{k}"].sum() for k in range(4, 8))
    firms["pyramid"] = np.log(bottom.where(bottom > 0) / top.where(top > 0))
    firms["entry_wage"] = np.log(pre.sal_entry.mean().where(pre.sal_entry.mean() > 0))
    ai = pd.read_parquet(A / "firm_year_ai.parquet").merge(link[["firm_rcid", "gvkey"]], on="firm_rcid")
    aip = ai[ai.year.between(2019, 2021)].groupby("gvkey")
    firms["ai_talent_pre"] = aip.aicore_ever.sum() / aip.n_active.sum()
    # industry HHI (NAICS-3, Compustat sales, FY2019-2021)
    cc = c[c.year.between(2019, 2021) & (c.sale > 0)].copy()
    cc["n3"] = cc.naics.astype(str).str[:3]
    sh = cc.groupby(["n3", "year"]).sale.transform(lambda s: s / s.sum())
    hhi = (sh ** 2).groupby([cc.n3, cc.year]).sum().groupby(level=0).mean()
    n3map = p.drop_duplicates("gvkey").set_index("gvkey").naics3.astype(str)
    firms["hhi_ind"] = n3map.reindex(firms.index).map(hhi)
    firms["pre_log_hc"] = np.log(pre.hc.mean())
    h19 = rfy[rfy.year == 2019].set_index("gvkey").hires
    h21 = rfy[rfy.year == 2021].set_index("gvkey").hires
    firms["pre_hire_growth"] = np.log1p(h21.reindex(firms.index)) - np.log1p(h19.reindex(firms.index))

    # ---- analysis sample ----
    ctrl = ["log_hc_l1", "hire_rate_l1", "sep_rate_l1", "log_at_l1", "roa_l1", "lev_l1", "sales_growth_l1", "rd_at_l1"]
    p = p[p.gvkey.isin(firms.index[firms.expo_beta.notna()])]
    step("  with predetermined exposure (2019-2021 role mix)", p)
    p = p[p.hires_all >= 10]
    step("  with >= 10 hires (S1 outcome definition)", p)
    p = p.dropna(subset=ctrl + ["tilt_exp", "hire_sen_mean", "entry_rate"])
    for cc_ in ctrl:
        p = p[np.isfinite(p[cc_])]
    step("  with non-missing lagged controls (analysis sample S1)", p)
    # firms observed in both pre and post periods
    has = p.groupby("gvkey").year.agg(lambda y: (y.min() <= 2022) and (y.max() >= 2023))
    p = p[p.gvkey.map(has)]
    step("  observed both before and after the shock", p)
    firms = firms.loc[firms.index.isin(p.gvkey.unique())].copy()

    # imputation of missing moderators (median), recorded
    mods = ["ilm_share", "slack_cash", "analysts", "firm_age", "remote", "ai_talent_pre", "pyramid",
            "entry_wage", "hhi_ind", "rd_int"]
    imp = []
    for m in mods:
        miss = firms[m].isna() | ~np.isfinite(firms[m])
        imp.append(dict(moderator=m, n_firms=len(firms), n_imputed=int(miss.sum()), share=float(miss.mean())))
        firms[m + "_raw"] = firms[m]
        firms[m + "_miss"] = miss.astype(int)
        firms.loc[miss, m] = firms.loc[~miss, m].median()
    pd.DataFrame(imp).to_csv(INTERMEDIATE / "moderator_imputation.csv", index=False)
    for v in ["pre_log_hc", "pre_hire_growth"]:
        firms[v] = firms[v].fillna(firms[v].median())
    # standardize firm-level treatment and moderators by cross-firm SD
    for v in ["expo_beta", "expo_human", "expo_lm", "expo_nonlang"] + mods:
        firms[v + "_z"] = (firms[v] - firms[v].mean()) / firms[v].std()

    p = p.merge(firms[["pre_log_hc", "pre_hire_growth"]], left_on="gvkey", right_index=True)
    for y in range(2017, 2026):
        p[f"c2_size_{y}"] = p.pre_log_hc * (p.year == y)
        p[f"c2_growth_{y}"] = p.pre_hire_growth * (p.year == y)
    p["post"] = (p.year >= 2023).astype(int)
    p["S1"] = True
    p["S2"] = p.hires_all >= 30
    nyr = p.groupby("gvkey").year.transform("nunique")
    p["S3"] = nyr >= 8
    # generative-AI adoption (2SLS endogenous variable)
    ai2 = ai.groupby(["gvkey", "year"])[["n_active", "genai_dated", "genai_ever"]].sum().reset_index()
    p = p.merge(ai2, on=["gvkey", "year"], how="left")
    p["genai_share"] = 100 * p.genai_dated / p.n_active
    p["genai_ever_share"] = 100 * p.genai_ever / p.n_active
    fi = {g_: i for i, g_ in enumerate(firms.index)}
    p["firm_idx"] = p.gvkey.map(fi)
    firms["firm_idx"] = firms.index.map(fi)
    firms["naics2"] = p.drop_duplicates("gvkey").set_index("gvkey").naics2.reindex(firms.index)
    firms["naics3"] = p.drop_duplicates("gvkey").set_index("gvkey").naics3.reindex(firms.index)
    firms = firms.reset_index().rename(columns={"index": "gvkey"})
    p.to_parquet(A / "panel.parquet", index=False)
    firms.to_parquet(A / "firms.parquet", index=False)
    pd.DataFrame(flow).to_csv(INTERMEDIATE / "sample_flow.csv", index=False)

    # leakage audit: time windows, shared fields, correlations, VIFs
    Z = firms[[v + "_z" for v in ["expo_beta"] + mods]]
    corr = Z.corr()
    vif = pd.Series(np.diag(np.linalg.inv(corr.to_numpy())), index=corr.index)
    fields = {"expo_beta": "role_k1500 (2019-2021) x external exposure", "ilm_share": "startdate, seniority, user_id (2016-2019 manager+ hires)",
              "slack_cash": "Compustat che, at", "analysts": "I/B/E/S ANALYS", "firm_age": "Revelio year_founded",
              "remote": "Revelio remote_suitability", "ai_talent_pre": "skills x positions (2019-2021)",
              "pyramid": "seniority stock (2019-2021)", "entry_wage": "salary (2019-2021)", "hhi_ind": "Compustat sale (industry)",
              "rd_int": "Compustat xrd, at"}
    la = pd.DataFrame({"variable": ["expo_beta"] + mods,
                       "fields": [fields[v] for v in ["expo_beta"] + mods],
                       "corr_with_expo": corr.iloc[0].to_numpy(),
                       "max_abs_corr_other": [corr.drop(index=r).loc[:, r].abs().max() for r in corr.index],
                       "vif": vif.to_numpy()})
    la["outcome_fields"] = "startdate x seniority of hires, 2016-2025"
    la.to_csv(INTERMEDIATE / "leakage_audit.csv", index=False)
    print(la.to_string())
    print(pd.DataFrame(imp).to_string())
    print(p.groupby("year").agg(n=("gvkey", "size"), tilt=("tilt_exp", "mean"), genai=("genai_share", "mean")).to_string())


if __name__ == "__main__":
    main()
