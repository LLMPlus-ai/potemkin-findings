"""02 - Build a Revelio role (role_k1500) -> O*NET-SOC crosswalk from job postings
and attach occupation-level generative-AI exposure scores.

Revelio Positions carry only the role_k1500 taxonomy; Revelio Job Postings
carry both role_k1500_v2 (same 1,500-value vocabulary, verified: 1,498 of
1,498 position roles appear in postings) and an O*NET-SOC code. We use
pre-shock postings (2018-2022) to estimate, for each role r, the share of its
postings in each O*NET occupation o, p(o | r). Role-level exposure is then the
probability-weighted average of occupation exposure:
    E_r = sum_o p(o | r) * E_o
which avoids forcing a single modal occupation onto heterogeneous roles.

Exposure sources (data/external, public):
  - Eloundou, Manning, Mishkin & Rock (2024, Science) GPT-4-rated beta exposure
    (dv_rating_beta) and human-rated beta (human_rating_beta), O*NET-SOC level.
  - Felten, Raj & Seamans (2021, SMJ) AIOE and the language-modeling AIOE
    (Felten, Raj & Seamans 2023), 6-digit SOC.
Outputs:
  data/raw_extracts/postings_role_onet_counts.parquet  (role, onet, n)
  data/analysis/role_exposure.parquet                  (role-level exposures)
  intermediate/02_crosswalk_audit.csv                  (coverage statistics)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np
from config import POSTINGS_DIR, RAW_EXTRACTS, EXTERNAL, ANALYSIS, INTERMEDIATE, duck

YEARS = [2018, 2019, 2020, 2021, 2022]


def main():
    out_counts = RAW_EXTRACTS / "postings_role_onet_counts.parquet"
    if not out_counts.exists():
        con = duck(memory="3GB", threads=4)
        files = ", ".join(f"'{POSTINGS_DIR / f'{y}.parquet'}'" for y in YEARS)
        t0 = time.time()
        con.execute(f"""COPY (SELECT role_k1500_v2 AS role, onet_code AS onet, count(*) AS n
                              FROM read_parquet([{files}])
                              WHERE role_k1500_v2 IS NOT NULL AND onet_code IS NOT NULL
                              GROUP BY 1, 2) TO '{out_counts}' (FORMAT PARQUET)""")
        print(f"postings aggregated in {time.time()-t0:.0f}s", flush=True)
    cw = pd.read_parquet(out_counts)
    cw = cw[cw.role.ne("empty") & cw.onet.str.match(r"^\d{2}-\d{4}\.\d{2}$")]
    cw["soc6"] = cw.onet.str[:7]
    cw["p"] = cw.n / cw.groupby("role").n.transform("sum")

    # Eloundou et al. (2024): O*NET-SOC 8-digit, fall back to 6-digit mean
    el = pd.read_csv(EXTERNAL / "eloundou2024" / "occ_level.csv").rename(columns={"O*NET-SOC Code": "onet"})
    el["soc6"] = el.onet.str[:7]
    el8 = el.set_index("onet")[["dv_rating_beta", "human_rating_beta", "dv_rating_alpha", "dv_rating_gamma"]]
    el6 = el.groupby("soc6")[["dv_rating_beta", "human_rating_beta", "dv_rating_alpha", "dv_rating_gamma"]].mean()
    # Felten, Raj & Seamans: 6-digit SOC
    fa = pd.read_excel(EXTERNAL / "felten2021" / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
    fa = fa.rename(columns={"SOC Code": "soc6", "AIOE": "aioe"}).set_index("soc6")[["aioe"]]
    fl = pd.read_excel(EXTERNAL / "felten2021" / "Language_Modeling_AIOE_and_AIIE.xlsx", sheet_name="LM AIOE")
    fl = fl.rename(columns={"SOC Code": "soc6", "Language Modeling AIOE": "lm_aioe"}).set_index("soc6")[["lm_aioe"]]

    x = cw.join(el8, on="onet")
    for c in el6.columns:
        x[c] = x[c].fillna(x.soc6.map(el6[c]))
    x = x.join(fa, on="soc6").join(fl, on="soc6")
    rows, audit = [], []
    for c in ["dv_rating_beta", "human_rating_beta", "dv_rating_alpha", "dv_rating_gamma", "aioe", "lm_aioe"]:
        ok = x[c].notna()
        cov = (x.n * ok).groupby(x.role).sum() / x.groupby("role").n.sum()
        e = (x.loc[ok, "p"] * x.loc[ok, c]).groupby(x.loc[ok, "role"]).sum() / x.loc[ok, "p"].groupby(x.loc[ok, "role"]).sum()
        rows.append(e.rename(c)); audit.append(pd.DataFrame({"measure": c, "mean_posting_coverage": [cov.mean()],
                                                            "roles_scored": [e.notna().sum()]}))
    re = pd.concat(rows, axis=1)
    top = x.sort_values("n", ascending=False).drop_duplicates("role").set_index("role")
    re["modal_onet"] = top.onet; re["modal_share"] = top.p
    re["n_postings"] = x.groupby("role").n.sum()
    re.index.name = "role"
    ANALYSIS.mkdir(parents=True, exist_ok=True); INTERMEDIATE.mkdir(exist_ok=True)
    re.reset_index().to_parquet(ANALYSIS / "role_exposure.parquet", index=False)
    a = pd.concat(audit); a["n_roles"] = x.role.nunique()
    a["median_modal_share"] = re.modal_share.median()
    a.to_csv(INTERMEDIATE / "02_crosswalk_audit.csv", index=False)
    print(a.to_string()); print(re.describe().T.to_string())
    print(re.sort_values("dv_rating_beta").iloc[np.r_[0:8, -8:0]][["dv_rating_beta", "lm_aioe", "modal_onet"]])


if __name__ == "__main__":
    main()
