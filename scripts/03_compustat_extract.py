"""03 - Extract Compustat North America quarterly fundamentals (1961Q3-2026Q2
release) and build a fiscal-year firm panel for fiscal years 2012-2025.

Why quarterly: the annual release on disk (updated 2025-01-02) does not yet
contain most fiscal-2024 and any fiscal-2025 reports, while the design needs
three post-shock years (2023-2025). Flows are summed over the four fiscal
quarters (a firm-year is kept only if all four quarters are reported);
stocks are taken at the fourth fiscal quarter.
Standard screens: indfmt='INDL', datafmt='STD', consol='C', popsrc='D'
(when present), currency USD or CAD.
Outputs:
  data/raw_extracts/compustat_q.parquet   (quarter-level extract)
  data/analysis/compustat_fy.parquet      (gvkey x fyear panel)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np
from config import COMPUSTAT_QUARTERLY_CSV, RAW_EXTRACTS, ANALYSIS, duck

COLS = ["gvkey", "datadate", "fyearq", "fqtr", "fyr", "conm", "tic", "cusip", "exchg", "loc", "naics", "sic",
        "indfmt", "datafmt", "consol", "curcdq", "costat", "saleq", "revtq", "cogsq", "xsgaq", "xrdq",
        "oibdpq", "oiadpq", "niq", "ibq", "atq", "cheq", "actq", "lctq", "ltq", "dlcq", "dlttq", "ceqq",
        "seqq", "ppentq", "cshoq", "prccq", "capxy"]
FLOWS = ["saleq", "revtq", "cogsq", "xsgaq", "xrdq", "oibdpq", "oiadpq", "niq", "ibq"]
STOCKS = ["atq", "cheq", "actq", "lctq", "ltq", "dlcq", "dlttq", "ceqq", "seqq", "ppentq", "cshoq", "prccq"]


def main():
    q_out = RAW_EXTRACTS / "compustat_q.parquet"
    if not q_out.exists():
        con = duck(memory="3GB", threads=4)
        sel = ", ".join(COLS)
        con.execute(f"""COPY (SELECT {sel} FROM read_csv('{COMPUSTAT_QUARTERLY_CSV}', header=true, all_varchar=true)
                             WHERE indfmt='INDL' AND datafmt='STD' AND consol='C'
                               AND curcdq IN ('USD','CAD') AND try_cast(fyearq AS INT) BETWEEN 2010 AND 2026)
                        TO '{q_out}' (FORMAT PARQUET)""")
    q = pd.read_parquet(q_out)
    num = FLOWS + STOCKS + ["fyearq", "fqtr", "fyr", "capxy"]
    for c in num:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["gvkey"] = pd.to_numeric(q.gvkey, errors="coerce").astype("Int64")
    q["datadate"] = pd.to_datetime(q.datadate, errors="coerce")
    q = q.dropna(subset=["gvkey", "fyearq", "fqtr"]).sort_values(["gvkey", "fyearq", "fqtr", "datadate"])
    q = q.drop_duplicates(["gvkey", "fyearq", "fqtr"], keep="last")
    g = q.groupby(["gvkey", "fyearq"])
    fy = g[FLOWS].sum(min_count=4)
    nq = g.fqtr.nunique().rename("n_quarters")
    q4 = q[q.fqtr == 4].set_index(["gvkey", "fyearq"])
    fy = fy.join(nq).join(q4[STOCKS + ["datadate", "conm", "tic", "naics", "sic", "loc", "exchg", "curcdq", "capxy"]])
    fy = fy[fy.n_quarters == 4].reset_index().rename(columns={"fyearq": "fyear"})
    fy.columns = [c[:-1] if c.endswith("q") and c not in ("curcdq",) and c[:-1] + "q" in FLOWS + STOCKS else c
                  for c in fy.columns]
    fy["mktval"] = fy.prcc * fy.csho
    fy = fy[(fy.fyear >= 2012) & (fy.fyear <= 2025)]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    fy.to_parquet(ANALYSIS / "compustat_fy.parquet", index=False)
    print(fy.shape, fy.gvkey.nunique()); print(fy.groupby("fyear").size().to_string())


if __name__ == "__main__":
    main()
