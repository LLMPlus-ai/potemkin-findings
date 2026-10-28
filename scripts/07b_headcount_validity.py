"""07b - Validity of Revelio headcount against Compustat employment (emp, thousands),
FY2016-2023 from the annual release. Output: intermediate/07b_headcount_validity.csv"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from config import COMPUSTAT_ANNUAL_DTA

rows = []
it = pd.read_stata(COMPUSTAT_ANNUAL_DTA, columns=["gvkey", "fyear", "emp", "indfmt", "datafmt", "consol"], iterator=True, chunksize=200000)
for ch in it:
    ch = ch[(ch.indfmt == "INDL") & (ch.datafmt == "STD") & (ch.consol == "C") & ch.fyear.between(2016, 2023)]
    rows.append(ch[["gvkey", "fyear", "emp"]])
e = pd.concat(rows); e["gvkey"] = pd.to_numeric(e.gvkey, errors="coerce")
e = e.dropna().drop_duplicates(["gvkey", "fyear"]).rename(columns={"fyear": "year"})
p = pd.read_parquet(ROOT / "data/analysis/panel.parquet")[["gvkey", "year", "hc", "naics2"]]
m = p.merge(e, on=["gvkey", "year"])
m = m[(m.emp > 0) & (m.hc > 0)]
m["lr"] = np.log(m.hc) - np.log(1000 * m.emp)
res = [dict(group="all", n=len(m), corr_log=np.corrcoef(np.log(m.hc), np.log(m.emp))[0, 1], median_coverage=float(np.exp(m.lr.median())))]
for y, g in m.groupby("year"):
    res.append(dict(group=f"year_{int(y)}", n=len(g), corr_log=np.corrcoef(np.log(g.hc), np.log(g.emp))[0, 1], median_coverage=float(np.exp(g.lr.median()))))
for s, g in m.groupby("naics2"):
    if len(g) >= 200:
        res.append(dict(group=f"naics2_{int(s)}", n=len(g), corr_log=np.corrcoef(np.log(g.hc), np.log(g.emp))[0, 1], median_coverage=float(np.exp(g.lr.median()))))
r = pd.DataFrame(res); r.to_csv(ROOT / "intermediate/07b_headcount_validity.csv", index=False); print(r.to_string())
