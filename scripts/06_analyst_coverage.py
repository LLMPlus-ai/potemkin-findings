"""06 - Analyst coverage from I/B/E/S detail (capital-market pressure moderator).

analysts(firm, year) = number of distinct analysts (ANALYS) issuing a fiscal-
year-1 EPS forecast (FPI='1', MEASURE='EPS') announced in the calendar year.
Linkage: I/B/E/S 8-character CUSIP -> first 8 characters of the Compustat
CUSIP (script 03 extract). Firms never covered are assigned zero analysts
(the absence of coverage is itself the lowest level of market scrutiny).
Outputs: data/analysis/ibes_coverage.parquet (gvkey, year, analysts)
         intermediate/06_ibes_match.csv (match audit)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import IBES_DETAIL, RAW_EXTRACTS, ANALYSIS, INTERMEDIATE, duck


def main():
    con = duck(memory="3GB", threads=4)
    out = ANALYSIS / "ibes_coverage.parquet"
    con.execute(f"""CREATE TEMP TABLE ib AS
        SELECT CUSIP AS cusip8, year(try_cast(ANNDATS AS DATE)) AS year, count(DISTINCT ANALYS) AS analysts
        FROM read_csv('{IBES_DETAIL}', header=true, all_varchar=true, quote='"', escape='"', delim=',')
        WHERE FPI='1' AND MEASURE='EPS' AND year(try_cast(ANNDATS AS DATE)) BETWEEN 2015 AND 2025
        GROUP BY 1, 2""")
    con.execute(f"""CREATE TEMP TABLE cu AS
        SELECT DISTINCT try_cast(gvkey AS BIGINT) AS gvkey, substr(cusip, 1, 8) AS cusip8
        FROM read_parquet('{RAW_EXTRACTS / 'compustat_q.parquet'}') WHERE cusip IS NOT NULL""")
    con.execute(f"""COPY (SELECT cu.gvkey, ib.year, max(ib.analysts) AS analysts
                          FROM ib JOIN cu USING (cusip8) GROUP BY 1, 2) TO '{out}' (FORMAT PARQUET)""")
    m = con.execute(f"""SELECT (SELECT count(DISTINCT cusip8) FROM ib) AS ibes_cusips,
                               (SELECT count(DISTINCT cusip8) FROM ib WHERE cusip8 IN (SELECT cusip8 FROM cu)) AS matched,
                               (SELECT count(DISTINCT gvkey) FROM read_parquet('{out}')) AS gvkeys""").df()
    m["match_rate"] = m.matched / m.ibes_cusips
    m.to_csv(INTERMEDIATE / "06_ibes_match.csv", index=False)
    print(m.to_string())


if __name__ == "__main__":
    main()
