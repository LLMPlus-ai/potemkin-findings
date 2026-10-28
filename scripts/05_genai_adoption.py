"""05 - Firm-year generative-AI adoption and pre-period AI talent.

Inputs : data/raw_extracts/skills_ai/*.parquet   (script 01)
         data/raw_extracts/positions_linked/<Y>.parquet (script 04)
Outputs: data/analysis/user_ai.parquet
           user_id, genai_first (earliest dated GenAI skill), genai_any,
           aicore_first, aicore_any
         data/analysis/firm_year_ai.parquet
           firm_rcid, year, n_active, genai_dated (active employees whose GenAI
           skill was first reported on or before 31 Dec of the year),
           genai_ever (active employees with any GenAI listing),
           aicore_ever (active employees with any core-AI listing)
         intermediate/05_adoption_audit.csv (dated share, adoption by year)

Construct notes. Generative-AI skills cannot predate 2022 in substance, so
genai_dated measures the timing of adoption where a date exists (resume-
sourced rows) and genai_ever the cumulative stock. Core-AI listings are
time-invariant profile attributes; ai_talent_pre (script 07) is their share
among the 2019-2021 workforce, the standard resume-based construction of firm
AI human capital (Babina et al. 2024).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_EXTRACTS, ANALYSIS, INTERMEDIATE, duck

YEARS = list(range(2016, 2026))


def main():
    con = duck(memory="4GB", threads=4)
    ua = ANALYSIS / "user_ai.parquet"
    if not ua.exists():
        con.execute(f"""COPY (
            SELECT user_id,
                   min(CASE WHEN tier='genai' THEN first_reported END) AS genai_first,
                   max((tier='genai')::INT)::TINYINT AS genai_any,
                   min(CASE WHEN tier='aicore' THEN first_reported END) AS aicore_first,
                   max((tier='aicore')::INT)::TINYINT AS aicore_any
            FROM read_parquet('{RAW_EXTRACTS / "skills_ai" / "*.parquet"}')
            GROUP BY user_id) TO '{ua}' (FORMAT PARQUET)""")
    print(con.execute(f"""SELECT count(*), sum(genai_any), sum((genai_first IS NOT NULL)::INT), sum(aicore_any)
                          FROM read_parquet('{ua}')""").fetchall(), flush=True)
    parts = []
    for Y in YEARS:
        f = RAW_EXTRACTS / "positions_linked" / f"{Y}.parquet"
        parts.append(f"""
          SELECT p.firm_rcid, {Y} AS year, count(*)::INT AS n_active,
                 sum((u.genai_first IS NOT NULL AND u.genai_first <= DATE '{Y}-12-31')::INT)::INT AS genai_dated,
                 sum(coalesce(u.genai_any, 0))::INT AS genai_ever,
                 sum(coalesce(u.aicore_any, 0))::INT AS aicore_ever
          FROM read_parquet('{f}') p LEFT JOIN read_parquet('{ua}') u USING (user_id)
          GROUP BY 1""")
    t0 = time.time()
    out = ANALYSIS / "firm_year_ai.parquet"
    con.execute(f"COPY ({' UNION ALL '.join(parts)}) TO '{out}' (FORMAT PARQUET)")
    a = con.execute(f"""SELECT year, sum(n_active) n, sum(genai_dated)/sum(n_active) dated_share,
                               sum(genai_ever)/sum(n_active) ever_share, sum(aicore_ever)/sum(n_active) aicore_share
                        FROM read_parquet('{out}') GROUP BY 1 ORDER BY 1""").df()
    a.to_csv(INTERMEDIATE / "05_adoption_audit.csv", index=False)
    print(a.to_string(), f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
