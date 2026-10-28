"""04 - Extract Revelio positions at Compustat-linked firms (2015-2025) and build
the firm-year workforce panel.

Revelio's Positions/<Y>.parquet is a yearly ACTIVE-STATE panel: a position
appears in file Y iff startdate <= Y <= enddate (missing ends treated as open).
Hence, within file Y:
  headcount(f, Y)   = positions at firm f
  hires(f, Y)       = positions at f with year(startdate) = Y
  separations(f, Y) = positions at f with year(enddate) = Y
Seniority follows Revelio's 7-step scale (1 entry, 2 junior, 3 associate,
4 manager, 5 director, 6 executive, 7 senior executive).

Firm linkage (Revelio common mapping -> Compustat gvkey): each linked rcid maps
to exactly one gvkey. If the linked rcid is its own ultimate parent, the firm
comprises all positions whose ultimate_parent_rcid equals it ("parent" mode);
otherwise the firm comprises positions at that rcid only ("entity" mode).
A position is assigned to an entity-mode firm first, so listed subsidiaries of
listed parents are not double counted. Only gvkeys present in the Compustat
North America fiscal-year panel (script 03) are kept.

Step A writes a slim per-year extract (reused by script 05 for GenAI adoption):
  data/raw_extracts/positions_linked/<Y>.parquet
Step B builds:
  data/analysis/firm_link.parquet
  data/analysis/revelio_firm_year.parquet   (firm x year workforce flows)
  data/analysis/firm_role_pre.parquet       (firm x role headcount, 2019-2021)
  data/analysis/firm_ilm_pre.parquet        (internal-fill share of manager+ hires, 2016-2019)
Resumable: finished years are skipped.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import POSITIONS_DIR, COMMON_MAP, RAW_EXTRACTS, ANALYSIS, LOGS, duck

YEARS = list(range(2015, 2026))
SLIM = RAW_EXTRACTS / "positions_linked"; SLIM.mkdir(parents=True, exist_ok=True)
log = open(LOGS / "04_positions_linked.log", "a")


def say(m):
    m = f"{time.strftime('%F %T')} {m}"; print(m, flush=True); log.write(m + "\n"); log.flush()


def build_link(con):
    out = ANALYSIS / "firm_link.parquet"
    if out.exists():
        return out
    cfy = ANALYSIS / "compustat_fy.parquet"
    con.execute(f"""COPY (
        WITH m AS (SELECT DISTINCT rcid, ultimate_parent_rcid, gvkey, hq_country, year_founded, naics_code,
                          rics_k50, rics_k200, company
                   FROM read_parquet('{COMMON_MAP}') WHERE gvkey IS NOT NULL),
             m1 AS (SELECT *, row_number() OVER (PARTITION BY rcid ORDER BY gvkey) rn FROM m),
             c AS (SELECT DISTINCT gvkey FROM read_parquet('{cfy}'))
        SELECT m1.rcid AS firm_rcid, m1.gvkey::BIGINT AS gvkey,
               CASE WHEN m1.rcid = m1.ultimate_parent_rcid THEN 'parent' ELSE 'entity' END AS mode,
               hq_country, year_founded, naics_code, rics_k50, rics_k200, company
        FROM m1 JOIN c ON m1.gvkey::BIGINT = c.gvkey WHERE rn = 1
    ) TO '{out}' (FORMAT PARQUET)""")
    n = con.execute(f"SELECT count(*), sum((mode='parent')::int) FROM read_parquet('{out}')").fetchone()
    say(f"firm_link: {n[0]:,} linked firms ({n[1]:,} parent mode)")
    return out


def extract_year(con, Y, link):
    out = SLIM / f"{Y}.parquet"
    if out.exists():
        return
    t0 = time.time()
    src = POSITIONS_DIR / f"{Y}.parquet"
    # The 2025 delivery renames role_k1500 to role_k1500_v2 (same 1,500-value
    # vocabulary; verified against postings in script 02).
    import pyarrow.parquet as pq
    role_col = "role_k1500" if "role_k1500" in pq.ParquetFile(src).schema_arrow.names else "role_k1500_v2"
    con.execute(f"""COPY (
        WITH ent AS (SELECT firm_rcid FROM read_parquet('{link}') WHERE mode='entity'),
             par AS (SELECT firm_rcid FROM read_parquet('{link}') WHERE mode='parent')
        SELECT CASE WHEN p.rcid IN (SELECT firm_rcid FROM ent) THEN p.rcid ELSE p.ultimate_parent_rcid END AS firm_rcid,
               p.user_id, p.position_id, p.startdate, p.enddate, p.seniority::TINYINT AS seniority,
               p.{role_col} AS role, p.weight::FLOAT AS weight, (p.country = 'United States') AS us,
               p.salary::FLOAT AS salary, p.remote_suitability::FLOAT AS remote
        FROM read_parquet('{src}') p
        WHERE (p.rcid IN (SELECT firm_rcid FROM ent) OR p.ultimate_parent_rcid IN (SELECT firm_rcid FROM par))
          AND coalesce(p.{role_col}, '') <> 'unemployed'
    ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    say(f"positions {Y}: {n:,} linked rows in {time.time()-t0:.0f}s")


def aggregate(con):
    fy_out = ANALYSIS / "revelio_firm_year.parquet"
    parts = []
    for Y in YEARS:
        f = SLIM / f"{Y}.parquet"
        sen = lambda cond, lab: ", ".join(
            f"sum(({cond}) AND seniority={k})::INT AS {lab}_s{k}" for k in range(1, 8))
        parts.append(f"""
          SELECT firm_rcid, {Y} AS year,
                 count(*)::INT AS hc, sum(weight) AS hc_w, sum(us::INT)::INT AS hc_us,
                 {sen('TRUE', 'hc')},
                 sum((year(startdate)={Y})::INT)::INT AS hires, sum(CASE WHEN year(startdate)={Y} THEN weight END) AS hires_w,
                 {sen(f'year(startdate)={Y}', 'hires')},
                 {sen(f'year(startdate)={Y} AND us', 'hiresus')},
                 sum((year(enddate)={Y})::INT)::INT AS seps,
                 {sen(f'year(enddate)={Y}', 'seps')},
                 avg(CASE WHEN year(startdate)={Y} AND seniority<=2 THEN salary END) AS sal_hire_entry,
                 avg(CASE WHEN year(startdate)={Y} AND seniority>=3 THEN salary END) AS sal_hire_exp,
                 avg(CASE WHEN seniority<=2 THEN salary END) AS sal_entry,
                 avg(remote) AS remote,
                 sum((startdate IS NULL)::INT)::INT AS n_nostart
          FROM read_parquet('{f}') GROUP BY firm_rcid""")
    con.execute(f"COPY ({' UNION ALL '.join(parts)}) TO '{fy_out}' (FORMAT PARQUET)")
    say("revelio_firm_year written")

    files = ", ".join(f"'{SLIM / f'{Y}.parquet'}'" for Y in (2019, 2020, 2021))
    con.execute(f"""COPY (SELECT firm_rcid, role, count(*)::INT AS n, sum(weight) AS n_w
                          FROM read_parquet([{files}]) WHERE role IS NOT NULL GROUP BY 1, 2)
                    TO '{ANALYSIS / 'firm_role_pre.parquet'}' (FORMAT PARQUET)""")
    say("firm_role_pre written")

    # Internal labor market: share of manager-and-above hires (seniority >= 4) in
    # 2016-2019 filled by a person who already held an earlier-starting position
    # at the same firm (an internal move), per Bidwell & Keller (2014).
    ilm = []
    for Y in (2016, 2017, 2018, 2019):
        f = SLIM / f"{Y}.parquet"
        ilm.append(f"""
          SELECT h.firm_rcid, {Y} AS year, count(*)::INT AS mgr_hires,
                 sum(CASE WHEN EXISTS (SELECT 1 FROM read_parquet('{f}') o
                                       WHERE o.user_id = h.user_id AND o.firm_rcid = h.firm_rcid
                                         AND o.position_id <> h.position_id AND o.startdate < h.startdate)
                          THEN 1 ELSE 0 END)::INT AS mgr_internal
          FROM read_parquet('{f}') h
          WHERE year(h.startdate) = {Y} AND h.seniority >= 4 GROUP BY 1""")
    con.execute(f"COPY ({' UNION ALL '.join(ilm)}) TO '{ANALYSIS / 'firm_ilm_pre.parquet'}' (FORMAT PARQUET)")
    say("firm_ilm_pre written")


def main():
    con = duck(memory="4GB", threads=4)
    link = build_link(con)
    for Y in YEARS:
        extract_year(con, Y, link)
    aggregate(con)
    say("DONE")


if __name__ == "__main__":
    main()
