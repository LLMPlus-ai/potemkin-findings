"""01 - Extract AI and generative-AI skill records from the raw Revelio User
Skills corpus (283 CSV shards, ~850 GB), one shard at a time.

Why this script exists: the construct "generative-AI adoption by a firm's
workforce" is measured from person-level skill records. Revelio's canonical
label (skill_k35000) is audited below and found to be too noisy for this
purpose (e.g., raw "github" is mapped to "GitHub Copilot"). We therefore
match an explicit, auditable dictionary (scripts/config.py) against the
lower-cased English-translated skill string (new 6-column shards) or the raw
string (legacy 4-column shards).

Memory: each shard is streamed by DuckDB with a 4 GB cap; only matched rows
(~0.3% of rows) are written. Output per shard:
  data/raw_extracts/skills_ai/<shard>.parquet
    user_id, skill_norm, tier ('genai'|'aicore'), skill_source, first_reported
and an audit tally per shard:
  data/raw_extracts/skills_ai_audit/<shard>.parquet
    canonical label x (matched by our dictionary?) counts, for the labels
    Revelio assigns to generative-AI concepts.
Re-runnable and resumable: finished shards are skipped.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SKILLS_DIR, RAW_EXTRACTS, LOGS, GENAI_EXACT, AICORE_EXACT, duck

OUT = RAW_EXTRACTS / "skills_ai"; OUT.mkdir(parents=True, exist_ok=True)
AUD = RAW_EXTRACTS / "skills_ai_audit"; AUD.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(exist_ok=True)
log = open(LOGS / "01_extract_ai_skills.log", "a")

CANON_GENAI = ["GitHub Copilot", "Generative AI", "Large Language Models", "Prompt Engineering",
               "ChatGPT", "Microsoft Copilot", "GPT-3", "Midjourney", "Hugging Face Transformers"]


def sql_list(xs):
    return ", ".join("'" + x.replace("'", "''") + "'" for x in xs)


def main():
    con = duck(memory="2GB", threads=4)
    con.execute(f"CREATE TEMP TABLE dict AS SELECT * FROM (VALUES "
                + ", ".join(f"('{s}', 'genai')" for s in GENAI_EXACT) + ", "
                + ", ".join(f"('{s}', 'aicore')" for s in AICORE_EXACT if s not in GENAI_EXACT)
                + ") t(skill_norm, tier)")
    shards = sorted(p for p in SKILLS_DIR.glob("*.csv"))
    # usage: 01_extract_ai_skills.py [worker_index n_workers]  (disjoint shard subsets)
    if len(sys.argv) == 3:
        k, n = int(sys.argv[1]), int(sys.argv[2])
        shards = [s for j, s in enumerate(shards) if j % n == k]
    for i, p in enumerate(shards):
        tag = p.stem.replace(" ", "_")
        out, aud = OUT / f"{tag}.parquet", AUD / f"{tag}.parquet"
        if out.exists() and aud.exists():
            continue
        t0 = time.time()
        header = open(p, "r", encoding="utf-8", errors="ignore").readline().strip().split(",")
        new_schema = "skill_k35000" in header
        cols = ("{'user_id':'VARCHAR','skill_raw':'VARCHAR','skill_translated':'VARCHAR','skill_source':'VARCHAR',"
                "'first_reported':'VARCHAR','skill_k35000':'VARCHAR'}" if new_schema else
                "{'user_id':'VARCHAR','skill_raw':'VARCHAR','skill_source':'VARCHAR','skill_mapped':'VARCHAR'}")
        src = (f"read_csv('{p}', header=true, columns={cols}, quote='\"', escape='\"', "
               f"ignore_errors=true, parallel=true, strict_mode=false)")
        # match on the raw string first, then on the machine translation; the
        # translation alone is unreliable (e.g., "generative ai" -> "generative to")
        raw_n = "lower(trim(skill_raw))"
        tr_n = "lower(trim(skill_translated))" if new_schema else raw_n
        norm = (f"CASE WHEN {raw_n} IN (SELECT skill_norm FROM dict) THEN {raw_n} ELSE {tr_n} END")
        fr = "try_cast(nullif(first_reported,'') AS DATE)" if new_schema else "CAST(NULL AS DATE)"
        canon = "skill_k35000" if new_schema else "skill_mapped"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE s AS
            SELECT try_cast(user_id AS BIGINT) AS user_id, {norm} AS skill_norm,
                   skill_source, {fr} AS first_reported, {canon} AS canon
            FROM {src}
            WHERE {raw_n} IN (SELECT skill_norm FROM dict) OR {tr_n} IN (SELECT skill_norm FROM dict)
               OR {canon} IN ({sql_list(CANON_GENAI)})
        """)
        con.execute(f"""COPY (SELECT s.user_id, s.skill_norm, d.tier, s.skill_source, s.first_reported
                              FROM s JOIN dict d USING (skill_norm) WHERE s.user_id IS NOT NULL)
                        TO '{out}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT canon, (skill_norm IN (SELECT skill_norm FROM dict)) AS matched,
                                     count(*) AS n
                              FROM s WHERE canon IN ({sql_list(CANON_GENAI)}) GROUP BY 1, 2)
                        TO '{aud}' (FORMAT PARQUET)""")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        msg = f"{time.strftime('%F %T')} [{i+1}/{len(shards)}] {p.name} schema={'new' if new_schema else 'legacy'} matched={n:,} in {time.time()-t0:.1f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    log.write(f"{time.strftime('%F %T')} DONE\n"); log.close()


if __name__ == "__main__":
    main()
