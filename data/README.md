# Data provenance

Every file below is produced by the numbered script shown, from the source listed in `scripts/config.py`. The licensed sources are read in place and are never modified.

| File | Script | Rows | Columns | MB | Built |
|---|---|---|---|---|---|
| `data/analysis/compustat_fy.parquet` | 03 | 160,378 | 34 | 16.2 | 2026-09-28 |
| `data/analysis/firm_ilm_pre.parquet` | 04 | 34,289 | 4 | 0.3 | 2026-09-29 |
| `data/analysis/firm_link.parquet` | 04 | 11,921 | 9 | 0.4 | 2026-09-29 |
| `data/analysis/firm_role_pre.parquet` | 04 | 3,518,151 | 4 | 39.6 | 2026-09-29 |
| `data/analysis/firm_year_ai.parquet` | 05 | 107,676 | 6 | 1.0 | 2026-09-29 |
| `data/analysis/firms.parquet` | 07 | 4,078 | 56 | 1.2 | 2026-09-29 |
| `data/analysis/ibes_coverage.parquet` | 06 | 47,556 | 3 | 0.2 | 2026-09-29 |
| `data/analysis/panel.parquet` | 07 | 34,333 | 99 | 9.7 | 2026-09-29 |
| `data/analysis/revelio_firm_year.parquet` | 04 | 118,346 | 41 | 11.5 | 2026-09-29 |
| `data/analysis/role_exposure.parquet` | 02 | 1,500 | 10 | 0.1 | 2026-09-28 |
| `data/analysis/user_ai.parquet` | 05 | 8,495,027 | 5 | 62.7 | 2026-09-29 |
| `data/raw_extracts/compustat_q.parquet` | 03 | 780,486 | 39 | 64.1 | 2026-09-28 |
| `data/raw_extracts/postings_role_onet_counts.parquet` | 02 | 1,057,468 | 3 | 4.8 | 2026-09-28 |

Additional extracts:
- `data/raw_extracts/skills_ai/` holds 283 per-shard files of AI and generative-AI skill records (script 01). `skills_ai_audit/` holds the canonical-label audits.
- `data/raw_extracts/positions_linked/<year>.parquet` holds positions at Compustat-linked firms, 2015–2025, about 14 GB (script 04).
- `data/external/eloundou2024/occ_level.csv` comes from https://github.com/openai/GPTs-are-GPTs (accessed 2026-09-28).
- `data/external/felten2021/*.xlsx` comes from https://github.com/AIOE-Data/AIOE (accessed 2026-09-28).
