"""Load the hash-locked registry and turn it into a loop grammar.

``verify`` recomputes the SHA-256 of the registry file and compares it with the
stored hash, so that every run of the loop proves it used the frozen design
space.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .loop import Grammar

YEARS_C2 = range(2017, 2026)


def data_paths(root: Path):
    """Analysis panel and firm table: real (licensed) data by default, or the public synthetic
    twin when the environment variable NULLWORLD_DATA=synthetic. Returns (panel, firms, out_tag)."""
    import os
    if os.environ.get("NULLWORLD_DATA", "").lower() == "synthetic":
        return root / "synthetic/panel_synthetic.parquet", root / "synthetic/firms_synthetic.parquet", "synthetic_"
    return root / "data/analysis/panel.parquet", root / "data/analysis/firms.parquet", ""


def verify(registry_path: Path, hash_path: Path) -> str:
    h = hashlib.sha256(Path(registry_path).read_bytes()).hexdigest()
    stored = Path(hash_path).read_text().split()[0]
    if h != stored:
        raise RuntimeError(f"Registry hash mismatch: {h} != {stored}")
    return h


def load(registry_path: Path) -> dict:
    return yaml.safe_load(Path(registry_path).read_text())


def grammar_from_registry(reg: dict, panel: pd.DataFrame, max_triples: int = 10) -> Grammar:
    outcomes = [o["name"] for o in reg["outcomes"]]
    outdir = [1 if o["h1_sign"] == "+" else -1 for o in reg["outcomes"]]
    treatments = [t["name"] + "_z" for t in reg["treatments"]]
    mods = [m["name"] + "_z" for m in reg["moderators"]]
    msign = [1 if m["sign"] == "amplifies" else -1 for m in reg["moderators"]]
    c1 = list(reg["controls"]["C1"])
    c2 = c1 + [f"c2_size_{y}" for y in YEARS_C2] + [f"c2_growth_{y}" for y in YEARS_C2]
    return Grammar(outcomes=outcomes, outcome_dir=outdir, treatments=treatments, moderators=mods,
                   mod_sign=msign, controls={"C0": [], "C1": c1, "C2": c2},
                   fes={"FE1": ["firm_idx", "year"], "FE2": ["firm_idx", "naics3_year"]},
                   samples={"S1": "S1", "S2": "S2", "S3": "S3"}, transforms=["W0", "W1"],
                   clusters={"CL1": "firm_idx", "CL2": "naics3"}, max_triples=max_triples)


def firm_matrices(firms: pd.DataFrame, g: Grammar):
    """Firm-level treatment (F x T) and moderator (F x K) matrices ordered by firm_idx."""
    f = firms.sort_values("firm_idx")
    E = f[g.treatments].to_numpy(dtype=float)
    M = f[g.moderators].to_numpy(dtype=float)
    return E, M, f["naics2"].to_numpy()


def exposure_strata(E0: np.ndarray, naics2: np.ndarray, q: int = 5) -> np.ndarray:
    """Sector x treatment-quintile strata for Type B worlds."""
    ranks = pd.Series(E0).rank(method="first")
    quint = np.asarray(pd.qcut(ranks, q, labels=False))
    return naics2.astype(int) * 10 + quint
