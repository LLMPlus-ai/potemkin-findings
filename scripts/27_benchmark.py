"""27 - What replay costs: one full replay of the registered loop (both policies, the whole design space)
with the nullworld engine versus the same number of model fits in a standard fixed-effects package
(pyfixest), on the real panel and in the same session.
Also prices the whole registered garden: the search-adjusted p of the best package anywhere in the design
space (T_max, the loop run to exhaustion) against that of the loop's actual stopping rule (T_first).
Output: tables/benchmark.json
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld import registry as R, worlds as W
from nullworld.loop import Loop
from nullworld.infer import rank_p


def main():
    reg = R.load(ROOT / "registry/design_space.yaml")
    panel = pd.read_parquet(ROOT / "data/analysis/panel.parquet")
    firms = pd.read_parquet(ROOT / "data/analysis/firms.parquet")
    g = R.grammar_from_registry(reg, panel)
    E, M, n2 = R.firm_matrices(firms, g)
    t0 = time.time(); loop = Loop(panel, g); build = time.time() - t0
    secs, fits = [], []
    for b in range(5):
        t0 = time.time()
        loop.set_world(W.type_a(E, n2, 10000 + b), M)
        reps = loop.run_both(full=True)
        secs.append(time.time() - t0); fits.append(reps["U"].n_models + reps["Ufree"].n_models)
    import pyfixest as pf
    p = panel.merge(firms[["firm_idx", "expo_beta_z", "pyramid_z", "remote_z", "firm_age_z"]], on="firm_idx")
    p["ep"] = p.expo_beta_z * p.post
    for m in ["pyramid_z", "remote_z", "firm_age_z"]:
        p["epm_" + m] = p.ep * p[m]; p["mp_" + m] = p[m] * p.post
    ctl = " + ".join(g.controls["C1"])
    fml = (f"entry_rate ~ ep + epm_pyramid_z + epm_remote_z + epm_firm_age_z + mp_pyramid_z + mp_remote_z + mp_firm_age_z + {ctl}"
           " | firm_idx + naics3_year")
    pf.feols(fml, data=p, vcov={"CRV1": "firm_idx"})
    ts = []
    for _ in range(10):
        t0 = time.time(); pf.feols(fml, data=p, vcov={"CRV1": "firm_idx"}); ts.append(time.time() - t0)
    per_model = float(np.median(ts))
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    null = pd.read_csv(ROOT / "intermediate/14_null_summary.csv")
    out = dict(engine_build_s=build, engine_world_s_median=float(np.median(secs)), engine_world_s=secs,
               fits_per_world_median=float(np.median(fits)), pyfixest_per_model_s=per_model,
               pyfixest_world_s=per_model * float(np.median(fits)),
               speedup=per_model * float(np.median(fits)) / float(np.median(secs)),
               p_T_first=rank_p(real["U"]["T_first"], null.U_T.to_numpy()),
               p_T_max=rank_p(real["U"]["T_max"], null.U_Tmax.to_numpy()),
               T_first=real["U"]["T_first"], T_max=real["U"]["T_max"])
    (ROOT / "tables/benchmark.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
