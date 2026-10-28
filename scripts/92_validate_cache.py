"""92 - Validate exact caching for Type B worlds: cached and uncached replays of the
loop must give identical reports. Output: intermediate/92_cache_validation.csv"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld import registry as R, worlds as W
from nullworld.loop import Loop
reg = R.load(ROOT / 'registry/design_space.yaml')
panel = pd.read_parquet(ROOT / 'data/analysis/panel.parquet'); firms = pd.read_parquet(ROOT / 'data/analysis/firms.parquet')
g = R.grammar_from_registry(reg, panel); E, M, n2 = R.firm_matrices(firms, g)
L = Loop(panel, g); L.set_world(E, M)
rec = {}; t0 = time.time(); L.run_both(full=True, record=rec); print('record pass', round(time.time() - t0, 1), 's', flush=True)
strata = R.exposure_strata(E[:, 0], n2)
rows = []
for m in ['pyramid_z', 'firm_age_z']:
    k = g.moderators.index(m)
    for b in [1, 2]:
        Mb = W.type_b(M, k, strata, b); L.set_world(E, Mb)
        t0 = time.time(); a = L.run_both(full=False); t1 = time.time()
        c = L.run_both(full=False, cache=rec, recompute=(k,)); t2 = time.time()
        for pol in ['U', 'Ufree']:
            rows.append(dict(moderator=m, world=b, policy=pol, same_success=a[pol].success == c[pol].success,
                             same_T=bool(np.isclose(a[pol].T_first, c[pol].T_first, rtol=0, atol=1e-10) or a[pol].T_first == c[pol].T_first),
                             same_pkg=a[pol].first_success == c[pol].first_success, same_models=a[pol].n_models == c[pol].n_models,
                             sec_uncached=round(t1 - t0, 1), sec_cached=round(t2 - t1, 1)))
        print(rows[-2:], flush=True)
r = pd.DataFrame(rows); r.to_csv(ROOT / 'intermediate/92_cache_validation.csv', index=False); print(r.to_string())
