"""91 - Validate the Frisch-Waugh-Lovell fast path of nullworld.loop against a direct
regression with all controls, on the real panel. Output: intermediate/91_fwl_validation.csv"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld import registry as R
from nullworld.loop import Loop
reg = R.load(ROOT / 'registry/design_space.yaml')
panel = pd.read_parquet(ROOT / 'data/analysis/panel.parquet'); firms = pd.read_parquet(ROOT / 'data/analysis/firms.parquet')
g = R.grammar_from_registry(reg, panel); E, M, n2 = R.firm_matrices(firms, g)
t0 = time.time(); L = Loop(panel, g); L.set_world(E, M); print('build+set', round(time.time() - t0, 1), 's', flush=True)
rows = []
for (s, fe) in [('S1', 'FE1'), ('S2', 'FE2')]:
    fr = L.frames[(s, fe)]
    for (w, cset, cl) in [('W0', 'C1', 'CL1'), ('W1', 'C2', 'CL2'), ('W0', 'C0', 'CL1')]:
        y = fr.Yr[(w, cset)][:, 0]; EP, MP, EPM = fr.Zr[(w, cset)]
        b1, se1, _ = fr.cl[cl].fit(y, np.column_stack([EP[:, [0]], EPM[:, 0, [6]], MP[:, [6]]]), k_extra=fr.kc[(w, cset)])
        sub = panel.iloc[fr.idx]; post = sub.post.to_numpy(float)
        e = E[sub.firm_idx.to_numpy(), 0]; m = M[sub.firm_idx.to_numpy(), 6]
        Z = fr.fe.demean(np.column_stack([e * post, e * post * m, m * post]))
        b2, se2, _ = fr.cl[cl].fit(fr.Y[w][:, 0], np.column_stack([Z, fr.Ctrl[(w, cset)]]))
        rows.append(dict(sample=s, fe=fe, transform=w, controls=cset, cluster=cl,
                         max_abs_diff_b=float(np.abs(b1 - b2[:3]).max()), max_rel_diff_se=float(np.abs(se1 / se2[:3] - 1).max())))
r = pd.DataFrame(rows); print(r.to_string(), flush=True); r.to_csv(ROOT / 'intermediate/91_fwl_validation.csv', index=False)
t0 = time.time(); rep = L.run('U', full=True)
print('U full run', round(time.time() - t0, 1), 's; success', rep.success, 'T_first', rep.T_first, 'models', rep.n_models, flush=True)
