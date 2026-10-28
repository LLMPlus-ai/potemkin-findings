"""nullworld: null-world calibration of replayable AI research loops.

Modules
-------
fe      multi-way fixed-effects demeaning and cluster-robust OLS
loop    a registry-driven, replayable search loop (policies U and Ufree)
worlds  null-world (Type A / Type B) and planted-world generators
infer   loop-level randomization p-values, Holm adjustment
synth   synthetic twin panel generator
"""
import os as _os
# One BLAS thread per process: parallelism comes from worker processes, and
# multithreaded BLAS thrashes badly on a loaded machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "0.1.0"
