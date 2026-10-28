"""Multi-way fixed-effects demeaning and cluster-robust OLS.

Designed for many regressions on one panel: the fixed-effect structure of a
sample is factorized once (``FEStructure``); any number of columns can then be
residualized on it by alternating projections (method of alternating
projections, Gaure 2013), vectorized across columns with sparse indicator
matrices. OLS on residualized columns reproduces the within estimator (Frisch-
Waugh-Lovell).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class FEStructure:
    """Fixed effects for a fixed set of rows.

    Parameters
    ----------
    codes : list of 1-d integer arrays (one per fixed effect), length n each.
    """

    def __init__(self, codes, tol: float = 1e-10, max_iter: int = 1000):
        self.n = len(codes[0])
        self.tol, self.max_iter = tol, max_iter
        self.mats, self.inv_counts = [], []
        for c in codes:
            c = np.asarray(c)
            _, idx = np.unique(c, return_inverse=True)
            g = idx.max() + 1
            D = sp.csr_matrix((np.ones(self.n), (np.arange(self.n), idx)), shape=(self.n, g))
            cnt = np.asarray(D.sum(axis=0)).ravel()
            self.mats.append(D)
            self.inv_counts.append(1.0 / cnt)
        self.n_levels = [m.shape[1] for m in self.mats]

    def demean(self, X: np.ndarray) -> np.ndarray:
        """Residualize the columns of X on all fixed effects."""
        X = np.array(X, dtype=np.float64, copy=True)
        one_d = X.ndim == 1
        if one_d:
            X = X[:, None]
        if len(self.mats) == 1:
            D, ic = self.mats[0], self.inv_counts[0]
            X -= D @ ((D.T @ X) * ic[:, None])
            return X[:, 0] if one_d else X
        scale = np.maximum(np.abs(X).max(axis=0), 1e-12)
        for _ in range(self.max_iter):
            X_old = X.copy()
            for D, ic in zip(self.mats, self.inv_counts):
                X -= D @ ((D.T @ X) * ic[:, None])
            if np.max(np.abs(X - X_old).max(axis=0) / scale) < self.tol:
                break
        return X[:, 0] if one_d else X


class ClusterOLS:
    """OLS on demeaned data with one-way cluster-robust variance.

    Small-sample factor G/(G-1) * (N-1)/(N-K) with K = number of regressors
    (absorbed fixed effects excluded), matching pyfixest/reghdfe defaults for
    fixed effects nested in clusters.
    """

    def __init__(self, cluster_codes):
        c = np.asarray(cluster_codes)
        _, idx = np.unique(c, return_inverse=True)
        self.G = idx.max() + 1
        self.C = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), self.G))

    def fit(self, y: np.ndarray, X: np.ndarray, k_extra: int = 0):
        """k_extra: regressors already partialled out of y and X (Frisch-Waugh-Lovell);
        they enter only the small-sample degrees-of-freedom term."""
        n, k = X.shape
        XtX = X.T @ X
        try:
            L = np.linalg.cholesky(XtX)
        except np.linalg.LinAlgError:
            return None
        beta = np.linalg.solve(XtX, X.T @ y)
        e = y - X @ beta
        bread = np.linalg.inv(XtX)
        S = self.C.T @ (X * e[:, None])
        meat = S.T @ S
        adj = self.G / (self.G - 1) * (n - 1) / (n - k - k_extra)
        V = adj * bread @ meat @ bread
        se = np.sqrt(np.maximum(np.diag(V), 0))
        return beta, se, e


def within_r2(y_dm, e):
    """Within R-squared given demeaned outcome and residuals."""
    return 1.0 - float(e @ e) / float(y_dm @ y_dm)
