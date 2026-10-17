"""Goodness-of-fit tests for the fitted copulas (Genest, Remillard & Beaudoin, 2009).

Two Cramer-von Mises statistics compare the empirical copula of the pseudo-observations with the copula fitted by
maximum pseudo-likelihood:

  S_n   = sum_i ( C_n(U_i) - C_theta(U_i) )^2                    (whole unit square; Genest et al., 2009, S_n)
  S_n^U = sum_{i: U_i1 > q, U_i2 > q} ( Cbar_n(U_i) - Cbar_theta(U_i) )^2
                                                                  (upper quadrant [q,1]^2, joint survival function)

Cbar(u, v) = P(U > u, V > v) = 1 - u - v + C(u, v) is the joint exceedance probability, which is the quantity that
drives shared top-K membership; S_n^U therefore targets the part of the dependence structure that the top-K
references depend on. P-values come from a parametric bootstrap in which each replicate is simulated from the
fitted copula, converted to pseudo-observations, refitted, and the statistic recomputed (Genest et al., 2009,
Appendix A). Everything is deterministic given the seeds.
"""
import numpy as np
from scipy import stats
from . import copulas as Q

def empirical_copula(u, at=None):
    """C_n evaluated at the rows of `at` (default: the sample itself)."""
    at = u if at is None else at
    return ((u[None, :, 0] <= at[:, None, 0]) & (u[None, :, 1] <= at[:, None, 1])).mean(axis=1)

def empirical_survival(u, at=None):
    at = u if at is None else at
    return ((u[None, :, 0] > at[:, None, 0]) & (u[None, :, 1] > at[:, None, 1])).mean(axis=1)

def statistics(fam, par, u, qs):
    """S_n and, for each q in qs, (S_n^U, number of points in the upper quadrant)."""
    b = Q.make(fam, par)
    c_model = b.cdf(u)
    s_n = float(np.sum((empirical_copula(u) - c_model) ** 2))
    up = {}
    for q in qs:
        m = (u[:, 0] > q) & (u[:, 1] > q)
        pts = u[m]
        sbar_model = 1 - pts[:, 0] - pts[:, 1] + c_model[m]
        up[q] = (float(np.sum((empirical_survival(u, pts) - sbar_model) ** 2)), int(m.sum()))
    return s_n, up

def _pobs(x):
    n = len(x)
    return np.column_stack([stats.rankdata(x[:, 0]) / (n + 1), stats.rankdata(x[:, 1]) / (n + 1)])

def gof_test(fam, u, B, qs, key):
    """One row per upper-quadrant threshold q: fitted parameter(s), statistics and parametric-bootstrap p-values.
    p-values are (1 + #{bootstrap statistic >= observed}) / (B + 1)."""
    par, _ = Q.fit(fam, u)
    s_n, up = statistics(fam, par, u, qs)
    k0 = Q.seed_for(*key); n = len(u)
    bn = np.empty(B); bu = {q: np.empty(B) for q in qs}
    for b in range(B):
        x = Q.make(fam, par).sample(n, seeds=[Q.C.BASE_SEED, k0, b])
        ub = _pobs(x)
        pb, _ = Q.fit(fam, ub)
        bn[b], upb = statistics(fam, pb, ub, qs)
        for q in qs:
            bu[q][b] = upb[q][0]
    p_n = (1 + np.sum(bn >= s_n)) / (B + 1)
    rows = []
    for q in qs:
        rows.append(dict(param1=par if np.isscalar(par) else par[0], param2=np.nan if np.isscalar(par) else par[1],
                         S_n=s_n, p_S_n=p_n, q_upper=q, n_upper=up[q][1], S_n_upper=up[q][0],
                         p_S_n_upper=(1 + np.sum(bu[q] >= up[q][0])) / (B + 1), B=B))
    return rows
