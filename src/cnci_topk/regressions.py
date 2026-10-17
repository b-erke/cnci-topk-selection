"""Descriptive OLS models with HC3 standard errors, additive log-ratio sensitivity, and paired bootstrap."""
import numpy as np, pandas as pd, statsmodels.api as sm
from . import config as C

SHARES = ["dmult", "bi", "tri", "quad"]          # domestic single-institution share is the reference

def composition_model(d, outcome="diff", alr_offset=None):
    y = d["cnci"] - d["ccnci"] if outcome == "diff" else d[outcome]
    if alr_offset is None:
        Z = d[SHARES].to_numpy(float)
    else:
        comp = d[["dsing"] + SHARES].to_numpy(float) + alr_offset
        Z = np.log(comp[:, 1:] / comp[:, [0]])
    X = sm.add_constant(np.column_stack([Z, np.log(d["docs"].to_numpy(float))]))
    return sm.OLS(y.to_numpy(float), X).fit(cov_type="HC3")

def intl_model(d, outcome):
    X = sm.add_constant(np.column_stack([d["intl"].to_numpy(float) / 10, np.log(d["docs"].to_numpy(float))]))
    return sm.OLS(d[outcome].to_numpy(float), X).fit(cov_type="HC3")

def tidy(m, names, **meta):
    ci = m.conf_int()
    return [dict(meta, term=n, coef=m.params[i], se_hc3=m.bse[i], ci_low=ci[i, 0], ci_high=ci[i, 1], r2=m.rsquared, n=int(m.nobs))
            for i, n in enumerate(names)]

def coef_diff_bootstrap(d, B=C.BOOT_COEF, seed=C.BASE_SEED):
    """Paired entity-level bootstrap of b_CNCI - b_CollabCNCI for international share (per 10 pp), adjusted for
    log documents. Units (countries or organizations) are resampled with replacement; both models are refitted
    on the same resample. Returns point estimate, percentile 95% interval and bootstrap SE."""
    X = np.column_stack([np.ones(len(d)), d["intl"].to_numpy(float) / 10, np.log(d["docs"].to_numpy(float))])
    y1 = d["cnci"].to_numpy(float); y2 = d["ccnci"].to_numpy(float)
    est = lambda idx: (np.linalg.lstsq(X[idx], y1[idx], rcond=None)[0][1] - np.linalg.lstsq(X[idx], y2[idx], rcond=None)[0][1])
    point = est(np.arange(len(d)))
    rng = np.random.default_rng(seed); N = len(d)
    bs = np.array([est(rng.integers(0, N, N)) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(point=point, ci_low=lo, ci_high=hi, boot_se=bs.std(ddof=1), B=B, seed=seed, method="percentile", unit="entity")
