#!/usr/bin/env python3
"""Supplementary robustness checks (aggregate outputs only).

    python scripts/supplementary_checks.py --data-dir data/raw --edges path/to/hierarchy_edges.csv --out-dir results

1. threshold_topk.csv            observed top-K overlap and Jaccard under the publication-volume thresholds
2. reference_param_uncertainty.csv  Gaussian and survival BB1 overlap references that propagate parameter-estimation
                                 uncertainty: units are resampled (i.i.d., and for organizations also by hierarchy
                                 family), each family is refitted by maximum pseudo-likelihood, and overlap is simulated
                                 at the refitted parameters; the pooled draws form the reference
3. modelsel_family_bootstrap.csv  copula selection by pseudo-AIC when organizations are resampled by hierarchy family
4. copula_references_mcse.csv     Monte Carlo standard errors of the reported tail proportions
Everything is deterministic given BASE_SEED."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cnci_topk import config as C, data as D, metrics as M, copulas as Q, hierarchy as H

LEVELS = ["Country/region", "Organization"]
KS = {"Country/region": C.K_COUNTRY, "Organization": C.K_ORG}
CAND = ["gauss", "tfree", "gumbel", "sclayton", "frank", "clayton", "sbb1"]

def pobs(d):
    return Q.pseudo_obs(d.reset_index(drop=True))

def resample(d, rng, clusters=None):
    if clusters is None:
        return d.iloc[rng.integers(0, len(d), len(d))]
    keys = clusters.unique(); pick = rng.choice(keys, size=len(keys), replace=True)
    idx = np.concatenate([np.flatnonzero(clusters.to_numpy() == k) for k in pick])
    return d.iloc[idx]

def param_uncertainty(d, lev, scheme, B, R, clusters, rng, rows, sel_rows):
    N = len(d); Ks = [K for K in KS[lev] if K < N]
    obs = {K: M.shared(d, K) for K in Ks}
    pooled = {f: [] for f in ("gauss", "sbb1")}
    for b in range(B):
        db = resample(d, rng, clusters); u = pobs(db)
        if scheme == "family":
            aics = {f: 2 * Q.n_params(f) - 2 * Q.fit(f, u)[1] for f in CAND}
            sel_rows.append(dict(level=lev, boot=b, best=min(aics, key=aics.get), n_units=len(db)))
        for f in pooled:
            par, _ = Q.fit(f, u)
            pooled[f].append(Q.simulate_overlap(f, par, N, Ks, R, ("paramunc", lev, scheme, f, b)))
    for f, sims in pooled.items():
        s = np.vstack(sims)
        for j, K in enumerate(Ks):
            x = s[:, j]
            rows.append(dict(level=lev, resampling=scheme, family=f, K=K, observed=obs[K], mean=x.mean(),
                             p05=np.percentile(x, 5), p95=np.percentile(x, 95),
                             p_sim_ge_obs=(x >= obs[K]).mean(), p_sim_le_obs=(x <= obs[K]).mean(),
                             boot=B, reps_per_boot=R))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--edges", default=ROOT / "data" / "hierarchy_edges.csv", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results", type=Path)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args(); out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    raw = D.load_raw(a.data_dir); base = {lev: D.eligible(raw[lev]) for lev in LEVELS}

    # 1. thresholds
    rows = []
    for lev in LEVELS:
        for t in C.DOC_THRESHOLDS:
            d = D.eligible(raw[lev], t)
            for K in KS[lev]:
                if K < len(d):
                    s = M.shared(d, K); rows.append(dict(level=lev, min_docs=t, N=len(d), K=K, shared=s, jaccard=s / (2 * K - s)))
    pd.DataFrame(rows).to_csv(out / "threshold_topk.csv", index=False); print("wrote threshold_topk.csv")

    # 2-3. parameter uncertainty and family-aware resampling
    rng = np.random.default_rng(C.BASE_SEED + 10); rows, sel = [], []
    org = base["Organization"].reset_index(drop=True)
    edges = H.load_edges(a.edges, list(org.name)); fam = H.families(edges)
    clusters = org.name.map(lambda n: fam.get(n, n))
    for lev in LEVELS:
        d = base[lev].reset_index(drop=True)
        param_uncertainty(d, lev, "units", a.boot, a.reps, None, rng, rows, sel)
    param_uncertainty(org, "Organization", "family", a.boot, a.reps, clusters, rng, rows, sel)
    pd.DataFrame(rows).to_csv(out / "reference_param_uncertainty.csv", index=False); print("wrote reference_param_uncertainty.csv")
    s = pd.DataFrame(sel); s.to_csv(out / "modelsel_family_bootstrap.csv", index=False)
    print(s.best.value_counts(normalize=True)); print("families:", clusters.nunique(), "multi-unit families:", (clusters.value_counts() > 1).sum())

    # 4. Monte Carlo standard errors
    ref = pd.read_csv(out / "copula_references.csv")
    for c in ("p_sim_ge_obs", "p_sim_le_obs"):
        ref[c + "_mcse"] = np.sqrt(ref[c] * (1 - ref[c]) / ref.reps)
    ref.to_csv(out / "copula_references_mcse.csv", index=False); print("wrote copula_references_mcse.csv")

if __name__ == "__main__":
    main()
