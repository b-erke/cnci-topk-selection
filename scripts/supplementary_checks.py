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
5. holdout_all_families.csv       split-sample prediction of held-out top-K overlap for every candidate family (fitted on
                                 one half by maximum pseudo-likelihood; same splits as holdout_splits.csv)
6. country_global_test.csv        multiplicity-aware check of the country cutoffs: the smallest two-sided tail proportion
                                 across K = 5-50 compared with its distribution under each reference
Everything is deterministic given BASE_SEED."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scipy import stats
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
    ap.add_argument("--holdout-reps", type=int, default=1000)
    ap.add_argument("--only", default="", help="comma-separated subset of sections 1-6")
    a = ap.parse_args(); out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    raw = D.load_raw(a.data_dir); base = {lev: D.eligible(raw[lev]) for lev in LEVELS}
    run = lambda *k: not a.only or any(x in a.only.split(",") for x in k)

    # 1. thresholds
    if run('1'):
        rows = []
        for lev in LEVELS:
            for t in C.DOC_THRESHOLDS:
                d = D.eligible(raw[lev], t)
                for K in KS[lev]:
                    if K < len(d):
                        s = M.shared(d, K); rows.append(dict(level=lev, min_docs=t, N=len(d), K=K, shared=s, jaccard=s / (2 * K - s)))
        pd.DataFrame(rows).to_csv(out / "threshold_topk.csv", index=False); print("wrote threshold_topk.csv")

    # 2-3. parameter uncertainty and family-aware resampling
    if run('2', '3'):
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
    if run('4'):
        ref = pd.read_csv(out / "copula_references.csv")
        for c in ("p_sim_ge_obs", "p_sim_le_obs"):
            ref[c + "_mcse"] = np.sqrt(ref[c] * (1 - ref[c]) / ref.reps)
        ref.to_csv(out / "copula_references_mcse.csv", index=False); print("wrote copula_references_mcse.csv")

    # 5. split-sample prediction for all families (same splits as the main holdout step)
    if run('5'):
        fams = Q.ONE_PARAM + ["sbb1", "tfree"]; org0 = base["Organization"]; rng = np.random.default_rng(C.BASE_SEED + 2)
        Ks = C.HOLDOUT_K; rows = []
        for sp in range(C.HOLDOUT_SPLITS):
            perm = rng.permutation(len(org0)); halves = (perm[: len(org0) // 2], perm[len(org0) // 2:])
            for direction, (fit_idx, test_idx) in enumerate([halves, halves[::-1]]):
                dfit = D.rank(org0.iloc[fit_idx]); dtest = D.rank(org0.iloc[test_idx]); u = Q.pseudo_obs(dfit)
                obs = {K: M.shared(dtest, K) for K in Ks}
                for f in fams:
                    par, _ = Q.fit(f, u)
                    sim = Q.simulate_overlap(f, par, len(dtest), Ks, a.holdout_reps, ("holdout_all", sp, direction, f))
                    for j, K in enumerate(Ks):
                        x = sim[:, j]; o = obs[K]; lo, hi = np.percentile(x, [5, 95])
                        rows.append(dict(split=sp, direction=direction, family=f, K=K, observed_test=o, mean=x.mean(),
                                         error=o - x.mean(), within_5_95=bool(lo <= o <= hi),
                                         log_score=np.log(((x == o).sum() + 0.5) / (len(x) + 1))))
        ha = pd.DataFrame(rows); ha.to_csv(out / "holdout_all_families.csv", index=False)
        summ = ha.groupby(["family", "K"]).agg(coverage_5_95=("within_5_95", "mean"), mean_error=("error", "mean"),
                                               mae=("error", lambda e: e.abs().mean()), mean_log_score=("log_score", "mean")).reset_index()
        summ.to_csv(out / "holdout_all_families_summary.csv", index=False)
        tot = ha.groupby("family").agg(mae=("error", lambda e: e.abs().mean()), mean_log_score=("log_score", "mean"),
                                       coverage_5_95=("within_5_95", "mean")).sort_values("mean_log_score", ascending=False)
        tot.to_csv(out / "holdout_all_families_overall.csv"); print(tot.round(3))

    # 6. country cutoffs: multiplicity-aware global check (same replicates as copula_references.csv)
    if run('6'):
        fits = pd.read_csv(out / "copula_fits.csv"); lev = "Country/region"; d = base[lev]; Kc = KS[lev]; rows = []
        obs = np.array([M.shared(d, K) for K in Kc])
        for f in ("gauss", "sbb1"):
            r = fits[(fits.level == lev) & (fits.family == f)].iloc[0]
            par = r.calibrated_param1 if f == "gauss" else (r.calibrated_param1, r.calibrated_param2)
            sim = Q.simulate_overlap(f, par, len(d), Kc, C.REPS, (lev, "V0", f + "_cal"))
            def two_sided(vals):
                out_ = np.empty(vals.shape)
                for j in range(len(Kc)):
                    srt = np.sort(sim[:, j]); n = len(srt)
                    ge = (n - np.searchsorted(srt, vals[..., j], side="left")) / n
                    le = np.searchsorted(srt, vals[..., j], side="right") / n
                    out_[..., j] = np.minimum(1, 2 * np.minimum(ge, le))
                return out_
            p_obs = two_sided(obs[None, :])[0]; pmin_null = two_sided(sim).min(axis=1)
            rows.append(dict(level=lev, family=f + "_cal", cutoffs=";".join(map(str, Kc)), min_two_sided=p_obs.min(),
                             argmin_K=Kc[int(p_obs.argmin())], global_p=(pmin_null <= p_obs.min()).mean(), reps=C.REPS))
        pd.DataFrame(rows).to_csv(out / "country_global_test.csv", index=False); print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
