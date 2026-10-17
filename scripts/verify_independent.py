#!/usr/bin/env python3
"""Independent check of the copula results using only NumPy/SciPy (no pyvinecopulib, no package code).

  python scripts/verify_independent.py --results results                 # needs no licensed data
  python scripts/verify_independent.py --results results --data-dir data/raw   # also recomputes pseudo-log-likelihoods

Checks (written to <results>/independent_verification.csv):
 1. calibration: Spearman correlation of 500,000 independent draws at each calibrated parameter vs the observed rho;
 2. references: mean shared top-K selections from independent samplers vs copula_references.csv (z = difference / MC SE);
 3. (with --data-dir) pseudo-log-likelihood at the fitted parameters from closed-form NumPy densities vs copula_fits.csv."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import gammaln

# ---------------- samplers (return two columns whose ranks follow the copula)
def stable(alpha, n, rng):                       # positive stable, Laplace transform exp(-t**alpha) (Kanter / CMS)
    W = rng.uniform(0, np.pi, n); E = rng.exponential(size=n)
    return (np.sin(alpha * W) / np.sin(W) ** (1 / alpha)) * (np.sin((1 - alpha) * W) / E) ** ((1 - alpha) / alpha)

def sample(fam, p1, p2, n, rng):
    if fam == "gauss":
        x = rng.standard_normal(n); return x, p1 * x + np.sqrt(1 - p1 ** 2) * rng.standard_normal(n)
    if fam in ("t10", "t5", "t3"):
        nu = float(fam[1:]); x = rng.standard_normal(n); y = p1 * x + np.sqrt(1 - p1 ** 2) * rng.standard_normal(n)
        w = np.sqrt(nu / rng.chisquare(nu, n)); return x * w, y * w
    if fam in ("clayton", "sclayton"):
        V = rng.gamma(1 / p1, 1, n); u = (1 + rng.exponential(size=n) / V) ** (-1 / p1); v = (1 + rng.exponential(size=n) / V) ** (-1 / p1)
        return (u, v) if fam == "clayton" else (1 - u, 1 - v)
    if fam == "gumbel":
        S = stable(1 / p1, n, rng); return np.exp(-(rng.exponential(size=n) / S) ** (1 / p1)), np.exp(-(rng.exponential(size=n) / S) ** (1 / p1))
    if fam == "frank":
        u = rng.uniform(size=n); w = rng.uniform(size=n); e = np.exp(-p1 * u)
        return u, -(np.log((1 - w) * e + w * np.exp(-p1)) - np.log(w + (1 - w) * e)) / p1
    if fam == "sbb1":                             # BB1: psi(s) = (1 + s**(1/delta))**(-1/theta); frailty G**delta * S
        th, de = p1, p2; V = rng.gamma(1 / th, 1, n) ** de * stable(1 / de, n, rng)
        psi = lambda s: (1 + s ** (1 / de)) ** (-1 / th)
        u = psi(rng.exponential(size=n) / V); v = psi(rng.exponential(size=n) / V)
        return 1 - u, 1 - v                       # survival (180-degree rotation)
    raise ValueError(fam)

# ---------------- log-densities
def logc(fam, p1, p2, u, v):
    nd = stats.norm
    if fam == "gauss":
        x, y, r = nd.ppf(u), nd.ppf(v), p1
        return -0.5 * np.log(1 - r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
    if fam in ("t10", "t5", "t3", "tfree"):
        nu = p2 if fam == "tfree" else float(fam[1:]); r = p1; td = stats.t(nu); x, y = td.ppf(u), td.ppf(v)
        q = (x * x - 2 * r * x * y + y * y) / (1 - r * r)
        return (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * np.log(1 - r * r) - (nu + 2) / 2 * np.log1p(q / nu)
                - td.logpdf(x) - td.logpdf(y))
    if fam in ("clayton", "sclayton"):
        if fam == "sclayton": u, v = 1 - u, 1 - v
        th = p1; return np.log1p(th) + (-th - 1) * (np.log(u) + np.log(v)) + (-2 - 1 / th) * np.log(u ** -th + v ** -th - 1)
    if fam == "gumbel":
        th = p1; x, y = -np.log(u), -np.log(v); s = x ** th + y ** th; A = s ** (1 / th)
        return -A - np.log(u) - np.log(v) + (th - 1) * np.log(x * y) + (-2 + 1 / th) * np.log(s) + np.log(A + th - 1)
    if fam == "frank":
        th = p1
        return np.log(th) + np.log(-np.expm1(-th)) + th * (u + v) - 2 * np.log(np.exp(th * u) + np.exp(th * v) - 1 - np.exp(th * (u + v - 1)))
    if fam == "sbb1":                             # Archimedean density psi''(phi(u)+phi(v)) phi'(u) phi'(v), at (1-u, 1-v)
        th, de = p1, p2; u, v = 1 - u, 1 - v; a, b = 1 / de, 1 / th
        phi = lambda t: (t ** -th - 1) ** de
        dphi = lambda t: de * (t ** -th - 1) ** (de - 1) * (-th) * t ** (-th - 1)
        s = phi(u) + phi(v); g = 1 + s ** a
        d2 = b * (b + 1) * g ** (-b - 2) * a * a * s ** (2 * a - 2) - b * g ** (-b - 1) * a * (a - 1) * s ** (a - 2)
        return np.log(d2) + np.log(dphi(u) * dphi(v))
    raise ValueError(fam)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=Path("results")); ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args(); R = a.results; rng = np.random.default_rng(a.seed)
    fits = pd.read_csv(R / "copula_fits.csv"); ref = pd.read_csv(R / "copula_references.csv"); agr = pd.read_csv(R / "agreement.csv").set_index("level")
    rows = []
    for _, f in fits[fits.family != "tfree"].iterrows():
        lev, fam = f.level, f.family
        p1, p2 = f.calibrated_param1, f.calibrated_param2
        x, y = sample(fam, p1, p2, 500_000, rng)
        rows.append(dict(check="calibration", level=lev, family=fam, package=agr.loc[lev, "spearman"], independent=stats.spearmanr(x, y)[0]))
    for _, f in fits[fits.family.isin(["gauss", "t10", "sbb1", "gumbel", "sclayton"])].iterrows():
        lev, fam = f.level, f.family; r = ref[(ref.level == lev) & (ref.family == fam + "_cal")]
        N = int(r.N.iloc[0]); Ks = r.K.to_list(); Kmax = max(Ks); sims = np.zeros((a.reps, len(Ks)))
        for i in range(a.reps):
            x, y = sample(fam, f.calibrated_param1, f.calibrated_param2, N, rng)
            ox = np.argsort(-x)[:Kmax]; oy = np.argsort(-y)[:Kmax]
            sims[i] = [np.intersect1d(ox[:K], oy[:K]).size for K in Ks]
        for j, K in enumerate(Ks):
            pk = r[r.K == K].iloc[0]; sd = sims[:, j].std(ddof=1)
            se = sd * np.sqrt(1 / a.reps + 1 / pk.reps) if sd > 0 else np.nan
            rows.append(dict(check="reference mean", level=lev, family=fam, K=K, package=pk["mean"], independent=sims[:, j].mean(),
                             z=(sims[:, j].mean() - pk["mean"]) / se if se else np.nan,
                             package_p05=pk.p05, independent_p05=np.percentile(sims[:, j], 5), package_p95=pk.p95, independent_p95=np.percentile(sims[:, j], 95)))
    if a.data_dir:
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from cnci_topk import data as D          # used only to apply the same eligibility rule to the raw files
        raw = D.load_raw(a.data_dir)
        for lev in ("Country/region", "Organization"):
            d = D.eligible(raw[lev]); N = len(d); u = stats.rankdata(d.cnci) / (N + 1); v = stats.rankdata(d.ccnci) / (N + 1)
            for _, f in fits[fits.level == lev].iterrows():
                ll = float(np.sum(logc(f.family, f.mpl_param1, f.mpl_param2, u, v)))
                rows.append(dict(check="pseudo-loglik at fitted parameters", level=lev, family=f.family, package=f.loglik, independent=ll))
    out = pd.DataFrame(rows); out["difference"] = out.independent - out.package
    out.to_csv(R / "independent_verification.csv", index=False)
    cal = out[out.check == "calibration"]; refm = out[out.check == "reference mean"]; ll = out[out.check.str.startswith("pseudo")]
    print(f"calibration: max |Spearman difference| = {cal.difference.abs().max():.4f}")
    print(f"references : max |z| = {refm.z.abs().max():.2f} over {len(refm)} cells; max |mean difference| = {refm.difference.abs().max():.3f}")
    if len(ll): print(f"pseudo-loglik: max |difference| = {ll.difference.abs().max():.2e}")

if __name__ == "__main__":
    main()
