"""Set-overlap, rank-agreement and boundary metrics."""
import numpy as np, pandas as pd
from scipy import stats
from . import config as C

def shared(d, K):
    return int(((d["r1"] <= K) & (d["r2"] <= K)).sum())

def jaccard(s, K):
    return s / (2 * K - s)

def rbo_complete(order_a, order_b, p):
    """Rank-biased overlap (Webber et al., 2010) for two complete rankings of the same N items.
    Returns (rbo_ext, rbo_trunc): rbo_ext is the extrapolated value at full depth, which for conjoint
    complete lists adds the residual weight p**N times agreement 1 at depth N; rbo_trunc omits it."""
    a, b = list(order_a), list(order_b)
    N = len(a)
    if sorted(a) != sorted(b):
        raise ValueError("RBO here requires two orderings of the same items")
    seen_a, seen_b, overlap, s = set(), set(), 0, 0.0
    for d in range(1, N + 1):
        x, y = a[d - 1], b[d - 1]
        if x == y:
            overlap += 1
        else:
            overlap += (x in seen_b) + (y in seen_a)
        seen_a.add(x); seen_b.add(y)
        s += p ** (d - 1) * overlap / d
    trunc = (1 - p) * s
    return trunc + p ** N, trunc

def agreement(d):
    rho = stats.spearmanr(d["cnci"], d["ccnci"])[0]
    tau = stats.kendalltau(d["cnci"], d["ccnci"])[0]
    out = dict(N=len(d), spearman=rho, kendall=tau, median_abs_rank_shift=float(np.median(np.abs(d["r1"] - d["r2"]))))
    oa = d.sort_values("r1").index.to_list(); ob = d.sort_values("r2").index.to_list()
    for p in C.RBO_P:
        ext, tr = rbo_complete(oa, ob, p)
        out[f"rbo_ext_p{p:.2f}"] = ext; out[f"rbo_trunc_p{p:.2f}"] = tr
    return out

def iqr(x, method=C.IQR_METHOD):
    q75, q25 = np.percentile(np.asarray(x, float), [75, 25], method=method)
    return q75 - q25

def switches(d, K):
    A = d["r1"] <= K; B = d["r2"] <= K
    return d[A & ~B], d[B & ~A]   # exits, entries

def boundary(d, K, method=C.IQR_METHOD):
    ex, en = switches(d, K)
    c1 = np.sort(d["cnci"].to_numpy())[::-1][K - 1]; c2 = np.sort(d["ccnci"].to_numpy())[::-1][K - 1]
    rank_gap = list(ex["r2"] - K) + list(en["r1"] - K)
    iqr_gap = list((c2 - ex["ccnci"]) / iqr(d["ccnci"], method)) + list((c1 - en["cnci"]) / iqr(d["cnci"], method))
    N = len(d)
    return dict(K=K, changed_units=len(rank_gap),
                median_rank_gap=float(np.median(rank_gap)) if rank_gap else np.nan,
                median_rank_gap_over_N=float(np.median(rank_gap)) / N if rank_gap else np.nan,
                median_iqr_gap=float(np.median(iqr_gap)) if iqr_gap else np.nan,
                max_iqr_gap=float(np.max(iqr_gap)) if iqr_gap else np.nan)

def profiles(d, K):
    ex, en = switches(d, K)
    rows = []
    for grp, g in (("Entry", en), ("Exit", ex)):
        rows.append(dict(K=K, group=grp, n=len(g), mean_intl=g["intl"].mean() if len(g) else np.nan,
                         mean_quad=g["quad"].mean() if len(g) else np.nan))
    return rows

def switch_list(d, K):
    ex, en = switches(d, K)
    fmt = lambda g: "; ".join(f"{n} ({a}->{b})" for n, a, b in zip(g["name"], g["r1"], g["r2"]))
    return dict(K=K, exits=fmt(ex), entries=fmt(en))
