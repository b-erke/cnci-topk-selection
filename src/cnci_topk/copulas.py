"""Copula families, pseudo-likelihood fits, Spearman calibration and top-K overlap simulation.

Simulation and fitting use pyvinecopulib. An independent NumPy implementation of the one-parameter
log-densities is kept for cross-checking pseudo-log-likelihoods (column loglik_numpy in fits)."""
import zlib
import numpy as np, pyvinecopulib as pv
from scipy import stats, optimize
from scipy.special import gammaln
from . import config as C

F = pv.BicopFamily
ONE_PARAM = ["gauss", "t10", "t5", "t3", "gumbel", "sclayton", "frank", "clayton"]
LABEL = {"gauss": "Gaussian", "t10": "t, nu = 10", "t5": "t, nu = 5", "t3": "t, nu = 3", "gumbel": "Gumbel",
         "sclayton": "Survival Clayton", "frank": "Frank", "clayton": "Clayton", "sbb1": "Survival BB1",
         "tfree": "t, nu estimated"}
BOUNDS = {"gauss": (0.3, 0.999), "t": (0.3, 0.999), "gumbel": (1.01, 49.9), "sclayton": (0.05, 27.9),
          "clayton": (0.05, 27.9), "frank": (0.5, 34.9)}

def make(fam, par):
    """Return a pyvinecopulib Bicop. par: float for one-parameter families, (theta, delta) for sbb1, (rho, nu) for tfree."""
    if fam == "gauss":
        return pv.Bicop(family=F.gaussian, parameters=np.array([[par]]))
    if fam in ("t10", "t5", "t3"):
        return pv.Bicop(family=F.student, parameters=np.array([[par], [float(fam[1:])]]))
    if fam == "tfree":
        return pv.Bicop(family=F.student, parameters=np.array([[par[0]], [par[1]]]))
    if fam == "gumbel":
        return pv.Bicop(family=F.gumbel, parameters=np.array([[par]]))
    if fam == "clayton":
        return pv.Bicop(family=F.clayton, parameters=np.array([[par]]))
    if fam == "sclayton":
        return pv.Bicop(family=F.clayton, rotation=180, parameters=np.array([[par]]))
    if fam == "frank":
        return pv.Bicop(family=F.frank, parameters=np.array([[par]]))
    if fam == "sbb1":
        return pv.Bicop(family=F.bb1, rotation=180, parameters=np.array([[par[0]], [par[1]]]))
    raise ValueError(fam)

def n_params(fam):
    return 2 if fam in ("sbb1", "tfree") else 1   # t10/t5/t3 use prespecified nu

def pseudo_obs(d):
    N = len(d)
    return np.column_stack([stats.rankdata(d["cnci"]) / (N + 1), stats.rankdata(d["ccnci"]) / (N + 1)])

def _bounds(fam):
    return BOUNDS["t"] if fam.startswith("t") else BOUNDS[fam]

def fit(fam, u):
    """Maximum pseudo-likelihood (Genest et al., 1995). Returns (params, loglik)."""
    if fam in ("t10", "t5", "t3"):
        r = optimize.minimize_scalar(lambda x: -make(fam, x).loglik(u), bounds=_bounds(fam), method="bounded",
                                     options={"xatol": 1e-7})
        return r.x, -r.fun
    fams = {"gauss": F.gaussian, "gumbel": F.gumbel, "clayton": F.clayton, "sclayton": F.clayton,
            "frank": F.frank, "sbb1": F.bb1, "tfree": F.student}
    rot = 180 if fam in ("sclayton", "sbb1") else 0
    b = pv.Bicop(family=fams[fam], rotation=rot)
    b.fit(u, controls=pv.FitControlsBicop(family_set=[fams[fam]], parametric_method="mle"))
    p = b.parameters.ravel()
    par = float(p[0]) if fam not in ("sbb1", "tfree") else (float(p[0]), float(p[1]))
    return par, b.loglik(u)

def tail_dependence(fam, par):
    """(lambda_L, lambda_U)."""
    if fam in ("gauss", "frank"):
        return 0.0, 0.0
    if fam in ("t10", "t5", "t3", "tfree"):
        r, nu = (par, float(fam[1:])) if fam != "tfree" else par
        lam = 2 * stats.t(nu + 1).cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)))
        return lam, lam
    if fam == "gumbel":
        return 0.0, 2 - 2 ** (1 / par)
    if fam == "clayton":
        return 2 ** (-1 / par), 0.0
    if fam == "sclayton":
        return 0.0, 2 ** (-1 / par)
    if fam == "sbb1":   # rotation by 180 degrees swaps the BB1 tails
        th, de = par
        return 2 - 2 ** (1 / de), 2 ** (-1 / (th * de))
    raise ValueError(fam)

def spearman_of(b, n=None, seed=7):
    n = C.CAL_N if n is None else n
    s = b.sample(n, seeds=[C.BASE_SEED, seed])
    return stats.spearmanr(s[:, 0], s[:, 1])[0]

def calibrate(fam, rho, fixed=None):
    """Parameter giving Spearman rho. Gaussian: closed form; others: root-finding on a fixed-seed Monte Carlo
    Spearman (300,000 draws). For sbb1 the first parameter is held at `fixed` and delta is calibrated."""
    if fam == "gauss":
        return 2 * np.sin(np.pi * rho / 6)
    if fam == "sbb1":
        de = optimize.brentq(lambda x: spearman_of(make("sbb1", (fixed, x))) - rho, 1.0001, 6.999, xtol=1e-4)
        return (fixed, de)
    lo, hi = _bounds(fam)
    return optimize.brentq(lambda x: spearman_of(make(fam, x)) - rho, lo, hi, xtol=1e-5)

def seed_for(*key):
    return zlib.crc32("|".join(map(str, key)).encode()) % 2**31

def simulate_overlap(fam, par, N, Ks, reps, key):
    """Shared top-K membership for `reps` replicates of size N. Deterministic given key."""
    b = make(fam, par); k0 = seed_for(*key); Kmax = max(Ks)
    out = np.zeros((reps, len(Ks)), dtype=int)
    for i in range(reps):
        s = b.sample(N, seeds=[C.BASE_SEED, k0, i])
        tx = np.argpartition(-s[:, 0], Kmax - 1)[:Kmax]; tx = tx[np.argsort(-s[tx, 0])]
        ty = np.argpartition(-s[:, 1], Kmax - 1)[:Kmax]; ty = ty[np.argsort(-s[ty, 1])]
        for j, K in enumerate(Ks):
            out[i, j] = np.intersect1d(tx[:K], ty[:K], assume_unique=True).size
    return out

# ---- independent NumPy log-densities (cross-check only)
def logc_numpy(fam, par, u, v):
    nd = stats.norm
    if fam == "gauss":
        x, y = nd.ppf(u), nd.ppf(v); r = par
        return -0.5 * np.log(1 - r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
    if fam in ("t10", "t5", "t3"):
        nu = float(fam[1:]); r = par; td = stats.t(nu); x, y = td.ppf(u), td.ppf(v)
        q = (x * x - 2 * r * x * y + y * y) / (1 - r * r)
        l2 = gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * np.log(1 - r * r) - (nu + 2) / 2 * np.log1p(q / nu)
        return l2 - td.logpdf(x) - td.logpdf(y)
    if fam in ("clayton", "sclayton"):
        if fam == "sclayton": u, v = 1 - u, 1 - v
        th = par
        return np.log1p(th) + (-th - 1) * (np.log(u) + np.log(v)) + (-2 - 1 / th) * np.log(u ** -th + v ** -th - 1)
    if fam == "gumbel":
        th = par; x, y = -np.log(u), -np.log(v); s = x ** th + y ** th; A = s ** (1 / th)
        return -A - np.log(u) - np.log(v) + (th - 1) * np.log(x * y) + (-2 + 1 / th) * np.log(s) + np.log(A + th - 1)
    if fam == "frank":
        th = par
        return (np.log(th) + np.log(-np.expm1(-th)) + th * (u + v)
                - 2 * np.log(np.exp(th * u) + np.exp(th * v) - 1 - np.exp(th * (u + v - 1))))
    raise ValueError(fam)
