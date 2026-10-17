"""pytest tests/  — unit checks of the metrics plus an end-to-end smoke run on synthetic data."""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pytest
from cnci_topk import metrics as M
try:
    from cnci_topk import copulas as Q
except ImportError:          # pyvinecopulib missing: metric tests still run
    Q = None
needs_pv = pytest.mark.skipif(Q is None, reason="pyvinecopulib not installed (pip install -r requirements.txt)")

def test_jaccard_and_overlap():
    d = pd.DataFrame({"r1": [1, 2, 3, 4], "r2": [2, 1, 4, 3]})
    assert M.shared(d, 2) == 2 and M.shared(d, 1) == 0 and M.jaccard(2, 2) == 1.0 and M.jaccard(1, 2) == 1 / 3

def test_rbo():
    ext, tr = M.rbo_complete(list("abcd"), list("abcd"), 0.9); assert abs(ext - 1) < 1e-12
    ext, tr = M.rbo_complete(list("abcd"), list("dcba"), 0.9); assert 0 < tr < ext < 1

@needs_pv
def test_density_crosscheck():
    rng = np.random.default_rng(0); u = rng.uniform(0.01, 0.99, (500, 2))
    for fam, par in [("gauss", 0.8), ("t5", 0.8), ("gumbel", 2.5), ("frank", 8.0), ("clayton", 3.0), ("sclayton", 3.0)]:
        assert abs(Q.make(fam, par).loglik(u) - Q.logc_numpy(fam, par, u[:, 0], u[:, 1]).sum()) < 1e-6

@needs_pv
def test_calibration():
    for fam in ["gauss", "gumbel"]:
        p = Q.calibrate(fam, 0.9); assert abs(Q.spearman_of(Q.make(fam, p), seed=123) - 0.9) < 0.003

@needs_pv
@needs_pv
def test_gof_empirical_copula_and_null():
    from cnci_topk import gof as GF
    u = np.array([[0.2, 0.3], [0.5, 0.1], [0.8, 0.9]])
    assert np.allclose(GF.empirical_copula(u), [1 / 3, 1 / 3, 1.0])
    assert np.allclose(GF.empirical_survival(u, np.array([[0.1, 0.1]])), [2 / 3])   # strict inequality: (0.5, 0.1) is not above v = 0.1
    # data simulated from the tested family should not be systematically rejected
    x = Q.make("gauss", 0.8).sample(300, seeds=[1, 2]); n = len(x)
    from scipy.stats import rankdata
    ux = np.column_stack([rankdata(x[:, 0]) / (n + 1), rankdata(x[:, 1]) / (n + 1)])
    r = GF.gof_test("gauss", ux, 19, [0.8], ("test",))
    assert 0 < r[0]["p_S_n"] <= 1 and r[0]["n_upper"] > 0

@needs_pv
def test_smoke(tmp_path):
    syn = tmp_path / "syn"; out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(syn)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--data-dir", str(syn), "--out-dir", str(out), "--quick", "--workers", "1",
                    "--edges", str(ROOT / "data" / "synthetic_hierarchy_edges.csv")], check=True)
    for f in ["agreement.csv", "table1_fixed_k.csv", "copula_fits.csv", "copula_references.csv", "hierarchy_overlap.csv",
              "regressions.csv", "bootstrap_coef_diff.csv", "figure1.png", "figure2.png", "run_info.json", "restricted/switches.csv"]:
        assert (out / f).exists(), f
    # shareable outputs must not contain unit names, InCites values or unit ranks
    for f in out.glob("*.csv"):
        cols = set(pd.read_csv(f).columns)
        assert not cols & {"name", "child", "parent", "exits", "entries", "value", "docs_child", "docs_parent"}, f.name
    for f in ["table5_profiles.csv", "hierarchy_profiles.csv"]:
        d = pd.read_csv(out / f); assert d.loc[d.n < 3, ["mean_intl", "mean_quad"]].isna().all().all()
