#!/usr/bin/env python3
"""Write synthetic InCites-format exports (fake values) so the pipeline can be tested without licensed data.
    python scripts/make_synthetic_data.py --out data/synthetic"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def frame(names, rng):
    n = len(names)
    z = rng.multivariate_normal([0, 0], [[1, .95], [.95, 1]], n)
    docs = np.round(np.exp(rng.normal(5.5, 1.3, n))).astype(int) + 1
    comp = rng.dirichlet([3, 4, 3, 1, .6], n) * 100
    return pd.DataFrame({
        "Name": names, "Web of Science Documents": docs, "Times Cited": docs * 20,
        "Collab-CNCI": np.exp(0.4 * z[:, 1]), "Rank": np.arange(1, n + 1), "% Docs Cited": 90.0,
        "Category Normalized Citation Impact": np.exp(0.4 * z[:, 0]), "% Documents in Top 10%": 10.0,
        "% International Documents": comp[:, 2:].sum(1), "% Domestic Single Documents": comp[:, 0],
        "% Domestic Multiple Documents": comp[:, 1], "% International Bilateral Documents": comp[:, 2],
        "% International Trilateral Documents": comp[:, 3], "% International Quadrilateral + Documents": comp[:, 4]})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("data/synthetic")); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    loc = [f"COUNTRY {i:03d}" for i in range(150)] + ["ENGLAND", "SCOTLAND", "UNITED KINGDOM"]
    org = [f"Organization {i:04d}" for i in range(900)] + ["Alphabet Inc.", "Google Incorporated", "DeepMind", "Chinese Academy of Sciences",
           "Institute of Automation, CAS", "University of California System", "University of California Berkeley"]
    frame(loc, rng).to_excel(a.out / "Incites_Locations.xlsx", index=False)
    frame(org, rng).to_excel(a.out / "Incites_Organizations.xlsx", index=False)
    print("synthetic exports written to", a.out)

if __name__ == "__main__":
    main()
