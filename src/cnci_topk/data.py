"""Load InCites exports, apply the eligibility rules, and rank units."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from . import config as C

COLS = {
    "Name": "name", "Web of Science Documents": "docs", "Times Cited": "cites",
    "Category Normalized Citation Impact": "cnci", "Collab-CNCI": "ccnci",
    "% International Documents": "intl", "% Domestic Single Documents": "dsing",
    "% Domestic Multiple Documents": "dmult", "% International Bilateral Documents": "bi",
    "% International Trilateral Documents": "tri", "% International Quadrilateral + Documents": "quad",
}
REQUIRED = list(COLS)

def _find(data_dir: Path, key: str) -> Path:
    p = data_dir / C.FILES[key]
    if p.exists():
        return p
    hits = sorted(data_dir.glob(f"*{C.FILES[key].split('_')[1].split('.')[0]}*.xlsx"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Expected {C.FILES[key]} in {data_dir} (see data/README.md); found {[h.name for h in hits]}")

def read_export(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing InCites columns {missing}")
    df = df[REQUIRED].rename(columns=COLS)
    df = df[df["name"].notna()]
    df["name"] = df["name"].astype(str).str.strip()
    return df

def load_raw(data_dir):
    data_dir = Path(data_dir)
    loc = read_export(_find(data_dir, "locations"))
    org = read_export(_find(data_dir, "organizations"))
    loc = loc[~loc["name"].str.upper().isin(C.UK_CONSTITUENTS)]
    return {"Country/region": loc, "Organization": org}

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Descending ordinal ranks; exact ties broken by alphabetical order of name."""
    d = df.sort_values("name").reset_index(drop=True).copy()
    d["r1"] = rankdata(-d["cnci"].to_numpy(), method="ordinal").astype(int)
    d["r2"] = rankdata(-d["ccnci"].to_numpy(), method="ordinal").astype(int)
    return d

def eligible(df: pd.DataFrame, min_docs: int = C.MIN_DOCS) -> pd.DataFrame:
    return rank(df[df["docs"] >= min_docs])
