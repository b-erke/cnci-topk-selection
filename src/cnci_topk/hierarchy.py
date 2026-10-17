"""Organizational hierarchy map (read from data/hierarchy_edges.csv) and unit-definition variants."""
import re, collections
from pathlib import Path
import pandas as pd, networkx as nx

def load_edges(path, names):
    """Expand the curated edge file against the eligible names. Rows whose child starts with 're:' are regex rules."""
    e = pd.read_csv(path)
    S = set(names); rows = []
    for _, r in e.iterrows():
        if r.parent not in S:
            continue
        if str(r.child).startswith("re:"):
            rx = re.compile(r.child[3:])
            rows += [dict(r, child=n, rule=r.child) for n in names if rx.search(n) and n != r.parent]
        elif r.child in S:
            rows.append(dict(r, rule="exact name"))
    out = pd.DataFrame(rows).drop_duplicates(["child", "parent"])
    g = nx.DiGraph(); g.add_edges_from(zip(out.child, out.parent))
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("hierarchy edges contain a cycle")
    return out

def variants(base, edges):
    children = set(edges.child); parents = set(edges.parent)
    dup = base[base.duplicated(["cnci", "ccnci", "docs"], keep=False)]
    dupsys = set(n for n in dup["name"] if "System" in n)
    return {"V0 Baseline": set(), "V1 Exact duplicates excluded": dupsys,
            "V2 Child records excluded": children, "V3 Parent records excluded": parents,
            "V4 Mapped units excluded": children | parents}

def families(edges):
    g = nx.Graph(); g.add_edges_from(zip(edges.child, edges.parent))
    return {n: min(c) for c in nx.connected_components(g) for n in c}

def redundancy(d, edges, K, col):
    fam = families(edges); top = set(d.loc[d[col] <= K, "name"])
    cnt = collections.Counter(fam[n] for n in top if n in fam)
    involved = set(edges.child) | set(edges.parent)
    return sum(n in involved for n in top), sum(v - 1 for v in cnt.values() if v > 1)

def edge_checks(edges, base):
    docs = dict(zip(base["name"], base["docs"]))
    e = edges.copy()
    e["docs_child"] = e.child.map(docs); e["docs_parent"] = e.parent.map(docs)
    e["child_docs_le_parent"] = e.docs_child <= e.docs_parent
    return e
