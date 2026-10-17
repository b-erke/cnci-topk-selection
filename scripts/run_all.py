#!/usr/bin/env python3
"""Run the full analysis. Example:
    python scripts/run_all.py --data-dir data/raw --out-dir results
    python scripts/run_all.py --data-dir data/synthetic --out-dir /tmp/smoke --quick
Steps: descriptives, topk, fits, references, modelsel, gof, holdout, hierarchy, regressions, figures (default: all)."""
import argparse, json, sys, time, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cnci_topk import config as C, data as D, metrics as M, copulas as Q, hierarchy as H, regressions as G, figures as FG, gof as GF
from scipy import stats

LEVELS = ["Country/region", "Organization"]
KS = {"Country/region": C.K_COUNTRY, "Organization": C.K_ORG}

def save(df, out, name):
    df = pd.DataFrame(df); df.to_csv(out / name, index=False); print(f"  wrote {name} ({len(df)} rows)"); return df

def _sim_task(args):
    fam, par, N, Ks, reps, key = args
    return key, Q.simulate_overlap(fam, par, N, Ks, reps, key)

def _gof_task(args):
    fam, u, B, qs, key, lev = args
    return lev, fam, GF.gof_test(fam, u, B, qs, key)

def run_sims(tasks, workers):
    if workers <= 1:
        return dict(_sim_task(t) for t in tasks)
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_sim_task, tasks))

def summarise(sim, Ks, obs, **meta):
    rows = []
    for j, K in enumerate(Ks):
        s = sim[:, j]
        rows.append(dict(meta, K=K, observed=obs[K], mean=s.mean(), p05=np.percentile(s, 5), p95=np.percentile(s, 95),
                         p_sim_ge_obs=(s >= obs[K]).mean(), p_sim_le_obs=(s <= obs[K]).mean(), reps=len(s)))
    return rows

MIN_CELL = 3
def suppress(rows):
    """Public profile tables: group means based on fewer than MIN_CELL units are withheld (they would reveal unit-level InCites values)."""
    df = pd.DataFrame(rows).copy(); small = df["n"] < MIN_CELL
    df.loc[small, ["mean_intl", "mean_quad"]] = np.nan; df["suppressed_n_lt_3"] = small
    return df

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results", type=Path)
    ap.add_argument("--edges", default=ROOT / "data" / "hierarchy_edges.csv", type=Path)
    ap.add_argument("--restricted-dir", type=Path, default=None,
                    help="unit-level outputs (names with InCites values or ranks); default <out-dir>/restricted; not for redistribution")
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--boot", type=int, default=C.BOOT_COEF)
    ap.add_argument("--boot-modelsel", type=int, default=C.BOOT_MODELSEL)
    ap.add_argument("--gof-boot", type=int, default=C.GOF_B)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="small replicate counts for a smoke test")
    ap.add_argument("--steps", default="all")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.boot, a.boot_modelsel, a.gof_boot = 200, 200, 10, 20
        C.HOLDOUT_SPLITS, C.HOLDOUT_REPS = 2, 100
        C.CAL_N = 30_000
    out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    rst = a.restricted_dir or out / "restricted"; rst.mkdir(parents=True, exist_ok=True)
    (rst / "README.txt").write_text("Unit-level outputs derived from licensed Clarivate InCites data. Do not redistribute.\n")
    steps = ["descriptives", "topk", "fits", "references", "modelsel", "gof", "holdout", "hierarchy", "regressions", "figures"] if a.steps == "all" else a.steps.split(",")
    t0 = time.time()
    raw = D.load_raw(a.data_dir)
    base = {lev: D.eligible(raw[lev]) for lev in LEVELS}
    run_info = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), reps=a.reps, boot=a.boot, cal_n=None,
                    boot_modelsel=a.boot_modelsel, gof_boot=a.gof_boot, base_seed=C.BASE_SEED, min_docs=C.MIN_DOCS, iqr_method=C.IQR_METHOD,
                    data_files=sorted(p.name for p in a.data_dir.glob("*.xlsx")), steps=steps)

    if "descriptives" in steps:
        print("[descriptives]")
        save([dict(level=lev, **M.agreement(base[lev])) for lev in LEVELS], out, "agreement.csv")
        rows = []
        for lev in LEVELS:
            for t in C.DOC_THRESHOLDS:
                d = D.eligible(raw[lev], t); rows.append(dict(level=lev, min_docs=t, N=len(d), spearman=stats.spearmanr(d.cnci, d.ccnci)[0]))
        save(rows, out, "threshold_sensitivity.csv")
        rows = []
        for lev in LEVELS:
            d = base[lev]
            for col in ("cnci", "ccnci"):
                t = d[d.duplicated(col, keep=False)].sort_values(col)
                rows += [dict(level=lev, indicator=col, name=n, value=v, rank=r) for n, v, r in zip(t.name, t[col], t["r1" if col == "cnci" else "r2"])]
        ties = save(pd.DataFrame(rows, columns=["level", "indicator", "name", "value", "rank"]), rst, "ties.csv")
        save([dict(level=lev, indicator=ind, tied_units=len(g), min_rank=g["rank"].min() if len(g) else np.nan, max_rank=g["rank"].max() if len(g) else np.nan)
              for lev in LEVELS for ind in ("cnci", "ccnci") for g in [ties[(ties.level == lev) & (ties.indicator == ind)]]], out, "ties_summary.csv")

    if "topk" in steps:
        print("[topk]")
        fixed, sweep, bnd, prof, sw = [], [], [], [], []
        for lev in LEVELS:
            d = base[lev]; N = len(d)
            for K in range(1, N):
                s = M.shared(d, K); row = dict(level=lev, N=N, K=K, shared=s, turnover=(K - s) / K, jaccard=M.jaccard(s, K))
                sweep.append(row)
                if K in KS[lev]:
                    fixed.append(row)
            for K in KS[lev]:
                bnd.append(dict(level=lev, **M.boundary(d, K)))
                bnd[-1]["median_iqr_gap_type7"] = M.boundary(d, K, "linear")["median_iqr_gap"]
                prof += [dict(level=lev, **r) for r in M.profiles(d, K)]
                sw.append(dict(level=lev, **M.switch_list(d, K)))
        save(fixed, out, "table1_fixed_k.csv"); save(sweep, out, "continuous_k.csv"); save(bnd, out, "table4_boundary.csv")
        save(prof, rst, "table5_profiles_full.csv"); save(suppress(prof), out, "table5_profiles.csv"); save(sw, rst, "switches.csv")
        rows = []
        for lev in LEVELS:
            d = base[lev]
            rows.append(dict(level=lev, spearman_shift_intl=stats.spearmanr(d.r2 - d.r1, d.intl)[0], spearman_shift_quad=stats.spearmanr(d.r2 - d.r1, d.quad)[0],
                             **{f"spearman_shift_{c}": stats.spearmanr(d.r2 - d.r1, d[c])[0] for c in ["dsing", "dmult", "bi", "tri"]}))
        save(rows, out, "shift_correlations.csv")

    fams_all = Q.ONE_PARAM + ["sbb1", "tfree"]
    if "fits" in steps or "references" in steps or "hierarchy" in steps:
        print("[fits]")
        fits, prof_t, params = [], [], {}
        for lev in LEVELS:
            d = base[lev]; u = Q.pseudo_obs(d); rho = stats.spearmanr(d.cnci, d.ccnci)[0]
            for fam in fams_all:
                par, ll = Q.fit(fam, u)
                row = dict(level=lev, family=fam, label=Q.LABEL[fam], n_params=Q.n_params(fam), loglik=ll, pseudo_aic=2 * Q.n_params(fam) - 2 * ll,
                           mpl_param1=par if np.isscalar(par) else par[0], mpl_param2=np.nan if np.isscalar(par) else par[1],
                           spearman_implied_by_mpl=Q.spearman_of(Q.make(fam, par)))
                if fam in Q.ONE_PARAM:
                    row["loglik_numpy"] = float(np.sum(Q.logc_numpy(fam, par, u[:, 0], u[:, 1])))
                    cal = Q.calibrate(fam, rho); row["calibrated_param1"] = cal; row["calibrated_param2"] = np.nan
                elif fam == "sbb1":
                    cal = Q.calibrate("sbb1", rho, fixed=par[0]); row["calibrated_param1"], row["calibrated_param2"] = cal
                else:
                    cal = None
                if cal is not None:
                    lam = Q.tail_dependence(fam, cal); row["lambda_L_cal"], row["lambda_U_cal"] = lam
                    row["spearman_check_cal"] = Q.spearman_of(Q.make(fam, cal), n=10 * C.CAL_N // 3, seed=99)
                params[(lev, fam)] = dict(mpl=par, cal=cal)
                fits.append(row)
            best = min(r["pseudo_aic"] for r in fits if r["level"] == lev)
            for r in fits:
                if r["level"] == lev: r["delta_pseudo_aic"] = r["pseudo_aic"] - best
            for nu in C.T_NU_GRID:
                r = Q.optimize.minimize_scalar(lambda x: -Q.pv.Bicop(family=Q.F.student, parameters=np.array([[x], [float(nu)]])).loglik(u),
                                               bounds=(0.3, 0.999), method="bounded", options={"xatol": 1e-7})
                prof_t.append(dict(level=lev, nu=nu, rho_hat=r.x, loglik=-r.fun))
            tf = next(f for f in fits if f["level"] == lev and f["family"] == "tfree")
            prof_t.append(dict(level=lev, nu="estimated", rho_hat=tf["mpl_param1"], nu_hat=tf["mpl_param2"], loglik=tf["loglik"]))
            # broader automatic selection (includes Joe, BB8 and nonparametric TLL)
            fc = Q.pv.FitControlsBicop(family_set=[Q.F.gaussian, Q.F.student, Q.F.gumbel, Q.F.clayton, Q.F.frank, Q.F.joe, Q.F.bb1, Q.F.bb6, Q.F.bb7, Q.F.bb8, Q.F.tll],
                                       selection_criterion="aic")
            sel = Q.pv.Bicop(); sel.select(u, controls=fc)
            run_info[f"broad_selection_{lev}"] = dict(family=str(sel.family), rotation=int(sel.rotation), aic=float(sel.aic(u)),
                                                      parameters=sel.parameters.ravel().round(4).tolist())
        if "fits" in steps:
            save(fits, out, "copula_fits.csv"); save(prof_t, out, "t_nu_profile.csv")

    if "references" in steps:
        print("[references]")
        tasks, meta = [], {}
        for lev in LEVELS:
            d = base[lev]; N = len(d); Ks = KS[lev]
            for fam in Q.ONE_PARAM + ["sbb1"]:
                key = (lev, "V0", fam + "_cal")
                tasks.append((fam, params[(lev, fam)]["cal"], N, Ks, a.reps, key)); meta[key] = (lev, fam + "_cal", "calibrated to observed Spearman")
            for fam in ["sbb1", "t10", "gauss"]:
                key = (lev, "V0", fam + "_mpl")
                tasks.append((fam, params[(lev, fam)]["mpl"], N, Ks, a.reps, key)); meta[key] = (lev, fam + "_mpl", "maximum pseudo-likelihood fit")
        sims = run_sims(tasks, a.workers); rows = []
        for key, sim in sims.items():
            lev, fam, how = meta[key]; d = base[lev]
            obs = {K: M.shared(d, K) for K in KS[lev]}
            rows += summarise(sim, KS[lev], obs, level=lev, N=len(d), family=fam, parameters=how)
        save(rows, out, "copula_references.csv")

    if "modelsel" in steps:
        print("[modelsel]")
        rows = []; rng = np.random.default_rng(C.BASE_SEED + 1)
        cand = ["gauss", "tfree", "gumbel", "sclayton", "frank", "clayton", "sbb1"]
        for lev in LEVELS:
            d = base[lev]; N = len(d)
            for b in range(a.boot_modelsel):
                idx = rng.integers(0, N, N); db = d.iloc[idx].reset_index(drop=True)
                u = Q.pseudo_obs(db); aics = {f: 2 * Q.n_params(f) - 2 * Q.fit(f, u)[1] for f in cand}
                best = min(aics, key=aics.get)
                rows.append(dict(level=lev, boot=b, best=best, **{f"aic_{f}": v for f, v in aics.items()}))
        ms = save(rows, out, "copula_modelsel_bootstrap.csv")
        save(ms.groupby("level").best.value_counts(normalize=True).rename("share").reset_index(), out, "copula_modelsel_summary.csv")

    if "gof" in steps:
        print("[gof]")
        # Parametric-bootstrap Cramer-von Mises goodness-of-fit tests (Genest et al., 2009) for every candidate family
        # fitted by maximum pseudo-likelihood; S_n covers the unit square, S_n^U the upper quadrant [q, 1]^2.
        tasks = [(fam, Q.pseudo_obs(base[lev]), a.gof_boot, C.GOF_Q, ("gof", lev, fam), lev)
                 for lev in LEVELS for fam in Q.ONE_PARAM + ["sbb1", "tfree"]]
        rows = []
        with ProcessPoolExecutor(max(1, a.workers)) as ex:
            for lev, fam, res in ex.map(_gof_task, tasks):
                rows += [dict(level=lev, family=fam, label=Q.LABEL[fam], **r) for r in res]
        save(rows, out, "copula_gof.csv")

    if "holdout" in steps:
        print("[holdout]")
        # Split-sample check: estimate the reference on a random half of organizations and compare it with the
        # observed top-K overlap in the other half, which played no part in the estimation.
        org = base["Organization"]; rng = np.random.default_rng(C.BASE_SEED + 2); Ks = C.HOLDOUT_K
        rows, tasks, meta = [], [], {}
        for sp in range(C.HOLDOUT_SPLITS):
            perm = rng.permutation(len(org)); halves = (perm[: len(org) // 2], perm[len(org) // 2:])
            for direction, (fit_idx, test_idx) in enumerate([halves, halves[::-1]]):
                dfit = D.rank(org.iloc[fit_idx]); dtest = D.rank(org.iloc[test_idx])
                rho = stats.spearmanr(dfit.cnci, dfit.ccnci)[0]; u = Q.pseudo_obs(dfit)
                fb = Q.fit("sbb1", u)[0]
                pars = {"gauss": Q.calibrate("gauss", rho), "sbb1": Q.calibrate("sbb1", rho, fixed=fb[0])}
                for fam, par in pars.items():
                    key = ("Organization", f"holdout{sp}-{direction}", fam)
                    tasks.append((fam, par, len(dtest), Ks, C.HOLDOUT_REPS, key))
                    meta[key] = dict(split=sp, direction=direction, family=fam, N_fit=len(dfit), N_test=len(dtest), rho_fit=rho,
                                     rho_test=stats.spearmanr(dtest.cnci, dtest.ccnci)[0], obs={K: M.shared(dtest, K) for K in Ks})
        sims = run_sims(tasks, a.workers)
        for key, sim in sims.items():
            m = meta[key]
            for j, K in enumerate(Ks):
                s = sim[:, j]; o = m["obs"][K]
                rows.append(dict(split=m["split"], direction=m["direction"], family=m["family"], K=K, N_fit=m["N_fit"], N_test=m["N_test"],
                                 rho_fit=m["rho_fit"], rho_test=m["rho_test"], observed_test=o, mean=s.mean(), p05=np.percentile(s, 5), p95=np.percentile(s, 95),
                                 mid_p=(s > o).mean() + 0.5 * (s == o).mean(), within_5_95=bool(np.percentile(s, 5) <= o <= np.percentile(s, 95))))
        ho = save(rows, out, "holdout_splits.csv")
        save(ho.groupby(["family", "K"]).agg(evaluations=("mid_p", "size"), coverage_5_95=("within_5_95", "mean"), median_mid_p=("mid_p", "median"),
                                              share_mid_p_below_0_05=("mid_p", lambda x: (x < 0.05).mean()), mean_obs_minus_ref=("observed_test", "mean")).reset_index()
             .assign(mean_obs_minus_ref=lambda df: df.mean_obs_minus_ref - ho.groupby(["family", "K"])["mean"].mean().values), out, "holdout_summary.csv")
    if "hierarchy" in steps:
        print("[hierarchy]")
        if not Path(a.edges).exists():
            sys.exit(f"hierarchy map not found: {a.edges}. Pass --edges (see data/README.md), "
                     "use data/synthetic_hierarchy_edges.csv with synthetic data, or omit the hierarchy step with --steps.")
        org = base["Organization"]; edges_all = H.load_edges(a.edges, list(org.name))
        ec = save(H.edge_checks(edges_all, org), rst, "hierarchy_edges_expanded.csv")
        both = set(ec.child) & set(ec.parent)
        save([dict(edges=len(ec), mapped_units=len(set(ec.child) | set(ec.parent)), parent_records=ec.parent.nunique(), child_records=ec.child.nunique(),
                   both=len(both), child_docs_le_parent=int(ec.child_docs_le_parent.sum()))], out, "hierarchy_edge_checks.csv")
        save(ec.groupby("relation_type").size().rename("edges").reset_index(), out, "hierarchy_relation_types.csv")
        # main map, and a sensitivity map in which legally separate joint-venture campuses are treated as independent units
        maps = {"main": edges_all, "joint ventures unmapped": edges_all[~edges_all.relation_type.str.startswith("joint-venture")]}
        Ks = C.K_HIER
        info, tasks, dmap = [], {}, {}
        for mp, edges in maps.items():
            for vn, drop in H.variants(org, edges).items():
                d = D.rank(org[~org.name.isin(drop)]); u = Q.pseudo_obs(d); rho = stats.spearmanr(d.cnci, d.ccnci)[0]
                vkey = "V0" if vn.startswith("V0") else vn.split()[0]
                if mp != "main" and vkey not in ("V0", "V1"):
                    vkey += "-jv"   # V0 and V1 do not depend on the joint-venture edges and reuse the main-map draws
                dmap[(mp, vn)] = (d, vkey)
                fit_g = Q.fit("gauss", u); fit_t = Q.fit("tfree", u); fit_b = Q.fit("sbb1", u)
                cal_b = Q.calibrate("sbb1", rho, fixed=fit_b[0][0]); cal_g = Q.calibrate("gauss", rho)
                info.append(dict(map=mp, variant=vn, N=len(d), removed=len(drop), spearman=rho, kendall=stats.kendalltau(d.cnci, d.ccnci)[0],
                                 pseudo_aic_gauss=2 - 2 * fit_g[1], pseudo_aic_tfree=4 - 2 * fit_t[1], nu_hat=fit_t[0][1],
                                 pseudo_aic_sbb1=4 - 2 * fit_b[1], sbb1_theta=fit_b[0][0], sbb1_delta_mpl=fit_b[0][1], sbb1_delta_cal=cal_b[1],
                                 spearman_shift_intl=stats.spearmanr(d.r2 - d.r1, d.intl)[0]))
                # V0 uses exactly the same keys (and therefore the same replicates) as copula_references.csv
                Kv = KS["Organization"] if vkey == "V0" else Ks
                for fam, par in (("gauss", cal_g), ("sbb1", cal_b)):
                    key = ("Organization", vkey, fam + "_cal"); tasks[key] = (fam, par, len(d), Kv, a.reps, key)
        save(info, out, "hierarchy_variants.csv")
        sims = run_sims(list(tasks.values()), a.workers); rows, bnd, prof, sw, red = [], [], [], [], []
        for (mp, vn), (d, vkey) in dmap.items():
            Kv = KS["Organization"] if vkey == "V0" else Ks
            obs = {K: M.shared(d, K) for K in Kv}
            for fam in ("gauss_cal", "sbb1_cal"):
                for r in summarise(sims[("Organization", vkey, fam)], Kv, obs, map=mp, variant=vn, N=len(d), family=fam):
                    if r["K"] in Ks: rows.append(r)
            for K in Ks:
                bnd.append(dict(map=mp, variant=vn, **M.boundary(d, K))); prof += [dict(map=mp, variant=vn, **r) for r in M.profiles(d, K)]
                if K == 20: sw.append(dict(map=mp, variant=vn, **M.switch_list(d, K)))
        for mp, edges in maps.items():
            involved = set(edges.child) | set(edges.parent)
            for K in Ks:
                for lab, col in (("CNCI", "r1"), ("Collab-CNCI", "r2")):
                    inv, redn = H.redundancy(org, edges, K, col); red.append(dict(map=mp, K=K, list=lab, hierarchy_mapped_units=inv, name_map_redundant_slots=redn))
                ex, en = M.switches(org, K)
                red.append(dict(map=mp, K=K, list="Switched units", hierarchy_mapped_units=int(sum(n in involved for n in pd.concat([ex, en]).name)),
                                name_map_redundant_slots=np.nan, switched_units=len(ex) + len(en)))
        save(rows, out, "hierarchy_overlap.csv"); save(bnd, out, "hierarchy_boundary.csv")
        save(prof, rst, "hierarchy_profiles_full.csv"); save(suppress(prof), out, "hierarchy_profiles.csv")
        save(sw, rst, "hierarchy_switches_k20.csv"); save(red, out, "hierarchy_redundancy.csv")

    if "regressions" in steps:
        print("[regressions]")
        rows, boots = [], []
        names_c = ["const", "dmult", "bi", "tri", "quad", "log_docs"]
        for lev in LEVELS:
            d = base[lev]
            for oc in ("cnci", "ccnci"):
                rows += G.tidy(G.intl_model(d, oc), ["const", "intl_per10pp", "log_docs"], level=lev, model=f"{oc} ~ intl + log docs", sample="primary")
            m = G.composition_model(d); rows += G.tidy(m, names_c, level=lev, model="diff ~ shares + log docs", sample="primary")
            for off in C.ALR_OFFSETS:
                rows += G.tidy(G.composition_model(d, alr_offset=off), ["const", "alr_dmult", "alr_bi", "alr_tri", "alr_quad", "log_docs"],
                               level=lev, model=f"diff ~ ALR(offset {off}) + log docs", sample="primary")
            r = np.abs(m.resid); keep = r < np.quantile(r, 0.99)
            rows += G.tidy(G.composition_model(d[keep]), names_c, level=lev, model="diff ~ shares + log docs", sample="largest 1% |residuals| removed")
            for t in (250, 500):
                rows += G.tidy(G.composition_model(D.eligible(raw[lev], t)), names_c, level=lev, model="diff ~ shares + log docs", sample=f">= {t} documents")
            boots.append(dict(level=lev, **G.coef_diff_bootstrap(d, B=a.boot)))
        save(rows, out, "regressions.csv"); save(boots, out, "bootstrap_coef_diff.csv")

    if "figures" in steps:
        print("[figures]")
        FG.figure1(pd.read_csv(out / "copula_references.csv"), out / "figure1.png")
        FG.figure2(pd.read_csv(out / "continuous_k.csv"), out / "figure2.png")

    run_info["cal_n"] = C.CAL_N
    run_info["seconds"] = round(time.time() - t0, 1)
    import pyvinecopulib, scipy, statsmodels
    run_info["versions"] = dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                                pyvinecopulib=pyvinecopulib.__version__)
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(f"done in {run_info['seconds']} s")

if __name__ == "__main__":
    main()
