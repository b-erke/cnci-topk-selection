# Code — From Rank Agreement to Shortlist Overlap: Collaboration-Normalized Citation Impact in Artificial Intelligence Research
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23434885.svg)](https://doi.org/10.5281/zenodo.23434885)
Berke Akkaya (Istanbul University). Analysis code for the manuscript of the same title. The code regenerates every table, figure and in-text statistic of the manuscript from Clarivate InCites exports.

**Data are not included.** The study uses Clarivate InCites Benchmarking & Analytics exports obtained under a subscription licence; they cannot be redistributed. Results derived from them and the curated organizational hierarchy map used in the study are available from the author on request. To run the pipeline without licensed data, use the synthetic data generator (Section 3).

## 1. Requirements
Python 3.11–3.13. Install the pinned versions (pyvinecopulib is required for the analysis and the copula tests; it installs from PyPI wheels):
```
python -m pip install -r requirements.txt
```

## 2. Run on licensed data
Place the two InCites exports in a folder (for example `data/raw/`); required file names, filters and columns are listed in `data/README.md`. Supply a hierarchy map with `--edges` (format in `data/README.md`).
```
python scripts/run_all.py --data-dir data/raw --edges path/to/hierarchy_edges.csv --out-dir results   # full run (about 50 min on 2 cores)
python scripts/run_all.py --data-dir data/raw --edges path/to/hierarchy_edges.csv --out-dir /tmp/check --quick
python scripts/run_all.py --data-dir data/raw --steps topk,regressions                                # selected steps only
```
Steps: descriptives, topk, fits, references, modelsel, gof, holdout, hierarchy, regressions, figures. Options: `--reps` (copula replicates, default 10,000), `--boot` (coefficient bootstrap, default 10,000), `--boot-modelsel` (default 200), `--gof-boot` (goodness-of-fit bootstrap, default 1,000), `--workers`.

Independent verification without pyvinecopulib (NumPy/SciPy only) and a check of a manuscript file against the outputs:
```
python scripts/verify_independent.py --results results [--data-dir data/raw]
python scripts/check_manuscript.py --docx manuscript.docx --results results
```

## 3. Run without licensed data
```
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/run_all.py --data-dir data/synthetic --edges data/synthetic_hierarchy_edges.csv --out-dir /tmp/synthetic --quick
python -m pytest tests          # unit tests + end-to-end smoke run on synthetic data
```
The synthetic values are random and do not reproduce the manuscript's results.

## 4. Determinism
All random numbers derive from `BASE_SEED = 20261013` (`src/cnci_topk/config.py`). Copula replicates use seeds `[BASE_SEED, crc32(level|variant|family) mod 2^31, replicate]`. A full run with the same package versions and input files reproduces every output exactly; `run_info.json` records versions and settings.

## 5. Methods notes
* Pseudo-observations u = rank/(N+1) use ascending ranks, so top-ranked units lie near u = 1. Pseudo-AIC = 2k − 2·pseudo-log-likelihood; t rows with fixed ν count k = 1, "t, ν estimated" and survival BB1 count k = 2.
* Reference distributions use parameters calibrated to the observed Spearman correlation (survival BB1: θ fixed at its pseudo-likelihood estimate, δ calibrated); references at the fitted parameters are reported alongside.
* Goodness-of-fit (`gof` step): Cramér–von Mises statistics over the unit square and over the upper quadrants [q,1]² (q = 0.8, 0.9), parametric bootstrap with refitting within the family; p-values are conditional on the family tested.
* RBO is the extrapolated full-depth value for two complete rankings of the same units.
* Outputs: aggregate files are written to `--out-dir`; files listing named units with InCites values or ranks are written to `<out-dir>/restricted/`. Do not redistribute outputs generated from licensed data without checking your licence. A unit test checks that no aggregate CSV contains unit-level columns.

## 6. Licence
Code under the MIT licence (`LICENSE`). InCites data © Clarivate; not included.
