"""Analysis constants. Every number that shapes a reported result is defined here."""
BASE_SEED = 20261013
MIN_DOCS = 100                                 # primary eligibility rule
DOC_THRESHOLDS = [50, 100, 200, 250, 500]      # publication-volume sensitivity
UK_CONSTITUENTS = ["ENGLAND", "SCOTLAND", "WALES", "NORTHERN IRELAND"]
K_COUNTRY = [5, 10, 20, 30, 50]
K_ORG = [5, 10, 20, 50, 100, 200]
K_HIER = [10, 20, 50, 100, 200]
RBO_P = [0.90, 0.98]
REPS = 10_000                                  # copula reference replicates (override with --reps)
BOOT_COEF = 10_000                             # paired bootstrap replicates for coefficient difference
BOOT_MODELSEL = 200                            # bootstrap replicates for copula model-selection stability
T_NU_GRID = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40, 50]   # pyvinecopulib caps nu at 50
ALR_OFFSETS = [0.1, 0.5, 1.0]
IQR_METHOD = "closest_observation"             # Hyndman & Fan Type 3 (as reported in Table 4)
FILES = {  # expected InCites export file names inside --data-dir
    "locations": "Incites_Locations.xlsx",
    "organizations": "Incites_Organizations.xlsx",
}
CAL_N = 300_000                                # Monte Carlo draws per Spearman evaluation during calibration
HOLDOUT_SPLITS = 20                           # random half-splits of organizations (both directions evaluated)
HOLDOUT_K = [10, 25, 50, 100]                 # cutoffs in the held-out half (about half of K = 20, 50, 100, 200)
HOLDOUT_REPS = 1000
GOF_B = 1000                                  # parametric-bootstrap replicates per goodness-of-fit test
GOF_Q = [0.8, 0.9]                            # upper-quadrant thresholds for the tail-focused statistic
