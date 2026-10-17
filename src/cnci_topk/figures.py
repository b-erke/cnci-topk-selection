"""Figures 1 and 2 (300 dpi PNG)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

def figure1(ref, path):
    """ref: copula_references rows (level, family, K, observed, mean, p05, p95) for families gauss and sbb1_cal."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, lev in zip(axs, ["Country/region", "Organization"]):
        g = ref[(ref.level == lev) & (ref.family == "gauss_cal")].sort_values("K")
        b = ref[(ref.level == lev) & (ref.family == "sbb1_cal")].sort_values("K")
        K = g.K.to_numpy(); x = np.arange(len(K))
        for df, dx, col, mk, lab in [(g, -0.2, C1, "o", "Gaussian reference (5th–95th)"), (b, 0.0, C2, "^", "Survival BB1 reference (5th–95th)")]:
            ax.errorbar(x + dx, df["mean"] / K, yerr=[(df["mean"] - df.p05) / K, (df.p95 - df["mean"]) / K], fmt=mk, ms=6.5,
                        color=col, ecolor=col, elinewidth=2, capsize=0, label=lab)
        ax.plot(x + 0.2, g.observed / K, "s", ms=7, color=INK, label="Observed", zorder=5)
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in K]); ax.set_xlabel("Selection capacity K")
        ax.set_title(f"{lev} (N = {int(g.N.iloc[0]):,})", fontsize=11, loc="left")
        ax.set_ylim(0, 1.02); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].set_ylabel("Shared selections / K")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)

def figure2(sweep, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, lev in zip(axs, ["Country/region", "Organization"]):
        s = sweep[sweep.level == lev]
        ax.plot(s.K / s.N, s.jaccard, color=C1, lw=1.6)
        ax.set_title(f"{lev} (N = {int(s.N.iloc[0]):,})", fontsize=11, loc="left")
        ax.set_xlabel("Selected fraction (K/N)"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].set_ylabel("Jaccard similarity")
    fig.tight_layout(); fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)
