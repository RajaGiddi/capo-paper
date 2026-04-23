"""
Two-panel precision-recall curves for minimap2, CAPO-BF, and CAPO-LogReg
on E. coli and B. subtilis.

Reads per-pair arrays written by capo_on_minimap.py into
results/<genome>/capo_arrays.npz, restricts to the held-out split, and
renders Figure 3 of the paper.

Usage:
    python scripts/plot_pr_curves.py
    python scripts/plot_pr_curves.py --genomes ecoli bsubtilis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from capo.genomes import get as get_genome


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "figures"

PALETTE = {
    "logreg":  "#264653",  # deep slate (paper hero)
    "bf":      "#E76F51",  # terracotta (secondary accent)
    "mm2":     "#8C8C8C",  # mid-grey baseline
    "text":    "#1a1a1a",
    "muted":   "#6a6a6a",
    "grid":    "#e6e6e6",
}


def style_axes():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.titlecolor": PALETTE["text"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 6,
        "axes.titlepad": 10,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def load_held_out(genome: str) -> dict[str, np.ndarray]:
    cfg = get_genome(genome)
    path = cfg.results_dir / "capo_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m capo.evaluation.capo_on_minimap "
            f"--genome {genome}` first (with the updated pipeline)."
        )
    arr = np.load(path)
    if "mm2_scores" not in arr.files:
        raise KeyError(
            f"{path} does not contain 'mm2_scores'. Re-run capo_on_minimap to "
            f"regenerate the arrays with the updated pipeline."
        )
    held = ~arr["fit_mask"]
    return {
        "labels":    arr["labels"][held].astype(int),
        "mm2":       arr["mm2_scores"][held].astype(float),
        "bf":        arr["bf_scores"][held],
        "logreg":    arr["lr_scores"][held],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genomes", nargs=2, default=["ecoli", "bsubtilis"],
                    metavar=("LEFT", "RIGHT"))
    ap.add_argument("--out", default=str(FIG_DIR / "pr_curves"),
                    help="Output path stem (no extension)")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    style_axes()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True,
                             constrained_layout=True)
    pretty = {
        "ecoli": r"$\it{E.\ coli}$ K-12 MG1655",
        "bsubtilis": r"$\it{B.\ subtilis}$ 168",
        "scerevisiae": r"$\it{S.\ cerevisiae}$ S288C",
        "toy": "Toy genome",
    }

    curves = [
        ("logreg", "CAPO-LogReg",  PALETTE["logreg"], "-",  1.8),
        ("bf",     "CAPO-BF",      PALETTE["bf"],     "-",  1.3),
    ]

    for ax, genome in zip(axes, args.genomes):
        data = load_held_out(genome)
        y = data["labels"]

        for key, label, color, linestyle, lw in curves:
            p, r, _ = precision_recall_curve(y, data[key])
            ap_score = average_precision_score(y, data[key])
            ax.plot(r, p, linestyle=linestyle, color=color, linewidth=lw,
                    label=f"{label}  (AP = {ap_score:.3f})")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Recall")
        ax.set_title(pretty.get(genome, genome))
        ax.set_aspect("equal")
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", handletextpad=0.7,
                  borderaxespad=0.4, fontsize=9)

    axes[0].set_ylabel("Precision")

    for ext in ("pdf", "png"):
        path = Path(f"{args.out}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
