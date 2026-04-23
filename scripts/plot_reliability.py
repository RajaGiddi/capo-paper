"""
Two-panel reliability diagram for CAPO-LogReg on E. coli and B. subtilis.

Reads per-pair arrays written by capo_on_minimap.py into
results/<genome>/capo_arrays.npz, bins held-out predictions into 10
equal-width bins on [0, 1], and renders Figure 2 of the paper.

Usage:
    python -m scripts.plot_reliability
    # or, with a custom pair of genomes:
    python -m scripts.plot_reliability --genomes ecoli bsubtilis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capo.genomes import get as get_genome


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "figures"


def per_bin_calibration(probs: np.ndarray, labels: np.ndarray,
                         n_bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, emp_rate, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if in_bin.sum() == 0:
            continue
        mean_pred.append(float(probs[in_bin].mean()))
        emp_rate.append(float(labels[in_bin].mean()))
        counts.append(int(in_bin.sum()))
    ece = float(np.sum(
        [c * abs(p - y) for p, y, c in zip(mean_pred, emp_rate, counts)]
    ) / sum(counts))
    return {
        "mean_pred": np.array(mean_pred),
        "emp_rate": np.array(emp_rate),
        "counts": np.array(counts),
        "ece": ece,
    }


def load_held_out(genome: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = get_genome(genome)
    path = cfg.results_dir / "capo_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m capo.evaluation.capo_on_minimap "
            f"--genome {genome}` first."
        )
    arr = np.load(path)
    held_out = ~arr["fit_mask"]
    return arr["lr_probs"][held_out], arr["labels"][held_out].astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genomes", nargs=2, default=["ecoli", "bsubtilis"],
                    metavar=("LEFT", "RIGHT"))
    ap.add_argument("--out", default=str(FIG_DIR / "reliability"),
                    help="Output path stem (no extension)")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), sharey=True)
    pretty = {
        "ecoli": r"$\it{E.\ coli}$ K-12 MG1655",
        "bsubtilis": r"$\it{B.\ subtilis}$ 168",
        "scerevisiae": r"$\it{S.\ cerevisiae}$ S288C",
        "toy": "Toy genome",
    }

    for ax, genome in zip(axes, args.genomes):
        probs, labels = load_held_out(genome)
        cal = per_bin_calibration(probs, labels, n_bins=10)

        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888",
                linewidth=1.0, label="Perfect calibration")
        sizes = 20.0 + 400.0 * cal["counts"] / max(cal["counts"].max(), 1)
        ax.scatter(cal["mean_pred"], cal["emp_rate"],
                   s=sizes, alpha=0.75, color="#2E75B6",
                   edgecolor="white", linewidth=0.5,
                   label="CAPO-LogReg")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(f"{pretty.get(genome, genome)}  (ECE = {cal['ece']:.3f})")
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    axes[0].set_ylabel("Empirical positive rate")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = Path(f"{args.out}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
