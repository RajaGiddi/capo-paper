"""
One-off patch: add `mm2_scores` to an existing results/<genome>/capo_arrays.npz
by re-parsing the cached PAF, without re-running the encoder or per-pair
feature computation.

Use this if you've already run capo_on_minimap before the mm2_scores field
was added to the npz output -- the full pipeline re-run takes ~15 min per
genome, this takes seconds.

Usage:
    python scripts/patch_mm2_scores.py                 # ecoli + bsubtilis
    python scripts/patch_mm2_scores.py --genomes toy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from Bio import SeqIO

from capo.genomes import get as get_genome
from capo.evaluation.capo_on_minimap import parse_paf_pairs


def patch(genome: str) -> None:
    cfg = get_genome(genome)
    paf_path = cfg.results_dir / "minimap2.paf"
    arr_path = cfg.results_dir / "capo_arrays.npz"

    if not paf_path.exists():
        raise FileNotFoundError(f"{paf_path} not found")
    if not arr_path.exists():
        raise FileNotFoundError(f"{arr_path} not found")

    read_id_to_idx = {
        str(r.id): i for i, r in enumerate(SeqIO.parse(str(cfg.reads), "fastq"))
    }

    mm2_pair_scores = parse_paf_pairs(paf_path, read_id_to_idx)
    pairs = sorted(mm2_pair_scores.keys())
    mm2_scores = np.array([mm2_pair_scores[p] for p in pairs], dtype=np.int32)

    arr = np.load(arr_path)
    if len(arr["labels"]) != len(mm2_scores):
        raise RuntimeError(
            f"{genome}: length mismatch between existing npz "
            f"({len(arr['labels'])}) and PAF-derived pairs "
            f"({len(mm2_scores)}). Re-run capo_on_minimap in full."
        )

    payload = {k: arr[k] for k in arr.files}
    payload["mm2_scores"] = mm2_scores
    np.savez_compressed(arr_path, **payload)
    print(f"Patched {genome}: added mm2_scores ({len(mm2_scores):,} pairs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genomes", nargs="+", default=["ecoli", "bsubtilis"])
    args = ap.parse_args()
    for genome in args.genomes:
        patch(genome)


if __name__ == "__main__":
    main()
