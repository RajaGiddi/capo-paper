"""
Minimap2 ava-pb baseline for any registered genome.

Runs `minimap2 -x ava-pb reads.fastq reads.fastq` (all-vs-all HiFi overlap),
parses the PAF, and computes precision/recall vs the same ground-truth edge
labels CAPO uses. Output is written to results/<genome>/minimap_baseline.json
so CAPO and minimap2 can be compared on matched P/R.

Usage:
    python evaluation/minimap_baseline.py --genome ecoli
    python evaluation/minimap_baseline.py --genome bsubtilis \
        --minimap2 /path/to/minimap2

Notes:
    - Minimap2 doesn't emit calibrated probabilities, so no ECE comparison.
    - The "score" we rank by is chain-score `s1` (col 14, tag `s1:i:`), which
      is a better continuous ranker than mapQ (which saturates at 60).
    - Candidate-set alignment: minimap2 reports only found overlaps. For the
      comparison, we treat any pair minimap2 did NOT report as score = 0,
      i.e. as if it had been considered and rejected. CAPO's 462k candidate
      set is the denominator in both cases.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from capo.genomes import get as get_genome


def run_minimap2(minimap2_bin: str, reads_fastq: Path, paf_out: Path,
                  preset: str = "ava-pb", threads: int = 4) -> None:
    """Run minimap2 all-vs-all and write PAF. Streams minimap2's stderr
    progress lines directly to our stderr so the user sees live output."""
    paf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [minimap2_bin, "-x", preset, "-t", str(threads),
           str(reads_fastq), str(reads_fastq)]
    print("Running:", " ".join(cmd), flush=True)
    with open(paf_out, "w") as f:
        # stderr=None inherits our stderr so minimap2's own progress is live
        proc = subprocess.run(cmd, stdout=f, stderr=None)
    if proc.returncode != 0:
        raise RuntimeError(f"minimap2 exited {proc.returncode}")
    print(f"Wrote {paf_out}", flush=True)


def parse_paf(paf_path: Path, read_id_to_idx: dict[str, int]) -> dict:
    """
    Parse PAF into {(i,j): score} with i<j. Collapse multiple hits for the
    same pair (different strands, split chains) by taking the max score.

    Returns a dict mapping (i,j) -> {'s1': chain_score, 'mapq': mapq, 'n': n_hits}.
    """
    pairs: dict[tuple[int, int], dict] = {}
    n_lines = 0
    n_dropped = 0
    with open(paf_path) as f:
        for line in f:
            n_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            q, t = fields[0], fields[5]
            mapq = int(fields[11])
            if q == t:
                continue  # self-hit

            qi = read_id_to_idx.get(q)
            ti = read_id_to_idx.get(t)
            if qi is None or ti is None:
                n_dropped += 1
                continue

            key = (min(qi, ti), max(qi, ti))

            # Chain score is in an optional tag, default to alignment block len
            s1 = None
            for tag in fields[12:]:
                if tag.startswith("s1:i:"):
                    s1 = int(tag[5:])
                    break
            if s1 is None:
                s1 = int(fields[10])  # residue-match / align-block proxy

            prev = pairs.get(key)
            if prev is None or s1 > prev["s1"]:
                pairs[key] = {"s1": s1, "mapq": mapq, "n": 1}
            else:
                prev["n"] += 1

    print(f"Parsed {n_lines} PAF lines, {len(pairs)} unique read pairs "
          f"({n_dropped} hits dropped for unknown IDs)")
    return pairs


def compute_metrics(pairs: dict, candidate_pairs: list, true_labels: dict,
                     score_key: str = "s1") -> dict:
    """
    Compute P/R over the CAPO candidate set. Pairs not in the PAF are
    treated as score = 0.
    """
    scores, ys = [], []
    for (i, j) in candidate_pairs:
        key = (min(i, j), max(i, j))
        s = pairs.get(key, {}).get(score_key, 0)
        scores.append(s)
        ys.append(1.0 if true_labels.get(key, 0.0) >= 0.5 else 0.0)

    scores = np.array(scores, dtype=float)
    ys = np.array(ys, dtype=float)
    n_genuine = int(ys.sum())

    # Threshold sweep on chain score
    uniq = np.unique(scores)
    # Subsample thresholds for speed on large sets
    if len(uniq) > 200:
        uniq = np.quantile(scores, np.linspace(0, 1, 201))
    sweep = []
    for t in uniq:
        tp = int(((scores > t) & (ys >= 0.5)).sum())
        fp = int(((scores > t) & (ys < 0.5)).sum())
        fn = int(((scores <= t) & (ys >= 0.5)).sum())
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-6)
        sweep.append({"threshold": float(t), "precision": float(P),
                      "recall": float(R), "f1": float(F1),
                      "tp": tp, "fp": fp, "fn": fn})

    # Best F1, and nearest-to-match-recall points
    best_f1 = max(sweep, key=lambda x: x["f1"])

    def at_recall(target_r):
        ok = [s for s in sweep if s["recall"] >= target_r]
        return min(ok, key=lambda x: x["recall"] - target_r) if ok else None

    match_75 = at_recall(0.75)
    match_751 = at_recall(0.751)

    return {
        "n_pairs": len(candidate_pairs),
        "n_genuine": n_genuine,
        "n_minimap_pairs": len(pairs),
        "best_f1": best_f1,
        "at_recall_0.75": match_75,
        "at_recall_0.751": match_751,
        "sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser(description="Minimap2 baseline for CAPO genomes")
    ap.add_argument("--genome", required=True, help="Registered genome name")
    from capo.genomes import REPO_ROOT
    ap.add_argument("--minimap2",
                    default=str(REPO_ROOT.parent / "minimap2" / "minimap2"),
                    help="Path to minimap2 binary (default: ../minimap2/minimap2)")
    ap.add_argument("--preset", default="ava-pb",
                    help="minimap2 preset (ava-pb, ava-ont)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="Re-run minimap2 even if PAF exists")
    args = ap.parse_args()

    cfg = get_genome(args.genome)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    paf_path = cfg.results_dir / "minimap2.paf"
    if args.force or not paf_path.exists():
        run_minimap2(args.minimap2, cfg.reads, paf_path,
                      preset=args.preset, threads=args.threads)
    else:
        print(f"Reusing existing PAF at {paf_path} (--force to re-run)")

    # Build read-id → candidate index mapping (same order as reads.fastq)
    from Bio import SeqIO
    read_ids = [str(r.id) for r in SeqIO.parse(str(cfg.reads), "fastq")]
    read_id_to_idx = {rid: i for i, rid in enumerate(read_ids)}
    print(f"Loaded {len(read_ids)} reads")

    pairs = parse_paf(paf_path, read_id_to_idx)

    # Load CAPO candidate set and ground truth
    raw_labels = json.load(open(cfg.edge_labels))
    true_labels = {eval(k): v for k, v in raw_labels.items()}
    candidate_pairs = list(true_labels.keys())
    print(f"Candidate set: {len(candidate_pairs)} pairs")

    metrics = compute_metrics(pairs, candidate_pairs, true_labels,
                                score_key="s1")

    out = cfg.results_dir / "minimap_baseline.json"
    # Trim sweep to every-10th row for readability; keep full inside file
    summary = {k: v for k, v in metrics.items() if k != "sweep"}
    summary["sweep_sampled"] = metrics["sweep"][::max(1, len(metrics["sweep"]) // 20)]
    summary["full_sweep_len"] = len(metrics["sweep"])
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out}")

    print("\n=== Minimap2 ava-pb baseline ===")
    print(f"  Candidate pairs:        {metrics['n_pairs']:,}")
    print(f"  Genuine overlaps:       {metrics['n_genuine']:,}")
    print(f"  Pairs reported by mm2:  {metrics['n_minimap_pairs']:,}")
    b = metrics["best_f1"]
    print(f"\n  Best F1:  t={b['threshold']:.1f}  "
          f"P={b['precision']:.3f}  R={b['recall']:.3f}  F1={b['f1']:.3f}")
    if metrics["at_recall_0.751"]:
        r = metrics["at_recall_0.751"]
        print(f"  @ R≈0.751: t={r['threshold']:.1f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")


if __name__ == "__main__":
    main()
