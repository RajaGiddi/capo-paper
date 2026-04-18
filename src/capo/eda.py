"""
Genome loader and exploratory data analysis.

Loads a reference genome from FASTA, computes metadata, and identifies
repeat structures. Reusable across different genomes.

Usage:
    python data/eda_genome.py data/ecoli_genome.fasta
"""
import json
import sys
from pathlib import Path
from Bio import SeqIO
import numpy as np


# ── Known repeat annotations (add entries for new genomes) ──────────

KNOWN_REPEATS = {
    "NC_000913.3": {
        "organism": "Escherichia coli K-12 MG1655",
        "repeats": [
            {"start": 223771,  "end": 229357,  "name": "rrnH", "type": "rRNA_operon"},
            {"start": 2726188, "end": 2731597, "name": "rrnE", "type": "rRNA_operon"},
            {"start": 3427221, "end": 3432653, "name": "rrnD", "type": "rRNA_operon"},
            {"start": 3941599, "end": 3947050, "name": "rrnC", "type": "rRNA_operon"},
            {"start": 4035531, "end": 4040996, "name": "rrnA", "type": "rRNA_operon"},
            {"start": 4166659, "end": 4172117, "name": "rrnB", "type": "rRNA_operon"},
            {"start": 4208147, "end": 4213567, "name": "rrnG", "type": "rRNA_operon"},
        ]
    }
}


def load_genome(fasta_path: str) -> tuple[str, str]:
    """
    Load genome from FASTA. Returns (sequence, accession).
    Sequence is uppercased.
    """
    record = SeqIO.read(fasta_path, "fasta")
    return str(record.seq).upper(), record.id


def compute_gc(seq: str, window: int = 0) -> float | list[float]:
    """
    GC content. If window > 0, returns per-window GC as a list.
    """
    if window <= 0:
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        return gc

    gc_track = []
    for i in range(0, len(seq) - window + 1, window):
        w = seq[i:i+window]
        gc_track.append((w.count('G') + w.count('C')) / len(w))
    return gc_track


def find_exact_repeats(seq: str, kmer_size: int = 50, min_copies: int = 2) -> list[dict]:
    """
    Find exact repeat k-mers in the genome.
    Returns list of {kmer, count, positions} for k-mers with >= min_copies.
    """
    from collections import Counter
    kmers = {}
    for i in range(len(seq) - kmer_size + 1):
        kmer = seq[i:i+kmer_size]
        if kmer not in kmers:
            kmers[kmer] = []
        kmers[kmer].append(i)

    repeats = []
    for kmer, positions in kmers.items():
        if len(positions) >= min_copies:
            repeats.append({
                'kmer': kmer[:20] + '...',
                'count': len(positions),
                'positions': positions[:10],  # cap for readability
            })

    repeats.sort(key=lambda x: -x['count'])
    return repeats[:50]  # top 50


def genome_summary(fasta_path: str, cfg=None) -> dict:
    """Full EDA summary of a genome.

    If cfg (GenomeConfig) is supplied, known_repeats and organism come from
    the registry; otherwise fall back to the legacy KNOWN_REPEATS dict keyed
    by accession.
    """
    seq, accession = load_genome(fasta_path)
    gc = compute_gc(seq)
    gc_windows = compute_gc(seq, window=10000)

    if cfg is not None:
        organism = cfg.organism
        known_reps = list(cfg.known_repeats)
    else:
        known = KNOWN_REPEATS.get(accession, {})
        organism = known.get("organism", "Unknown")
        known_reps = known.get("repeats", [])

    # Basic stats
    summary = {
        "accession": accession,
        "organism": organism,
        "length": len(seq),
        "gc_content": round(gc, 5),
        "gc_std_10kb": round(float(np.std(gc_windows)), 5) if gc_windows else 0.0,
        "gc_min_10kb": round(float(np.min(gc_windows)), 5) if gc_windows else 0.0,
        "gc_max_10kb": round(float(np.max(gc_windows)), 5) if gc_windows else 0.0,
        "known_repeats": known_reps,
        "repeat_locs": [[r["start"], r["end"], i] for i, r in enumerate(known_reps)],
    }

    print(f"Genome: {organism}")
    print(f"  Accession: {accession}")
    print(f"  Length: {len(seq):,} bp")
    print(f"  GC: {gc:.3f} (10kb window std: {summary['gc_std_10kb']:.4f})")
    print(f"  Known repeats: {len(known_reps)}")
    for r in known_reps:
        size = r['end'] - r['start']
        print(f"    {r['name']}: {r['start']:,}–{r['end']:,} ({size:,} bp, {r['type']})")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genome EDA")
    ap.add_argument("--genome", help="Name from src/capo/genomes.py registry "
                    "(ecoli, bsubtilis, scerevisiae, ...)")
    ap.add_argument("--fasta", help="Path to FASTA (overrides --genome)")
    args = ap.parse_args()

    cfg = None
    if args.fasta:
        fasta_path = args.fasta
    else:
        if not args.genome:
            ap.error("Pass --genome NAME or --fasta PATH")
        from capo.genomes import get
        cfg = get(args.genome)
        fasta_path = str(cfg.fasta)

    summary = genome_summary(fasta_path, cfg=cfg)

    if cfg is not None:
        out_path = cfg.meta
    else:
        out_path = Path(str(Path(fasta_path).with_suffix('')) + '_meta.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetadata saved to {out_path}")
