"""
HiFi read simulator for arbitrary genomes.

Reused from Paper 1 with scale adjustments for E. coli (4.6Mb, ~9200 reads).
Ground-truth labeling uses efficient spatial indexing instead of O(n^2) brute force.
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# HiFi error model
EPSILON_SUB   = 0.003
EPSILON_INDEL = 0.001
HP_THRESHOLD  = 3

COMPLEMENT = str.maketrans('ACGT', 'TGCA')

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def apply_hifi_errors(seq: str, rng) -> tuple[str, list[int]]:
    bases = list(seq)
    quals = []
    i = 0
    out_bases = []

    while i < len(bases):
        b = bases[i]
        hp_len = 1
        while i + hp_len < len(bases) and bases[i+hp_len] == b:
            hp_len += 1
        indel_rate = EPSILON_INDEL * (hp_len if hp_len >= HP_THRESHOLD else 1)

        r = rng.random()
        if r < EPSILON_SUB:
            others = [x for x in 'ACGT' if x != b]
            out_bases.append(rng.choice(others))
            quals.append(15)
            i += 1
        elif r < EPSILON_SUB + indel_rate:
            i += 1  # deletion
        elif r < EPSILON_SUB + 2 * indel_rate:
            out_bases.append(rng.choice(list('ACGT')))
            out_bases.append(b)
            quals.extend([15, 30])
            i += 1
        else:
            out_bases.append(b)
            quals.append(30)
            i += 1

    return ''.join(out_bases), quals


def simulate_hifi_reads(
    genome:     str,
    coverage:   int   = 30,
    mean_len:   int   = 15_000,
    sd_len:     int   = 3_000,
    seed:       int   = 42,
) -> list[dict]:
    """Simulate PacBio HiFi reads from a genome (linear or circular)."""
    rng = np.random.default_rng(seed)
    genome_len = len(genome)
    n_reads = int(coverage * genome_len / mean_len)

    genome_circular = genome + genome

    reads = []
    for i in tqdm(range(n_reads), desc='Simulating reads', unit='read'):
        length = int(np.clip(
            rng.lognormal(np.log(mean_len), sd_len/mean_len),
            min(1000, mean_len//4),
            min(60_000, genome_len//2)
        ))

        start = rng.integers(0, genome_len)
        end   = start + length

        subseq = genome_circular[start : start + length]
        if len(subseq) < length:
            subseq = subseq + genome_circular[:length - len(subseq)]

        strand = rng.choice(['+', '-'])
        if strand == '-':
            subseq = reverse_complement(subseq)

        error_seq, quals = apply_hifi_errors(subseq, rng)

        reads.append({
            'id':          f'read_{i:06d}',
            'sequence':    error_seq,
            'quality':     quals,
            'true_start':  int(start % genome_len),
            'true_end':    int(end % genome_len),
            'true_length': length,
            'strand':      strand,
        })

    return reads


def write_fastq(reads: list, path: str):
    with open(path, 'w') as f:
        for r in reads:
            f.write(f'@{r["id"]}\n')
            f.write(r['sequence'] + '\n')
            f.write('+\n')
            f.write(''.join(chr(min(q,40)+33) for q in r['quality']) + '\n')


def compute_edge_labels(reads: list, genome_len: int,
                         min_overlap: int = 5000,
                         repeat_locs: list = None) -> dict:
    """
    Ground truth edge labels using interval-based spatial indexing.

    For E. coli scale (~9200 reads), brute-force O(n^2) = 42M pairs
    is slow but most pairs have zero overlap. We bin reads by genomic
    position and only check pairs in nearby bins.
    """
    repeat_locs = repeat_locs or []

    # Bin reads by start position (bin size = mean read length)
    bin_size = 15000
    n_bins = (genome_len // bin_size) + 2
    bins = [[] for _ in range(n_bins)]

    for i, r in enumerate(reads):
        start = r['true_start']
        end = r['true_end']
        if end < start:  # circular wraparound
            end += genome_len

        start_bin = start // bin_size
        end_bin = min(end // bin_size, n_bins - 1)

        for b in range(start_bin, end_bin + 1):
            bins[b % n_bins].append(i)

    # Check pairs within overlapping bins
    labels = {}
    checked = set()

    for bin_reads in tqdm(bins, desc='Computing labels', unit='bin'):
        for ii in range(len(bin_reads)):
            for jj in range(ii + 1, len(bin_reads)):
                i, j = min(bin_reads[ii], bin_reads[jj]), max(bin_reads[ii], bin_reads[jj])
                if (i, j) in checked:
                    continue
                checked.add((i, j))

                ri, rj = reads[i], reads[j]
                si, ei = ri['true_start'], ri['true_end']
                sj, ej = rj['true_start'], rj['true_end']

                # Handle circular wraparound
                if ei < si:
                    ei += genome_len
                if ej < sj:
                    ej += genome_len

                # Check both linear and wrapped positions
                overlap = max(0, min(ei, ej) - max(si, sj))

                # Also check circular case: one read wraps, other doesn't
                if overlap < min_overlap:
                    overlap2 = max(0, min(ei, ej + genome_len) - max(si, sj + genome_len))
                    overlap = max(overlap, overlap2)

                if overlap >= min_overlap:
                    labels[(i, j)] = 1.0
                else:
                    labels[(i, j)] = 0.0

    return labels


if __name__ == '__main__':
    from Bio import SeqIO
    import argparse

    ap = argparse.ArgumentParser(description="Simulate HiFi reads from a genome")
    ap.add_argument("--genome", help="Name from src/capo/genomes.py registry")
    ap.add_argument("--fasta", help="Path to FASTA (overrides --genome)")
    ap.add_argument("--coverage", type=int, default=None, help="Override coverage")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.fasta:
        fasta_path = args.fasta
        reads_out = 'data/reads.fastq'
        positions_out = 'data/read_positions.json'
        labels_out = 'data/edge_labels.json'
        meta_path = fasta_path.replace('.fasta', '_meta.json')
        coverage = args.coverage or 30
        mean_len, sd_len, min_ov = 15_000, 3_000, 5_000
    else:
        if not args.genome:
            ap.error("Pass --genome NAME or --fasta PATH")
        from capo.genomes import get
        cfg = get(args.genome)
        fasta_path = str(cfg.fasta)
        reads_out = str(cfg.reads)
        positions_out = str(cfg.read_positions)
        labels_out = str(cfg.edge_labels)
        meta_path = str(cfg.meta)
        coverage = args.coverage or cfg.coverage
        mean_len, sd_len, min_ov = cfg.mean_read_len, cfg.sd_read_len, cfg.min_overlap

    record = SeqIO.read(fasta_path, "fasta")
    genome = str(record.seq).upper()
    print(f'Loaded {len(genome):,} bp genome from {fasta_path}')

    reads = simulate_hifi_reads(genome, coverage=coverage,
                                 mean_len=mean_len, sd_len=sd_len,
                                 seed=args.seed)
    write_fastq(reads, reads_out)

    read_positions = [
        {'true_start': r['true_start'], 'true_end': r['true_end']}
        for r in reads
    ]
    json.dump(read_positions, open(positions_out, 'w'))

    meta = json.load(open(meta_path)) if Path(meta_path).exists() else {}
    repeat_locs = meta.get('repeat_locs', [])

    labels = compute_edge_labels(reads, len(genome),
                                  min_overlap=min_ov,
                                  repeat_locs=repeat_locs)

    json.dump({str(k): v for k, v in labels.items()},
              open(labels_out, 'w'))

    n_genuine = sum(1 for v in labels.values() if v > 0)
    print(f'Simulated {len(reads)} reads -> {reads_out}')
    print(f'Genuine overlaps: {n_genuine} / {len(labels)} pairs -> {labels_out}')
