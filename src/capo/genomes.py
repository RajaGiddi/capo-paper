"""
Genome registry and path helpers.

One source of truth for everything per-genome: fasta location, known repeat
annotations, simulation parameters, and where pipeline artefacts live.

Adding a new genome:
    1. Drop or symlink the FASTA into data/genes/<name>/genome.fasta
    2. Add an entry to GENOMES below (at minimum: accession, organism, topology)
    3. Run the pipeline with --genome <name>
"""
from pathlib import Path
from dataclasses import dataclass, field


# src/capo/genomes.py → ../../.. = repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENES_ROOT = REPO_ROOT / "data" / "genes"


@dataclass
class GenomeConfig:
    name: str
    accession: str
    organism: str
    topology: str = "circular"          # "circular" or "linear"
    coverage: int = 30                  # simulated HiFi coverage
    mean_read_len: int = 15_000
    sd_read_len: int = 3_000
    min_overlap: int = 5_000            # for ground-truth labeling
    known_repeats: list = field(default_factory=list)

    # ── Derived paths ────────────────────────────────────────────────
    @property
    def dir(self) -> Path:
        return GENES_ROOT / self.name

    @property
    def fasta(self) -> Path:
        return self.dir / "genome.fasta"

    @property
    def meta(self) -> Path:
        return self.dir / "meta.json"

    @property
    def reads(self) -> Path:
        return self.dir / "reads.fastq"

    @property
    def edge_labels(self) -> Path:
        return self.dir / "edge_labels.json"

    @property
    def read_positions(self) -> Path:
        return self.dir / "read_positions.json"

    @property
    def encoder_ckpt(self) -> Path:
        return REPO_ROOT / "checkpoints" / f"encoder_{self.name}.pt"

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "results" / self.name


# ── Registered genomes ───────────────────────────────────────────────

GENOMES = {
    "toy": GenomeConfig(
        name="toy",
        accession="synthetic",
        organism="Synthetic toy genome",
        topology="linear",
        coverage=30,
        mean_read_len=8_000,
        sd_read_len=1_500,
        min_overlap=2_000,
        known_repeats=[],
    ),
    "ecoli": GenomeConfig(
        name="ecoli",
        accession="NC_000913.3",
        organism="Escherichia coli K-12 MG1655",
        topology="circular",
        known_repeats=[
            {"start": 223771,  "end": 229357,  "name": "rrnH", "type": "rRNA_operon"},
            {"start": 2726188, "end": 2731597, "name": "rrnE", "type": "rRNA_operon"},
            {"start": 3427221, "end": 3432653, "name": "rrnD", "type": "rRNA_operon"},
            {"start": 3941599, "end": 3947050, "name": "rrnC", "type": "rRNA_operon"},
            {"start": 4035531, "end": 4040996, "name": "rrnA", "type": "rRNA_operon"},
            {"start": 4166659, "end": 4172117, "name": "rrnB", "type": "rRNA_operon"},
            {"start": 4208147, "end": 4213567, "name": "rrnG", "type": "rRNA_operon"},
        ],
    ),
    "bsubtilis": GenomeConfig(
        name="bsubtilis",
        accession="NC_000964.3",
        organism="Bacillus subtilis 168",
        topology="circular",
        # 10 rRNA operons (rrnA..rrnJ) on reference chromosome
        known_repeats=[
            {"start":    9810, "end":   15428, "name": "rrnO", "type": "rRNA_operon"},
            {"start":   94998, "end":  100560, "name": "rrnA", "type": "rRNA_operon"},
            {"start":  171278, "end":  176840, "name": "rrnJ", "type": "rRNA_operon"},
            {"start":  627829, "end":  633396, "name": "rrnW", "type": "rRNA_operon"},
            {"start":  649839, "end":  655401, "name": "rrnI", "type": "rRNA_operon"},
            {"start":  661796, "end":  667359, "name": "rrnH", "type": "rRNA_operon"},
            {"start":  680528, "end":  686097, "name": "rrnG", "type": "rRNA_operon"},
            {"start": 3211832, "end": 3217397, "name": "rrnE", "type": "rRNA_operon"},
            {"start": 3641478, "end": 3647037, "name": "rrnD", "type": "rRNA_operon"},
            {"start": 4154577, "end": 4160141, "name": "rrnB", "type": "rRNA_operon"},
        ],
    ),
    "scerevisiae": GenomeConfig(
        name="scerevisiae",
        accession="GCF_000146045.2",
        organism="Saccharomyces cerevisiae S288C",
        topology="linear",                  # 16 linear chromosomes + mitochondrion
        coverage=30,
        # rDNA tandem repeat on chromosome XII (~100-200 copies of ~9.1 kb unit)
        # Not a simple interval list; populate after EDA
        known_repeats=[],
    ),
}


def get(name: str) -> GenomeConfig:
    if name not in GENOMES:
        raise KeyError(
            f"Unknown genome '{name}'. Registered: {sorted(GENOMES.keys())}"
        )
    return GENOMES[name]


def list_genomes() -> list[str]:
    return sorted(GENOMES.keys())
