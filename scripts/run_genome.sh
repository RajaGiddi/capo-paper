#!/bin/bash
# run_genome.sh — BAWM pipeline for any registered genome
#
# Usage: bash scripts/run_genome.sh <genome_name> [stage]
#
#   genome_name: toy | ecoli | bsubtilis | scerevisiae  (see src/bawm/genomes.py)
#   stage:       eda | simulate | pretrain | infer | eval | all  (default: all)
#
# Examples:
#   bash scripts/run_genome.sh ecoli all
#   bash scripts/run_genome.sh toy simulate
#
# Run from the repo root. Requires `pip install -e .` (so `python -m bawm.*` works).
#
# Prerequisites:
#   - The genome's FASTA at data/genes/<name>/genome.fasta
#   - For pretrain stage: Modal CLI configured (modal token set)

set -e

GENOME="${1:-ecoli}"
STAGE="${2:-all}"

echo '═══════════════════════════════════════'
echo " BAWM pipeline — $GENOME ($STAGE)"
echo '═══════════════════════════════════════'

mkdir -p checkpoints "results/$GENOME"

run_eda() {
    echo ''
    echo '[EDA] Genome analysis...'
    python -m bawm.eda --genome "$GENOME"
}

run_simulate() {
    echo ''
    echo '[Simulate] HiFi reads...'
    python -m bawm.simulator --genome "$GENOME"
}

run_pretrain() {
    echo ''
    echo '[Pretrain] Contrastive encoder on GPU...'
    modal run scripts/modal_pretrain.py --genome "$GENOME"
}

run_infer() {
    echo ''
    echo '[Infer] Calibration + graph building...'
    python -c "
import json
from Bio import SeqIO
from bawm.genomes import get
from bawm.training import run_inference

cfg = get('$GENOME')
reads_raw = list(SeqIO.parse(str(cfg.reads), 'fastq'))
reads = [{'id': str(r.id), 'sequence': str(r.seq)} for r in reads_raw]
raw_labels = json.load(open(cfg.edge_labels))
labels = {eval(k): v for k, v in raw_labels.items()}

graph, bf_model, encoder = run_inference(
    reads, labels, encoder_ckpt=str(cfg.encoder_ckpt)
)

cfg.results_dir.mkdir(parents=True, exist_ok=True)
edges_out = {}
for (i,j), e in graph.edges.items():
    edges_out[f'({i},{j})'] = {
        'confidence': e.confidence, 'log_bf': e.log_bayes_factor,
        'containment': e.containment, 'chain_coverage': e.chain_coverage,
        'chain_score': e.chain_score, 'end_anchoring': e.end_anchoring,
        'unique_containment': e.unique_containment, 'cosine_sim': e.cosine_sim,
    }
json.dump(edges_out, open(cfg.results_dir / 'edges.json', 'w'), indent=2)
print(f'Saved {len(edges_out)} edges to {cfg.results_dir}/edges.json')
"
}

run_eval() {
    echo ''
    echo '[Eval] Metrics + calibration...'
    python -c "
import json
from bawm.genomes import get
from bawm.models.graph_state import AssemblyGraph, EdgeData
from bawm.evaluation.metrics import (compute_edge_metrics, threshold_sweep,
                                      feature_ablation, plot_calibration)

cfg = get('$GENOME')
raw_labels = json.load(open(cfg.edge_labels))
labels = {eval(k): v for k, v in raw_labels.items()}

edges_data = json.load(open(cfg.results_dir / 'edges.json'))
graph = AssemblyGraph()
for key_str, edata in edges_data.items():
    i, j = eval(key_str)
    graph.add_edge(i, j, EdgeData(
        confidence=edata['confidence'], log_bayes_factor=edata['log_bf'],
        containment=edata['containment'], chain_coverage=edata['chain_coverage'],
        chain_score=edata['chain_score'], end_anchoring=edata['end_anchoring'],
        unique_containment=edata['unique_containment'], cosine_sim=edata['cosine_sim']))

read_positions = json.load(open(cfg.read_positions))
meta = json.load(open(cfg.meta))
repeat_locs = meta.get('repeat_locs', [])

metrics = compute_edge_metrics(graph, labels, read_positions, repeat_locs)
json.dump(metrics, open(cfg.results_dir / 'metrics.json', 'w'), indent=2)
print(json.dumps(metrics, indent=2))

sweep = threshold_sweep(graph, labels)
json.dump(sweep, open(cfg.results_dir / 'threshold_sweep.json', 'w'), indent=2)

ablation = feature_ablation(graph, labels)
json.dump(ablation, open(cfg.results_dir / 'feature_ablation.json', 'w'), indent=2)
print()
print('=== Feature Ablation ===')
for fname, stats in ablation.items():
    print(f'  {fname:25s} genuine={stats[\"genuine_mean\"]:.3f} null={stats[\"null_mean\"]:.3f} sep={stats[\"separation\"]:.3f}')

plot_calibration(graph, labels, str(cfg.results_dir / 'calibration.png'))
"
}

case "$STAGE" in
    eda)      run_eda ;;
    simulate) run_simulate ;;
    pretrain) run_pretrain ;;
    infer)    run_infer ;;
    eval)     run_eval ;;
    all)
        run_eda
        run_simulate
        run_pretrain
        run_infer
        run_eval
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid: eda | simulate | pretrain | infer | eval | all"
        exit 1
        ;;
esac

echo ''
echo "═══════════════════════════════════════"
echo " $GENOME ($STAGE) complete"
echo " Results: results/$GENOME/"
echo "═══════════════════════════════════════"
