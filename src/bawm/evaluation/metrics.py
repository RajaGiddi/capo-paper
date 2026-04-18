"""
Evaluation metrics for the assembly graph.

Computes precision, recall, ECE, per-feature ablation, and
rRNA operon-specific analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from bawm.models.graph_state import AssemblyGraph


def compute_edge_metrics(graph: AssemblyGraph,
                          true_labels: dict,
                          read_positions: list = None,
                          repeat_locs: list = None) -> dict:
    """Precision, recall, ECE, and per-feature diagnostics."""
    p_pred, y_true = [], []

    for (i, j), edge in graph.edges.items():
        key = (min(i, j), max(i, j))
        y = true_labels.get(key, 0.0)
        p_pred.append(edge.confidence)
        y_true.append(y)

    p_pred = np.array(p_pred)
    y_true = np.array(y_true)

    tp = ((p_pred >= 0.5) & (y_true >= 0.5)).sum()
    fp = ((p_pred >= 0.5) & (y_true < 0.5)).sum()
    fn = ((p_pred < 0.5) & (y_true >= 0.5)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(p_pred)) * abs(conf - acc)

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'ece': float(ece),
        'n_edges': len(p_pred),
        'n_genuine': int((y_true >= 0.5).sum()),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }

    # Repeat/rRNA operon analysis
    if read_positions is not None and repeat_locs is not None:
        repeat_tp, repeat_fp, repeat_fn = 0, 0, 0
        repeat_confs = []

        for (i, j), edge in graph.edges.items():
            si = read_positions[i]['true_start']
            sj = read_positions[j]['true_start']
            in_repeat = False
            for rloc in repeat_locs:
                rs, re = rloc[0], rloc[1]
                if rs <= si < re or rs <= sj < re:
                    in_repeat = True
                    break

            if in_repeat:
                repeat_confs.append(edge.confidence)
                key = (min(i, j), max(i, j))
                y = true_labels.get(key, 0.0)
                if edge.confidence >= 0.5 and y >= 0.5:
                    repeat_tp += 1
                elif edge.confidence >= 0.5 and y < 0.5:
                    repeat_fp += 1
                elif edge.confidence < 0.5 and y >= 0.5:
                    repeat_fn += 1

        metrics['repeat_n_edges'] = len(repeat_confs)
        metrics['repeat_conf_mean'] = float(np.mean(repeat_confs)) if repeat_confs else 0.0
        metrics['repeat_precision'] = repeat_tp / max(repeat_tp + repeat_fp, 1)
        metrics['repeat_recall'] = repeat_tp / max(repeat_tp + repeat_fn, 1)

    # Logit distribution
    logits = np.array([np.log(e.confidence / (1 - e.confidence + 1e-12) + 1e-12)
                       for e in graph.edges.values()])
    y_arr = np.array([true_labels.get((min(i,j),max(i,j)), 0.0) >= 0.5
                      for (i,j) in graph.edges.keys()])
    if y_arr.sum() > 0:
        metrics['genuine_logit_mean'] = float(logits[y_arr].mean())
        metrics['genuine_logit_std'] = float(logits[y_arr].std())
    if (~y_arr).sum() > 0:
        metrics['null_logit_mean'] = float(logits[~y_arr].mean())
        metrics['null_logit_std'] = float(logits[~y_arr].std())

    return metrics


def threshold_sweep(graph: AssemblyGraph, true_labels: dict,
                     thresholds=None) -> list[dict]:
    """Precision/recall at multiple logit thresholds."""
    logits, ys = [], []
    for (i, j), edge in graph.edges.items():
        key = (min(i, j), max(i, j))
        c = edge.confidence
        logits.append(np.log(c / (1 - c + 1e-12) + 1e-12))
        ys.append(true_labels.get(key, 0.0) >= 0.5)

    logits = np.array(logits)
    ys = np.array(ys)

    if thresholds is None:
        thresholds = np.arange(-2.0, 4.25, 0.25)

    results = []
    for t in thresholds:
        tp = ((logits > t) & ys).sum()
        fp = ((logits > t) & ~ys).sum()
        fn = ((logits <= t) & ys).sum()
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-6)
        results.append({'threshold': float(t), 'precision': float(P),
                        'recall': float(R), 'f1': float(F1)})
    return results


def feature_ablation(graph: AssemblyGraph, true_labels: dict) -> dict:
    """Per-feature contribution analysis from stored edge features."""
    feature_names = ['containment', 'chain_coverage', 'chain_score',
                     'end_anchoring', 'unique_containment', 'cosine_sim']

    ablation = {}
    for fname in feature_names:
        genuine_vals = []
        null_vals = []
        for (i, j), edge in graph.edges.items():
            key = (min(i, j), max(i, j))
            y = true_labels.get(key, 0.0)
            val = getattr(edge, fname, 0.0)
            if y >= 0.5:
                genuine_vals.append(val)
            else:
                null_vals.append(val)

        g = np.array(genuine_vals) if genuine_vals else np.array([0.0])
        n = np.array(null_vals) if null_vals else np.array([0.0])
        ablation[fname] = {
            'genuine_mean': float(g.mean()),
            'genuine_std': float(g.std()),
            'null_mean': float(n.mean()),
            'null_std': float(n.std()),
            'separation': float(g.mean() - n.mean()),
        }

    return ablation


def plot_calibration(graph: AssemblyGraph, true_labels: dict,
                      save_path: str = 'results/calibration.png'):
    """Reliability diagram and confidence histogram."""
    p_pred, y_true = [], []
    for (i, j), edge in graph.edges.items():
        key = (min(i, j), max(i, j))
        p_pred.append(edge.confidence)
        y_true.append(true_labels.get(key, 0.0))

    p_pred = np.array(p_pred)
    y_true = np.array(y_true)
    bins = np.linspace(0, 1, 11)
    bin_mids, accs, confs, ns = [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() < 2:
            continue
        bin_mids.append((lo + hi) / 2)
        accs.append(y_true[mask].mean())
        confs.append(p_pred[mask].mean())
        ns.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax1.scatter(confs, accs, s=[n * 2 for n in ns], alpha=0.8, color='#2E75B6')
    ax1.set_xlabel('Mean predicted confidence')
    ax1.set_ylabel('Fraction of genuine overlaps')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    genuine = p_pred[y_true >= 0.5]
    false_p = p_pred[y_true < 0.5]
    if len(genuine) > 0:
        ax2.hist(genuine, bins=20, alpha=0.6, color='#1E6B3C', label='Genuine overlaps')
    if len(false_p) > 0:
        ax2.hist(false_p, bins=20, alpha=0.6, color='#8B1A1A', label='Null pairs')
    ax2.set_xlabel('Predicted confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution by Label')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved calibration plot to {save_path}')
    return fig
