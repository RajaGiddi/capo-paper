"""
CAPO as a calibrated scoring layer on top of minimap2 candidates.

Takes minimap2 ava-pb overlaps as the candidate set, computes CAPO
features (containment, chain_coverage, cosine_sim) per candidate pair,
fits the multi-feature Bayes-factor model from labels, and evaluates
P / R / ECE over the full minimap2 candidate set.

This is the paper's centerpiece experiment: rather than competing with
minimap2 on candidate generation, we layer CAPO on top of it to provide
what minimap2 lacks — calibrated probabilistic overlap confidences.

Usage:
    python -m capo.evaluation.capo_on_minimap --genome ecoli
    python -m capo.evaluation.capo_on_minimap --genome bsubtilis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Bio import SeqIO
from capo.genomes import get as get_genome
from capo.models.encoder import ReadEncoder, seq_to_tokens
from capo.models.features import (
    extract_minimiser_positions,
    compute_chain_features,
    compute_containment,
)
from capo.models.likelihood import (
    fit_multi_feature_model,
    compute_log_bayes_factor,
    print_model_summary,
)


MAX_TOK = 2048


# ── PAF → candidate pair set ─────────────────────────────────────────

def parse_paf_pairs(paf_path: Path,
                    read_id_to_idx: dict[str, int]) -> set[tuple[int, int]]:
    """Collect unique (i, j) read-index pairs (i < j) from a PAF file."""
    pairs: set[tuple[int, int]] = set()
    n_lines = 0
    n_dropped = 0
    with open(paf_path) as f:
        for line in f:
            n_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            q, t = fields[0], fields[5]
            if q == t:
                continue
            qi = read_id_to_idx.get(q)
            ti = read_id_to_idx.get(t)
            if qi is None or ti is None:
                n_dropped += 1
                continue
            pairs.add((min(qi, ti), max(qi, ti)))
    print(f"Parsed {n_lines:,} PAF lines → {len(pairs):,} unique pairs "
          f"({n_dropped} dropped for unknown IDs)")
    return pairs


# ── Per-read feature precomputation ──────────────────────────────────

def precompute_read_features(reads: list[dict],
                              encoder: ReadEncoder,
                              device: str,
                              batch_size: int = 32) -> dict:
    """Tokenise, extract minimiser sets/positions, and encode every read."""
    n = len(reads)
    tokens_all = []
    mins_sets = []
    min_positions = []
    read_lens = []

    print("Tokenising + minimiser extraction...")
    for r in tqdm(reads, unit="read"):
        toks = seq_to_tokens(r["sequence"])
        tokens_all.append(toks)
        read_lens.append(len(toks))
        # Minimiser set (w=5 windows)
        mset = set()
        w = 5
        for i in range(len(toks) - w + 1):
            mset.add(min(toks[i:i + w]))
        mins_sets.append(mset)
        min_positions.append(extract_minimiser_positions(toks))

    # Encoder pass — batch reads, pad to MAX_TOK
    print("Encoding reads...")
    mus = torch.zeros(n, 64)
    encoder.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), unit="batch"):
            batch = tokens_all[start:start + batch_size]
            maxlen = min(MAX_TOK, max(len(t) for t in batch))
            padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
            mask = torch.zeros(len(batch), maxlen, dtype=torch.bool)
            for i, toks in enumerate(batch):
                t = toks[:maxlen]
                padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)
                mask[i, :len(t)] = True
            padded = padded.to(device)
            mask = mask.to(device)
            mu, _ = encoder(padded, mask)
            mus[start:start + len(batch)] = mu.cpu()

    # L2-normalised mus for cosine similarity
    mus_norm = F.normalize(mus, dim=-1)

    return {
        "mins": mins_sets,
        "min_positions": min_positions,
        "read_lens": read_lens,
        "mus_norm": mus_norm,
    }


# ── Per-pair feature extraction ──────────────────────────────────────

def compute_pair_features(pairs: list[tuple[int, int]],
                           per_read: dict) -> dict[str, np.ndarray]:
    """Compute (containment, chain_coverage, cosine_sim) for every pair."""
    mins = per_read["mins"]
    mpos = per_read["min_positions"]
    rlen = per_read["read_lens"]
    mus_norm = per_read["mus_norm"]

    n = len(pairs)
    containment = np.zeros(n, dtype=np.float32)
    chain_cov = np.zeros(n, dtype=np.float32)
    cos_sim = np.zeros(n, dtype=np.float32)

    print("Computing per-pair features...")
    for k, (i, j) in enumerate(tqdm(pairs, unit="pair")):
        containment[k] = compute_containment(mins[i], mins[j])
        chain = compute_chain_features(mpos[i], mpos[j], rlen[i], rlen[j])
        chain_cov[k] = chain["chain_coverage"]
        cos_sim[k] = float(torch.dot(mus_norm[i], mus_norm[j]).item())

    return {
        "containment": containment,
        "chain_coverage": chain_cov,
        "cosine_sim": cos_sim,
    }


# ── BF fit + scoring ────────────────────────────────────────────────

def score_all(features: dict[str, np.ndarray],
              labels: np.ndarray,
              fit_frac: float = 0.3,
              seed: int = 0,
              disable_copula: bool = False) -> tuple[np.ndarray, object]:
    """Fit BF model on a random fit_frac of the pairs, score all."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    fit_idx = rng.choice(n, size=int(n * fit_frac), replace=False)
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[fit_idx] = True

    genuine_features = defaultdict(list)
    null_features = defaultdict(list)
    for k in fit_idx:
        target = genuine_features if labels[k] >= 0.5 else null_features
        for name, arr in features.items():
            target[name].append(float(arr[k]))

    n_g = sum(1 for k in fit_idx if labels[k] >= 0.5)
    pi = n_g / max(len(fit_idx), 1)
    print(f"\nFitting on {len(fit_idx):,} pairs ({n_g:,} genuine, pi={pi:.3f})")

    model = fit_multi_feature_model(genuine_features, null_features, pi)
    if disable_copula:
        model.copula_enabled = False
        print("  [ablation] copula correction DISABLED")
    print_model_summary(model)

    print("Scoring all pairs...")
    scores = np.zeros(n, dtype=np.float64)
    for k in tqdm(range(n), unit="pair"):
        feats = {name: float(arr[k]) for name, arr in features.items()}
        scores[k] = compute_log_bayes_factor(feats, model)
    return scores, model, fit_mask


# ── Metrics ─────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)


def threshold_sweep(scores: np.ndarray, labels: np.ndarray,
                    n_points: int = 200) -> list[dict]:
    uniq = np.unique(scores)
    if len(uniq) > n_points:
        uniq = np.quantile(scores, np.linspace(0, 1, n_points + 1))
    sweep = []
    y = (labels >= 0.5)
    for t in uniq:
        pos = scores > t
        tp = int((pos & y).sum())
        fp = int((pos & ~y).sum())
        fn = int((~pos & y).sum())
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        sweep.append({"threshold": float(t), "precision": float(P),
                      "recall": float(R), "f1": float(F1),
                      "tp": tp, "fp": fp, "fn": fn})
    return sweep


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CAPO scoring over minimap2 candidates")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--fit-frac", type=float, default=0.3,
                    help="Fraction of pairs used to fit the BF model (rest is eval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-copula", action="store_true",
                    help="Disable t-copula correction (ablation study)")
    args = ap.parse_args()

    cfg = get_genome(args.genome)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load reads + encoder + ground-truth labels
    print(f"Loading reads from {cfg.reads}")
    reads_raw = list(SeqIO.parse(str(cfg.reads), "fastq"))
    reads = [{"id": str(r.id), "sequence": str(r.seq)} for r in reads_raw]
    read_id_to_idx = {r["id"]: i for i, r in enumerate(reads)}
    print(f"  {len(reads):,} reads")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
              else "cpu")
    print(f"Device: {device}")

    encoder = ReadEncoder()
    encoder.load_state_dict(torch.load(cfg.encoder_ckpt, map_location="cpu"))
    encoder = encoder.to(device)

    raw_labels = json.load(open(cfg.edge_labels))
    true_labels = {eval(k): v for k, v in raw_labels.items()}

    # 2. Parse minimap2 PAF → candidate pair set
    paf_path = cfg.results_dir / "minimap2.paf"
    if not paf_path.exists():
        raise FileNotFoundError(
            f"Expected minimap2 PAF at {paf_path}. "
            f"Run evaluation/minimap_baseline.py first."
        )
    mm2_pairs_set = parse_paf_pairs(paf_path, read_id_to_idx)
    pairs = sorted(mm2_pairs_set)
    labels = np.array([1.0 if true_labels.get(p, 0.0) >= 0.5 else 0.0
                       for p in pairs], dtype=np.float32)
    n_g = int(labels.sum())
    print(f"Candidate set from mm2: {len(pairs):,} pairs, "
          f"{n_g:,} genuine ({n_g/len(pairs)*100:.1f}%)")

    # 3. Per-read precompute
    per_read = precompute_read_features(reads, encoder, device)

    # 4. Per-pair features
    features = compute_pair_features(pairs, per_read)

    # 5. Fit BF model + score
    log_bf, model, fit_mask = score_all(
        features, labels, fit_frac=args.fit_frac, seed=args.seed,
        disable_copula=args.no_copula,
    )
    probs = sigmoid(log_bf)
    eval_mask = ~fit_mask

    # 6. Metrics — report BOTH in-sample and held-out for honesty
    def block_metrics(mask: np.ndarray, tag: str,
                      probs_: np.ndarray = None,
                      scores_: np.ndarray = None) -> dict:
        probs_ = probs if probs_ is None else probs_
        scores_ = log_bf if scores_ is None else scores_
        p = probs_[mask]
        y = labels[mask]
        s = scores_[mask]
        tp = int(((p >= 0.5) & (y >= 0.5)).sum())
        fp = int(((p >= 0.5) & (y < 0.5)).sum())
        fn = int(((p < 0.5) & (y >= 0.5)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        ece = ece_score(p, y)
        sweep = threshold_sweep(s, y)
        best = max(sweep, key=lambda x: x["f1"])
        at_751 = None
        for row in sweep:
            if row["recall"] >= 0.751:
                if at_751 is None or row["precision"] > at_751["precision"]:
                    at_751 = row
        print(f"\n=== {tag} ({int(mask.sum()):,} pairs) ===")
        print(f"  @ t=0 (p=0.5): P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  ECE={ece:.4f}")
        print(f"  Best F1:       t={best['threshold']:+.2f}  "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}")
        if at_751:
            print(f"  @ R≥0.751:    t={at_751['threshold']:+.2f}  "
                  f"P={at_751['precision']:.3f}  R={at_751['recall']:.3f}  "
                  f"F1={at_751['f1']:.3f}")
        return {
            "n_pairs": int(mask.sum()),
            "n_genuine": int(y.sum()),
            "at_t0": {"precision": prec, "recall": rec, "f1": f1, "ece": ece,
                      "tp": tp, "fp": fp, "fn": fn},
            "best_f1": best,
            "at_recall_0.751": at_751,
            "sweep": sweep,
        }

    out = {
        "genome": cfg.name,
        "n_candidates": len(pairs),
        "n_genuine": n_g,
        "fit_frac": args.fit_frac,
        "model": {
            "logit_pi": model.logit_pi,
            "copula_enabled": model.copula_enabled,
            "copula_rho_genuine": model.copula_rho_genuine,
            "copula_rho_null": model.copula_rho_null,
        },
        "bf": {
            "in_sample": block_metrics(fit_mask, "BF IN-SAMPLE (fit split)"),
            "held_out":  block_metrics(eval_mask, "BF HELD-OUT (eval split)"),
            "all":       block_metrics(np.ones_like(labels, dtype=bool), "BF ALL pairs"),
        },
    }

    # 7. LogReg baseline on the same features + fit split
    from sklearn.linear_model import LogisticRegression
    feature_names = ["containment", "chain_coverage", "cosine_sim"]
    X = np.stack([features[n] for n in feature_names], axis=1)
    y_all = labels.astype(int)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X[fit_mask], y_all[fit_mask])
    lr_probs = lr.predict_proba(X)[:, 1]
    lr_scores = lr.decision_function(X)
    print(f"\nLogReg coefs: "
          f"{dict(zip(feature_names, [float(c) for c in lr.coef_[0]]))}")
    print(f"LogReg intercept: {float(lr.intercept_[0]):.3f}")

    out["logreg"] = {
        "coef": {n: float(c) for n, c in zip(feature_names, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
        "in_sample": block_metrics(fit_mask, "LogReg IN-SAMPLE",
                                   probs_=lr_probs, scores_=lr_scores),
        "held_out":  block_metrics(eval_mask, "LogReg HELD-OUT",
                                   probs_=lr_probs, scores_=lr_scores),
        "all":       block_metrics(np.ones_like(labels, dtype=bool),
                                   "LogReg ALL pairs",
                                   probs_=lr_probs, scores_=lr_scores),
    }

    suffix = "_no_copula" if args.no_copula else ""
    out_path = cfg.results_dir / f"capo_on_minimap{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
