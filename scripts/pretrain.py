"""
Contrastive pretraining for the CAPO read encoder.

Device auto-selection: CUDA > MPS (Apple Silicon) > CPU. Runs locally with
whatever hardware the user has; no cloud dependency.

Usage:
    python scripts/pretrain.py --genome ecoli
    python scripts/pretrain.py --genome bsubtilis --epochs 30
    python scripts/pretrain.py --genome toy --batch-size 4

Output: checkpoints/encoder_<genome>.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm

from capo.genomes import get as get_genome
from capo.models.encoder import ReadEncoder, seq_to_tokens, collate_reads


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def train(genome: str,
          epochs: int = 30,
          batch_size: int = 16,
          n_neg: int = 16,
          lr: float = 3e-4,
          free_bits: float = 0.5,
          max_tok: int = 1024,
          latent_dim: int = 64,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = get_genome(genome)
    device = pick_device()
    print(f"Genome: {cfg.organism} ({cfg.name})")
    print(f"Device: {describe_device(device)}")
    if device.type == "cpu":
        print("  (CPU fallback; expect training to be much slower. "
              "Consider --epochs 5 for a quick sanity run.)")

    # ── Load reads and edge labels ──────────────────────────────────
    print(f"Loading reads from {cfg.reads}")
    reads_raw = list(SeqIO.parse(str(cfg.reads), "fastq"))
    reads = [{"id": str(r.id), "sequence": str(r.seq)} for r in reads_raw]
    n = len(reads)
    print(f"  {n} reads loaded")

    raw_labels = json.load(open(cfg.edge_labels))
    # ast.literal_eval avoids executing arbitrary Python from JSON keys
    import ast
    labels = {ast.literal_eval(k): v for k, v in raw_labels.items()}

    positives = {i: set() for i in range(n)}
    for (i, j), y in labels.items():
        if y >= 0.5:
            positives[i].add(j)
            positives[j].add(i)

    n_with_pos = sum(1 for p in positives.values() if len(p) > 0)
    avg_pos = np.mean([len(p) for p in positives.values()])
    print(f"  {n_with_pos}/{n} reads have >=1 positive, avg {avg_pos:.1f}/read")

    print("Tokenising reads...")
    all_tokens = [seq_to_tokens(r["sequence"]) for r in tqdm(reads)]

    # ── Encoder setup ────────────────────────────────────────────────
    encoder = ReadEncoder().to(device)

    # Orthogonality kick on the readout (preserves anti-collapse behaviour)
    with torch.no_grad():
        for p in encoder.mu_head.parameters():
            p.data += torch.randn_like(p.data) * 0.1

    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scale = 1.0 / np.sqrt(latent_dim)

    def kl_loss(mu, logvar):
        kl = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
        return torch.clamp(kl - free_bits, min=0).sum(-1).mean()

    cfg.encoder_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(epochs):
        if epoch < 10:
            tau = 0.5
        else:
            tau = 0.5 + (0.1 - 0.5) * (epoch - 10) / max(epochs - 10, 1)

        encoder.train()
        anchor_order = np.random.permutation(n).tolist()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(0, n, batch_size),
                    desc=f"Epoch {epoch+1:2d}/{epochs}", unit="batch")
        for batch_start in pbar:
            anchor_ids = anchor_order[batch_start : batch_start + batch_size]

            batch_set = set(anchor_ids)
            pos_for_anchor = {}
            for a in anchor_ids:
                pos_cand = list(positives[a])
                if not pos_cand:
                    continue
                p = int(np.random.choice(pos_cand))
                pos_for_anchor[a] = p
                batch_set.add(p)

            if not pos_for_anchor:
                continue

            valid_negs = [k for k in range(n) if k not in batch_set]
            if len(valid_negs) > n_neg:
                negs = np.random.choice(valid_negs, n_neg,
                                         replace=False).tolist()
            else:
                negs = valid_negs
            batch_set.update(negs)

            batch_ids = sorted(batch_set)
            id_to_idx = {rid: idx for idx, rid in enumerate(batch_ids)}
            batch_tokens = [all_tokens[rid][:max_tok] for rid in batch_ids]
            padded, mask = collate_reads(batch_tokens)
            padded, mask = padded.to(device), mask.to(device)

            mu, logvar = encoder(padded, mask)

            loss_terms = []
            for a in anchor_ids:
                if a not in pos_for_anchor:
                    continue
                p = pos_for_anchor[a]
                a_idx = id_to_idx[a]
                p_idx = id_to_idx[p]

                anchor_pos = positives[a]
                neg_idxs = [id_to_idx[ng] for ng in negs
                            if ng not in anchor_pos]
                if not neg_idxs:
                    continue

                mu_a = mu[a_idx]
                mu_p = mu[p_idx]
                mu_n = mu[neg_idxs]

                pos_sim = torch.dot(mu_a, mu_p) * scale / tau
                neg_sim = (mu_n @ mu_a) * scale / tau

                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                target = torch.zeros(1, dtype=torch.long, device=device)
                loss_terms.append(F.cross_entropy(logits.unsqueeze(0), target))

            if not loss_terms:
                continue

            L_contrastive = torch.stack(loss_terms).mean()

            mu_norm = F.normalize(mu, dim=-1)
            sim_matrix = mu_norm @ mu_norm.T
            off_diag_mask = ~torch.eye(len(mu), dtype=torch.bool, device=device)
            L_spread = sim_matrix[off_diag_mask].abs().mean()

            L_kl = kl_loss(mu, logvar)

            loss = L_contrastive + 0.5 * L_spread + 0.01 * L_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss/max(n_batches,1):.4f}",
                             tau=f"{tau:.3f}")

        sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        print(f"  => loss={avg_loss:.4f}  tau={tau:.3f}  elapsed={elapsed/60:.1f} min")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(encoder.cpu().state_dict(), cfg.encoder_ckpt)
            encoder = encoder.to(device)
            print(f"  saved checkpoint to {cfg.encoder_ckpt}")

    print(f"\nTraining complete. Final checkpoint: {cfg.encoder_ckpt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", required=True,
                    help="Registered genome name (toy, ecoli, bsubtilis, ...)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-neg", type=int, default=16,
                    help="Number of negatives per anchor")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train(genome=args.genome, epochs=args.epochs,
          batch_size=args.batch_size, n_neg=args.n_neg,
          lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
