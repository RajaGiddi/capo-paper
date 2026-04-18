"""
Modal GPU deployment for contrastive pretraining.

Usage:
    modal run scripts/modal_pretrain.py                      # default: ecoli
    modal run scripts/modal_pretrain.py --genome bsubtilis
    modal run scripts/modal_pretrain.py --genome scerevisiae

Output: checkpoints/encoder_<genome>.pt downloaded locally.
"""
import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Image with dependencies ─────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "biopython",
        "tqdm",
        "scipy",
    )
)

# ── Persistent volume (shared; per-genome subdirs) ──────────────────
volume = modal.Volume.from_name("bawm-ecoli-vol", create_if_missing=True)

app = modal.App("bawm-pretrain", image=image)


# ── Upload data to volume (local function) ──────────────────────────
@app.local_entrypoint()
def main(genome: str = "ecoli"):
    """Upload local data, run GPU pretraining, download checkpoint."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from bawm.genomes import get as get_genome

    cfg = get_genome(genome)
    print(f"Genome: {cfg.organism} ({cfg.name})")

    reads_rel = f"/data/{cfg.name}/reads.fastq"
    labels_rel = f"/data/{cfg.name}/edge_labels.json"
    ckpt_rel = f"/checkpoints/encoder_{cfg.name}.pt"

    print("Uploading data to Modal volume...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(cfg.reads, reads_rel)
        batch.put_file(cfg.edge_labels, labels_rel)
        batch.put_file(PROJECT_ROOT / "src" / "bawm" / "models" / "encoder.py",
                       "/code/encoder.py")

    print("Starting GPU training...")
    train_on_gpu.remote(reads_rel, labels_rel, ckpt_rel)

    print("Downloading checkpoint...")
    ckpt_bytes = download_checkpoint.remote(ckpt_rel)

    cfg.encoder_ckpt.parent.mkdir(exist_ok=True)
    cfg.encoder_ckpt.write_bytes(ckpt_bytes)
    print(f"Saved checkpoint to {cfg.encoder_ckpt}")


# ── Training function on GPU ────────────────────────────────────────
@app.function(
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=60 * 60 * 6,  # 6 hour timeout
)
def train_on_gpu(reads_path: str = "/data/ecoli/reads.fastq",
                  labels_path: str = "/data/ecoli/edge_labels.json",
                  ckpt_path: str = "/checkpoints/encoder_ecoli.pt"):
    """Run contrastive pretraining on GPU. Paths are relative to /vol."""
    import os, sys
    sys.path.insert(0, "/vol/code")

    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from Bio import SeqIO
    from encoder import ReadEncoder, seq_to_tokens, collate_reads

    vol_reads = f"/vol{reads_path}"
    vol_labels = f"/vol{labels_path}"
    vol_ckpt = f"/vol{ckpt_path}"
    os.makedirs(os.path.dirname(vol_ckpt), exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")

    # ── Hyperparameters (tuned for GPU) ────────────────────────────
    LR         = 3e-4
    EPOCHS     = 30
    BATCH_SIZE = 16       # larger batch on GPU (was 4 on M2)
    N_NEG      = 16       # more negatives per anchor
    FREE_BITS  = 0.5
    MAX_TOK    = 1024
    LATENT_DIM = 64

    # ── Load data ──────────────────────────────────────────────────
    print("Loading reads...")
    reads_raw = list(SeqIO.parse(vol_reads, "fastq"))
    reads = [{"id": str(r.id), "sequence": str(r.seq)} for r in reads_raw]
    n = len(reads)
    print(f"  {n} reads loaded")

    raw_labels = json.load(open(vol_labels))
    labels = {eval(k): v for k, v in raw_labels.items()}

    # Build positive sets
    positives = {i: set() for i in range(n)}
    for (i, j), y in labels.items():
        if y >= 0.5:
            positives[i].add(j)
            positives[j].add(i)

    n_with_pos = sum(1 for p in positives.values() if len(p) > 0)
    avg_pos = np.mean([len(p) for p in positives.values()])
    print(f"  {n_with_pos}/{n} reads have >=1 positive, avg {avg_pos:.1f}/read")

    # Pre-tokenize
    print("Tokenising reads...")
    all_tokens = [seq_to_tokens(r["sequence"]) for r in tqdm(reads)]

    # ── Encoder setup ──────────────────────────────────────────────
    encoder = ReadEncoder().to(DEVICE)

    # Orthogonality kick
    with torch.no_grad():
        for p in encoder.mu_head.parameters():
            p.data += torch.randn_like(p.data) * 0.1

    opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    scale = 1.0 / np.sqrt(LATENT_DIM)

    def kl_loss(mu, logvar):
        kl = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
        return torch.clamp(kl - FREE_BITS, min=0).sum(-1).mean()

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(EPOCHS):
        # Temperature schedule
        if epoch < 10:
            tau = 0.5
        else:
            tau = 0.5 + (0.1 - 0.5) * (epoch - 10) / (EPOCHS - 10)

        encoder.train()
        anchor_order = np.random.permutation(n).tolist()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            range(0, n, BATCH_SIZE),
            desc=f"Epoch {epoch+1:2d}/{EPOCHS}",
            unit="batch",
        )
        for batch_start in pbar:
            anchor_ids = anchor_order[batch_start : batch_start + BATCH_SIZE]

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
            if len(valid_negs) > N_NEG:
                negs = np.random.choice(valid_negs, N_NEG, replace=False).tolist()
            else:
                negs = valid_negs
            batch_set.update(negs)

            batch_ids = sorted(batch_set)
            id_to_idx = {rid: idx for idx, rid in enumerate(batch_ids)}
            batch_tokens = [all_tokens[rid][:MAX_TOK] for rid in batch_ids]
            padded, mask = collate_reads(batch_tokens)
            padded, mask = padded.to(DEVICE), mask.to(DEVICE)

            mu, logvar = encoder(padded, mask)

            loss_terms = []
            for a in anchor_ids:
                if a not in pos_for_anchor:
                    continue
                p = pos_for_anchor[a]
                a_idx = id_to_idx[a]
                p_idx = id_to_idx[p]

                anchor_pos = positives[a]
                neg_idxs = [id_to_idx[ng] for ng in negs if ng not in anchor_pos]
                if not neg_idxs:
                    continue

                mu_a = mu[a_idx]
                mu_p = mu[p_idx]
                mu_n = mu[neg_idxs]

                pos_sim = torch.dot(mu_a, mu_p) * scale / tau
                neg_sim = (mu_n @ mu_a) * scale / tau

                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                target = torch.zeros(1, dtype=torch.long, device=DEVICE)
                loss_terms.append(F.cross_entropy(logits.unsqueeze(0), target))

            if not loss_terms:
                continue

            L_contrastive = torch.stack(loss_terms).mean()

            mu_norm = F.normalize(mu, dim=-1)
            sim_matrix = mu_norm @ mu_norm.T
            off_diag_mask = ~torch.eye(len(mu), dtype=torch.bool, device=DEVICE)
            L_spread = sim_matrix[off_diag_mask].abs().mean()

            L_kl = kl_loss(mu, logvar)

            loss = L_contrastive + 0.5 * L_spread + 0.01 * L_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(
                loss=f"{epoch_loss/max(n_batches,1):.4f}",
                tau=f"{tau:.3f}",
            )

        sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  => loss={avg_loss:.4f}  tau={tau:.3f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            torch.save(encoder.cpu().state_dict(), vol_ckpt)
            volume.commit()
            encoder = encoder.to(DEVICE)

    print("Training complete")
    torch.save(encoder.cpu().state_dict(), vol_ckpt)
    volume.commit()


@app.function(volumes={"/vol": volume})
def download_checkpoint(ckpt_path: str = "/checkpoints/encoder_ecoli.pt"):
    """Read checkpoint from volume and return bytes."""
    with open(f"/vol{ckpt_path}", "rb") as f:
        return f.read()
