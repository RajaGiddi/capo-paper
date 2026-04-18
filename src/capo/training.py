"""
CAPO training and inference pipeline for E. coli scale.

Frozen-encoder mode: contrastive-pretrained encoder provides cosine features,
all BF parameters fitted from calibration, no encoder gradient updates.

Multi-feature Bayes factor with chain coverage, end-anchoring, unique
containment, and copula correction.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from tqdm import tqdm
from collections import defaultdict

from capo.models.encoder import ReadEncoder, seq_to_tokens, collate_reads
from capo.models.graph_state import AssemblyGraph, NodeData, EdgeData
from capo.models.features import (
    extract_minimiser_positions, compute_all_features, compute_containment
)
from capo.models.likelihood import (
    MultiFeatureBFModel, fit_multi_feature_model,
    compute_log_bayes_factor, print_model_summary
)

K_CANDIDATES = 50

DEVICE = (
    'mps' if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else 'cpu'
)


def extract_minimisers(tokens: list[int], w: int = 5) -> set:
    mins = set()
    for i in range(len(tokens) - w + 1):
        mins.add(min(tokens[i:i+w]))
    return mins


def gc_content(seq: str) -> float:
    gc = seq.count('G') + seq.count('C')
    return gc / max(len(seq), 1)


def _kl_sym_ranking(mu_cpu, logvar_cpu, graph, cand_ids):
    """KL-symmetric distance ranking, returns top-K candidate ids."""
    cand_mus = torch.stack([graph.nodes[c].mu for c in cand_ids])
    cand_lvs = torch.stack([graph.nodes[c].logvar for c in cand_ids])

    var_new  = logvar_cpu.exp()
    var_cand = cand_lvs.exp()
    kl_sym   = 0.5 * (
        (var_new / var_cand + (cand_mus - mu_cpu).pow(2) / var_cand - 1 + cand_lvs - logvar_cpu).sum(-1) +
        (var_cand / var_new + (mu_cpu - cand_mus).pow(2) / var_new - 1 + logvar_cpu - cand_lvs).sum(-1)
    ) * 0.5

    k = min(K_CANDIDATES, len(cand_ids))
    top_k_idx = torch.topk(kl_sym, k, largest=False).indices.tolist()
    return [cand_ids[i] for i in top_k_idx]


def calibration_pass(reads, encoder, true_labels, device):
    """
    One no-grad pass to collect all feature values for genuine vs null pairs.
    Fits the multi-feature BF model.
    """
    graph = AssemblyGraph()
    genuine_features = defaultdict(list)
    null_features = defaultdict(list)
    n_genuine, n_total = 0, 0

    order = np.random.permutation(len(reads)).tolist()

    encoder.eval()
    with torch.no_grad():
        for read_id in tqdm(order, desc='Calibration', unit='read'):
            read = reads[read_id]
            seq = read['sequence']
            tokens = seq_to_tokens(seq)
            mins = extract_minimisers(tokens)
            min_positions = extract_minimiser_positions(tokens)
            gc_val = gc_content(seq)
            read_len_tokens = len(tokens)

            tok_t = torch.tensor(tokens[:2048], dtype=torch.long).unsqueeze(0).to(device)
            mu, logvar = encoder(tok_t)
            mu = mu.squeeze(0)
            mu_cpu = mu.detach().cpu()
            logvar_cpu = logvar.squeeze(0).detach().cpu()

            candidates = graph.get_candidates(mins, exclude=read_id)

            if not candidates:
                node = NodeData(mu=mu_cpu, logvar=logvar_cpu,
                                read_id=read['id'], gc=gc_val, minimisers=mins,
                                minimiser_positions=min_positions,
                                read_len=read_len_tokens)
                graph.add_node(read_id, node)
                continue

            cand_ids = list(candidates.keys())
            top_cands = _kl_sym_ranking(mu_cpu, logvar_cpu, graph, cand_ids)

            # Cosine similarities
            mu_norm = F.normalize(mu.unsqueeze(0), dim=-1)
            top_mus = torch.stack([graph.nodes[c].mu for c in top_cands]).to(device)
            top_mus_norm = F.normalize(top_mus, dim=-1)
            cos_sims = F.cosine_similarity(mu_norm, top_mus_norm, dim=1)

            for idx, cid in enumerate(top_cands):
                cand_node = graph.nodes[cid]
                cos = float(cos_sims[idx].item())

                # Compute all features
                feats = compute_all_features(
                    {'minimisers': mins, 'minimiser_positions': min_positions,
                     'read_len': read_len_tokens},
                    {'minimisers': cand_node.minimisers,
                     'minimiser_positions': cand_node.minimiser_positions,
                     'read_len': cand_node.read_len},
                    graph.kmer_doc_freq, graph.n_reads_seen,
                    cos_sim=cos,
                )

                key = (min(read_id, cid), max(read_id, cid))
                y = true_labels.get(key, 0.0)
                n_total += 1

                target = genuine_features if y > 0.5 else null_features
                if y > 0.5:
                    n_genuine += 1
                for fname, fval in feats.items():
                    target[fname].append(fval)

            node = NodeData(mu=mu_cpu, logvar=logvar_cpu,
                            read_id=read['id'], gc=gc_val, minimisers=mins,
                            minimiser_positions=min_positions,
                            read_len=read_len_tokens)
            graph.add_node(read_id, node)

    pi = n_genuine / max(n_total, 1)
    model = fit_multi_feature_model(genuine_features, null_features, pi)

    # Diagnostics
    print(f'\n  Calibration: {n_genuine}/{n_total} genuine (pi={pi:.3f})')
    print_model_summary(model)

    return model


def process_read(read: dict, read_id: int,
                  encoder: ReadEncoder,
                  graph: AssemblyGraph,
                  true_labels: dict,
                  bf_model: MultiFeatureBFModel,
                  device: str = 'cpu') -> dict:
    """
    Process one read: encode, find candidates, score edges, update graph.
    Frozen encoder — no gradient computation.
    """
    seq = read['sequence']
    tokens = seq_to_tokens(seq)
    mins = extract_minimisers(tokens)
    min_positions = extract_minimiser_positions(tokens)
    gc = gc_content(seq)
    read_len_tokens = len(tokens)

    tok_t = torch.tensor(tokens[:2048], dtype=torch.long).unsqueeze(0).to(device)
    mu, logvar = encoder(tok_t)
    mu, logvar = mu.squeeze(0), logvar.squeeze(0)

    mu_cpu = mu.detach().cpu()
    logvar_cpu = logvar.detach().cpu()

    candidates = graph.get_candidates(mins, exclude=read_id)

    if not candidates:
        node = NodeData(mu=mu_cpu, logvar=logvar_cpu,
                        read_id=read['id'], gc=gc, minimisers=mins,
                        minimiser_positions=min_positions,
                        read_len=read_len_tokens)
        graph.add_node(read_id, node)
        return {'n_proposed': 0}

    cand_ids = list(candidates.keys())
    top_cands = _kl_sym_ranking(mu_cpu, logvar_cpu, graph, cand_ids)

    # Cosine similarities
    mu_norm = F.normalize(mu.unsqueeze(0), dim=-1)
    top_mus = torch.stack([graph.nodes[c].mu for c in top_cands]).to(device)
    top_mus_norm = F.normalize(top_mus, dim=-1)
    cos_sims = F.cosine_similarity(mu_norm, top_mus_norm, dim=1)

    # Score each candidate
    for idx, cid in enumerate(top_cands):
        cand_node = graph.nodes[cid]
        cos = float(cos_sims[idx].item())

        feats = compute_all_features(
            {'minimisers': mins, 'minimiser_positions': min_positions,
             'read_len': read_len_tokens},
            {'minimisers': cand_node.minimisers,
             'minimiser_positions': cand_node.minimiser_positions,
             'read_len': cand_node.read_len},
            graph.kmer_doc_freq, graph.n_reads_seen,
            cos_sim=cos,
        )

        logit = compute_log_bayes_factor(feats, bf_model)
        confidence = float(1.0 / (1.0 + np.exp(-logit)))

        edge = EdgeData(
            confidence=confidence,
            log_bayes_factor=logit,
            containment=feats['containment'],
            chain_coverage=feats['chain_coverage'],
            chain_score=feats['chain_ratio'],
            end_anchoring=feats['end_anchoring'],
            unique_containment=feats['unique_containment'],
            cosine_sim=feats['cosine_sim'],
        )
        graph.add_edge(read_id, cid, edge)
        graph.coverage_depth[cid] = graph.coverage_depth.get(cid, 0) + 1

    # Update graph
    node = NodeData(mu=mu_cpu, logvar=logvar_cpu,
                    read_id=read['id'], gc=gc, minimisers=mins,
                    minimiser_positions=min_positions,
                    read_len=read_len_tokens)
    graph.add_node(read_id, node)

    graph.update_lambda(len(candidates) / 30.0)
    graph.update_gc(gc)

    return {'n_proposed': len(top_cands)}


def run_inference(reads, labels, encoder_ckpt, device=None):
    """
    Full inference pipeline: load encoder, calibrate, build graph.
    Returns (graph, bf_model, encoder).
    """
    if device is None:
        device = DEVICE
    print(f'Using device: {device}')

    encoder = ReadEncoder()
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location='cpu'))
    encoder = encoder.to(device)
    encoder.eval()

    print('Running calibration pass...')
    bf_model = calibration_pass(reads, encoder, labels, device)

    print('Building assembly graph...')
    graph = AssemblyGraph()
    with torch.no_grad():
        for i, read in enumerate(tqdm(reads, desc='Inference', unit='read')):
            process_read(read, i, encoder, graph, labels,
                         bf_model=bf_model, device=device)

    return graph, bf_model, encoder
