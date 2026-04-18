"""
Orthogonal feature extraction for overlap scoring.

Features (per candidate pair):
  1. Containment index J        — fraction of shared k-mers (baseline)
  2. Chain coverage             — fraction of read covered by best colinear chain
  3. Chain score ratio f_s/f_t  — secondary-to-primary chain ratio (repeat diagnostic)
  4. End-anchoring score        — dovetail geometry penalty
  5. Unique containment         — C_unique / C_all, downweighting repetitive k-mers
  6. Encoder cosine similarity  — from contrastive-pretrained encoder
"""
import numpy as np
from bawm.models.encoder import K, STRIDE


# ── Minimiser chaining ───────────────────────────────────────────────

def extract_minimiser_positions(tokens: list[int], w: int = 5) -> list[tuple[int, int]]:
    """
    Extract (position_in_read, token) pairs for minimisers.
    Position = index of the k-mer in the token sequence (not bp position).
    Returns sorted list of (pos, token).
    """
    positions = []
    for i in range(len(tokens) - w + 1):
        window = tokens[i:i+w]
        min_val = min(window)
        min_idx = i + window.index(min_val)
        positions.append((min_idx, min_val))

    # Deduplicate (same minimiser selected by adjacent windows)
    seen = set()
    deduped = []
    for pos, tok in positions:
        if (pos, tok) not in seen:
            seen.add((pos, tok))
            deduped.append((pos, tok))
    return sorted(deduped, key=lambda x: x[0])


def _chain_dp(matches: list[tuple[int, int]], max_gap: int = 50) -> list[list[tuple[int, int]]]:
    """
    Sparse dynamic programming chaining on (pos_i, pos_j) anchor pairs.
    Returns list of chains, each a list of (pos_i, pos_j) anchors.
    max_gap: maximum gap in either coordinate between consecutive anchors.

    Finds the longest colinear subsequence (both coordinates increasing,
    gap-constrained).
    """
    if not matches:
        return []

    # Sort by pos_i, then pos_j
    matches = sorted(matches, key=lambda x: (x[0], x[1]))
    n = len(matches)

    # DP: longest increasing subsequence in pos_j with gap constraint
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i - 1, max(-1, i - 200), -1):  # bounded lookback
            pi, pj = matches[i]
            qi, qj = matches[j]
            if qj < pj and (pi - qi) <= max_gap and (pj - qj) <= max_gap:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

    # Backtrack best chain
    best_end = max(range(n), key=lambda i: dp[i])
    chain1 = []
    idx = best_end
    while idx != -1:
        chain1.append(matches[idx])
        idx = parent[idx]
    chain1.reverse()

    # Find second-best chain (excluding anchors used in chain1)
    used = set(chain1)
    remaining = [m for m in matches if m not in used]

    chain2 = []
    if remaining:
        dp2 = [1] * len(remaining)
        parent2 = [-1] * len(remaining)
        for i in range(1, len(remaining)):
            for j in range(i - 1, max(-1, i - 200), -1):
                pi, pj = remaining[i]
                qi, qj = remaining[j]
                if qj < pj and (pi - qi) <= max_gap and (pj - qj) <= max_gap:
                    if dp2[j] + 1 > dp2[i]:
                        dp2[i] = dp2[j] + 1
                        parent2[i] = j
        best2 = max(range(len(remaining)), key=lambda i: dp2[i])
        idx = best2
        while idx != -1:
            chain2.append(remaining[idx])
            idx = parent2[idx]
        chain2.reverse()

    chains = [chain1]
    if chain2:
        chains.append(chain2)
    return chains


def compute_chain_features(pos_i: list[tuple[int, int]],
                            pos_j: list[tuple[int, int]],
                            read_len_i: int,
                            read_len_j: int) -> dict:
    """
    Compute chain coverage and chain score ratio for a read pair.

    pos_i, pos_j: list of (position_in_read, token) from extract_minimiser_positions
    read_len_i, read_len_j: read lengths in tokens

    Returns dict with:
        chain_coverage: fraction of shorter read spanned by primary chain
        chain_ratio:    |secondary chain| / |primary chain| (0 if no secondary)
    """
    # Build (pos_i, pos_j) anchor pairs from shared tokens
    tok_to_posj = {}
    for pos, tok in pos_j:
        if tok not in tok_to_posj:
            tok_to_posj[tok] = []
        tok_to_posj[tok].append(pos)

    matches = []
    for pos_a, tok in pos_i:
        if tok in tok_to_posj:
            for pos_b in tok_to_posj[tok]:
                matches.append((pos_a, pos_b))

    if not matches:
        return {'chain_coverage': 0.0, 'chain_ratio': 0.0}

    chains = _chain_dp(matches)

    if not chains or not chains[0]:
        return {'chain_coverage': 0.0, 'chain_ratio': 0.0}

    primary = chains[0]

    # Chain coverage: span of primary chain / shorter read length (in token positions)
    span_i = primary[-1][0] - primary[0][0] + 1
    span_j = primary[-1][1] - primary[0][1] + 1
    shorter_len = min(read_len_i, read_len_j)
    chain_coverage = max(span_i, span_j) / max(shorter_len, 1)
    chain_coverage = min(chain_coverage, 1.0)

    # Chain ratio: secondary / primary
    chain_ratio = 0.0
    if len(chains) > 1 and chains[1]:
        chain_ratio = len(chains[1]) / max(len(primary), 1)

    return {
        'chain_coverage': float(chain_coverage),
        'chain_ratio': float(chain_ratio),
    }


# ── End-anchoring score ──────────────────────────────────────────────

def compute_end_anchoring(pos_i: list[tuple[int, int]],
                           pos_j: list[tuple[int, int]],
                           read_len_i: int,
                           read_len_j: int) -> float:
    """
    End-anchoring score: dovetail geometry check.

    A genuine dovetail overlap should have the chain extending to within
    ~200bp of at least one end of each read. The score penalizes
    internal matches (both hangs > threshold).

    Returns: exp(-(hang_A + hang_B) / 200) where hang_A and hang_B are
    the minimum distances from the chain endpoints to the read ends.
    """
    tok_to_posj = {}
    for pos, tok in pos_j:
        if tok not in tok_to_posj:
            tok_to_posj[tok] = []
        tok_to_posj[tok].append(pos)

    matches = []
    for pos_a, tok in pos_i:
        if tok in tok_to_posj:
            for pos_b in tok_to_posj[tok]:
                matches.append((pos_a, pos_b))

    if not matches:
        return 0.0

    chains = _chain_dp(matches)
    if not chains or not chains[0]:
        return 0.0

    primary = chains[0]

    # Hangs: distance from chain endpoints to read endpoints (in token positions)
    # Convert token positions to approximate bp: pos * STRIDE
    start_i = primary[0][0]
    end_i = primary[-1][0]
    start_j = primary[0][1]
    end_j = primary[-1][1]

    # Hang for read i: min distance from chain start/end to read start/end
    hang_i = min(start_i, read_len_i - 1 - end_i)
    # Hang for read j: min distance from chain start/end to read start/end
    hang_j = min(start_j, read_len_j - 1 - end_j)

    # Convert to bp scale
    hang_bp = (hang_i + hang_j) * STRIDE
    return float(np.exp(-hang_bp / 200.0))


# ── Unique k-mer containment ────────────────────────────────────────

def compute_unique_containment(mins_i: set, mins_j: set,
                                kmer_doc_freq: dict,
                                n_reads: int,
                                coverage: float = 30.0) -> float:
    """
    Unique k-mer containment: fraction of shared k-mers that are
    globally rare (appear in <= 2x expected haploid coverage reads).

    C_unique = |shared k-mers with doc_freq <= threshold| / |shared k-mers|

    Rare k-mers are informative because they come from unique genomic
    regions. Shared repetitive k-mers are uninformative for overlap
    identity (they match reads from different genomic copies).
    """
    shared = mins_i & mins_j
    if not shared:
        return 0.0

    # Threshold: k-mers appearing in <= 2x expected coverage are "unique"
    threshold = max(int(2 * coverage), 4)

    n_unique = sum(1 for m in shared
                   if kmer_doc_freq.get(m, 0) <= threshold)

    return float(n_unique / len(shared))


# ── Containment index (baseline) ────────────────────────────────────

def compute_containment(mins_i: set, mins_j: set) -> float:
    min_size = min(len(mins_i), len(mins_j))
    if min_size == 0:
        return 0.0
    return len(mins_i & mins_j) / min_size


# ── Full feature vector ─────────────────────────────────────────────

def compute_all_features(node_i_data: dict, node_j_data: dict,
                          kmer_doc_freq: dict, n_reads: int,
                          cos_sim: float = 0.0) -> dict:
    """
    Compute all 6 features for a candidate pair.

    node_i_data, node_j_data: dicts with keys
        minimisers, minimiser_positions, read_len (in tokens)
    """
    mins_i = node_i_data['minimisers']
    mins_j = node_j_data['minimisers']
    pos_i = node_i_data['minimiser_positions']
    pos_j = node_j_data['minimiser_positions']
    len_i = node_i_data['read_len']
    len_j = node_j_data['read_len']

    containment = compute_containment(mins_i, mins_j)

    chain = compute_chain_features(pos_i, pos_j, len_i, len_j)

    end_anchor = compute_end_anchoring(pos_i, pos_j, len_i, len_j)

    unique_cont = compute_unique_containment(
        mins_i, mins_j, kmer_doc_freq, n_reads)

    return {
        'containment': containment,
        'chain_coverage': chain['chain_coverage'],
        'chain_ratio': chain['chain_ratio'],
        'end_anchoring': end_anchor,
        'unique_containment': unique_cont,
        'cosine_sim': cos_sim,
    }
