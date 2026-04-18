import torch
import numpy as np
from dataclasses import dataclass, field

@dataclass
class NodeData:
    mu:         torch.Tensor   # (latent_dim,) — mean embedding
    logvar:     torch.Tensor   # (latent_dim,) — log variance
    read_id:    str
    gc:         float
    minimisers: set
    # Positional minimiser list for chaining (ordered by position in read)
    minimiser_positions: list = field(default_factory=list)
    read_len:   int = 0

@dataclass
class EdgeData:
    confidence:      float       # p_ij in (0,1) — posterior belief
    log_bayes_factor: float      # combined log BF before prior
    # Individual feature values (for diagnostics)
    containment:     float = 0.0
    chain_coverage:  float = 0.0
    chain_score:     float = 0.0
    end_anchoring:   float = 0.0
    unique_containment: float = 0.0
    cosine_sim:      float = 0.0

class AssemblyGraph:
    """Probabilistic assembly graph for E. coli scale."""

    def __init__(self):
        self.nodes: dict[int, NodeData]              = {}
        self.edges: dict[tuple[int,int], EdgeData]   = {}
        self.minimiser_index: dict[int, list[int]]   = {}

        # Global k-mer document frequency for TF-IDF / unique containment
        self.kmer_doc_freq: dict[int, int]            = {}
        self.n_reads_seen: int                        = 0

        # Prior state
        self.coverage_depth: dict[int, int]           = {}
        self.lambda_estimate: float                   = 30.0
        self.lambda_n:        int                     = 0
        self.gc_mean:         float                   = 0.5
        self.gc_n:            int                     = 0

    def add_node(self, node_id: int, node: NodeData):
        self.nodes[node_id] = node
        for m in node.minimisers:
            if m not in self.minimiser_index:
                self.minimiser_index[m] = []
            self.minimiser_index[m].append(node_id)

        # Update document frequency
        self.n_reads_seen += 1
        for m in node.minimisers:
            self.kmer_doc_freq[m] = self.kmer_doc_freq.get(m, 0) + 1

    def add_edge(self, i: int, j: int, edge: EdgeData):
        self.edges[(min(i,j), max(i,j))] = edge

    def get_candidates(self, minimisers: set,
                        exclude: int, min_shared: int = 2) -> dict:
        """Return {node_id: shared_count} for minimiser-filtered candidates."""
        counts = {}
        for m in minimisers:
            for nid in self.minimiser_index.get(m, []):
                if nid != exclude:
                    counts[nid] = counts.get(nid, 0) + 1
        return {k:v for k,v in counts.items() if v >= min_shared}

    def update_lambda(self, depth: float):
        self.lambda_n += 1
        self.lambda_estimate += (depth - self.lambda_estimate) / self.lambda_n

    def update_gc(self, gc: float):
        self.gc_n += 1
        self.gc_mean += (gc - self.gc_mean) / self.gc_n
