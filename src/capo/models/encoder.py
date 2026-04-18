import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

K        = 11      # k-mer length
STRIDE   = 5       # sliding window stride
VOCAB    = 16384   # hash vocabulary
LATENT   = 64      # latent dimension

COMPLEMENT_MAP = {'A':'T','T':'A','C':'G','G':'C','N':'N'}

def rc(seq: str) -> str:
    return ''.join(COMPLEMENT_MAP.get(b,'N') for b in reversed(seq))

def hash_kmer(kmer: str) -> int:
    h = 0
    for c in kmer:
        h = (h * 5 + ord(c)) & 0x3FFF
    return h

def seq_to_tokens(seq: str) -> list[int]:
    seq = seq.upper()
    tokens = []
    for i in range(0, len(seq) - K + 1, STRIDE):
        kmer     = seq[i:i+K]
        kmer_rc  = rc(kmer)
        canonical = min(kmer, kmer_rc)
        tokens.append(hash_kmer(canonical))
    return tokens


class ReadEncoder(nn.Module):
    """
    Lightweight stochastic read encoder.
    Input:  token_ids (B, T) — canonical k-mer hashes
    Output: mu (B, LATENT), logvar (B, LATENT)
    """
    def __init__(self, vocab=VOCAB, d_model=64, n_heads=4,
                 n_layers=3, d_ffn=256, latent=LATENT, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embed   = nn.Embedding(vocab + 1, d_model, padding_idx=0)
        self.cls     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_bias = nn.Embedding(2048, n_heads)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ffn, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mu_head     = nn.Linear(d_model, latent)
        self.logvar_head = nn.Linear(d_model, latent)

        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, token_ids, padding_mask=None):
        B, T = token_ids.shape
        x   = self.embed(token_ids)
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x      = self.transformer(x, src_key_padding_mask=padding_mask)
        cls_out = x[:, 0, :]

        mu     = self.mu_head(cls_out)
        logvar = torch.clamp(self.logvar_head(cls_out), -8, 2)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu


def collate_reads(reads: list[list[int]], max_len: int = 2048):
    tokens = [torch.tensor(r[:max_len], dtype=torch.long) for r in reads]
    lengths = [len(t) for t in tokens]
    padded  = torch.zeros(len(tokens), max_len, dtype=torch.long)
    mask    = torch.ones(len(tokens), max_len, dtype=torch.bool)
    for i, (t, l) in enumerate(zip(tokens, lengths)):
        padded[i, :l] = t
        mask[i, :l]   = False
    return padded, mask
