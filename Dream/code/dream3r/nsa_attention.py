"""
Native Sparse Attention (NSA) — 3-branch selective attention for Dream3R.

Three branches operate in parallel on the same query:
  1. Compressed: attends to n_compress latent state tokens (long-term context)
  2. Selected:   top-k lookup from an external bank (spatial recall)
  3. Sliding:    local window of recent tokens (short-term continuity)

Outputs are fused via a learned per-query gate that weights the three branches.

Reference: DeepSeek NSA (2025), adapted for 3D reconstruction streaming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NSABranch(nn.Module):
    """Single multi-head attention branch used by all three NSA paths."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [B, Q, D]
            key:   [B, K, D]
            value: [B, K, D]
            mask:  [B, Q, K] or None — True = attend, False = mask out
        Returns:
            output: [B, Q, D]
        """
        B, Q, _ = query.shape
        K = key.shape[1]

        q = self.q_proj(query).view(B, Q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Q, -1)
        return self.out_proj(out)


class CompressedBranch(nn.Module):
    """Branch 1: attends to a fixed set of compressed latent tokens."""

    def __init__(self, d_model: int, n_compress: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.n_compress = n_compress
        self.compress_proj = nn.Linear(d_model, d_model)
        self.attn = NSABranch(d_model, n_heads, dropout)

    def forward(self, query: torch.Tensor,
                compressed_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:          [B, Q, D]
            compressed_ctx: [B, C, D] where C = n_compress
        Returns:
            output: [B, Q, D]
        """
        ctx = self.compress_proj(compressed_ctx)
        return self.attn(query, ctx, ctx)


class SelectedBranch(nn.Module):
    """Branch 2: top-k selection from an external key/value bank."""

    def __init__(self, d_model: int, n_select_k: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.n_select_k = n_select_k
        self.score_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn = NSABranch(d_model, n_heads, dropout)

    def forward(self, query: torch.Tensor,
                bank_keys: torch.Tensor,
                bank_values: torch.Tensor,
                bank_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:       [B, Q, D]
            bank_keys:   [B, M, D] — full bank keys
            bank_values: [B, M, D] — full bank values
            bank_mask:   [B, M] — True = valid entry, False = empty/quarantined
        Returns:
            output:  [B, Q, D]
            indices: [B, Q, K] — selected bank indices per query
        """
        B, Q, D = query.shape
        M = bank_keys.shape[1]
        K = min(self.n_select_k, M)

        score_q = self.score_proj(query)
        scores = torch.bmm(score_q, bank_keys.transpose(1, 2)) * (D ** -0.5)

        if bank_mask is not None:
            mask_expanded = bank_mask.unsqueeze(1).expand(B, Q, M)
            scores = scores.masked_fill(~mask_expanded, torch.finfo(scores.dtype).min)

        topk_scores, topk_idx = scores.topk(K, dim=-1)

        flat_idx = topk_idx.reshape(B, -1)
        sel_k = bank_keys.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, D)).view(B, Q, K, D)
        sel_v = bank_values.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, D)).view(B, Q, K, D)

        sel_k = sel_k.reshape(B * Q, K, D)
        sel_v = sel_v.reshape(B * Q, K, D)
        q_flat = query.reshape(B * Q, 1, D)

        out = self.attn(q_flat, sel_k, sel_v)
        out = out.view(B, Q, D)

        return out, topk_idx


class SlidingBranch(nn.Module):
    """Branch 3: local sliding window attention over recent tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = NSABranch(d_model, n_heads, dropout)

    def forward(self, query: torch.Tensor,
                sliding_buffer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:          [B, Q, D]
            sliding_buffer: [B, W, D] — last W*P tokens concatenated
        Returns:
            output: [B, Q, D]
        """
        return self.attn(query, sliding_buffer, sliding_buffer)


class NSAAttention(nn.Module):
    """
    3-branch Native Sparse Attention.

    Fuses compressed (long-term), selected (spatial recall), and sliding
    (short-term) branches via a learned per-query gate.

    Args:
        d_model:        token dimension
        n_compress:     number of compressed latent tokens
        n_select_k:     top-k entries to select from bank per query
        n_heads:        attention heads per branch
        dropout:        attention dropout
    """

    def __init__(self, d_model: int = 128, n_compress: int = 32,
                 n_select_k: int = 8, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_compress = n_compress
        self.n_select_k = n_select_k

        self.compressed = CompressedBranch(d_model, n_compress, n_heads, dropout)
        self.selected = SelectedBranch(d_model, n_select_k, n_heads, dropout)
        self.sliding = SlidingBranch(d_model, n_heads, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor,
                compressed_ctx: torch.Tensor,
                bank_keys: torch.Tensor,
                bank_values: torch.Tensor,
                sliding_buffer: torch.Tensor,
                bank_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query:          [B, Q, D]
            compressed_ctx: [B, C, D]
            bank_keys:      [B, M, D]
            bank_values:    [B, M, D]
            sliding_buffer: [B, W, D]
            bank_mask:      [B, M] — True = valid
        Returns:
            output:               [B, Q, D]
            branch_weights:       [B, Q, 3]
            selected_indices:     [B, Q, K]
        """
        out_c = self.compressed(query, compressed_ctx)
        out_s, sel_idx = self.selected(query, bank_keys, bank_values, bank_mask)
        out_w = self.sliding(query, sliding_buffer)

        gate_logits = self.gate(query)
        gate_weights = F.softmax(gate_logits, dim=-1)

        branches = torch.stack([out_c, out_s, out_w], dim=-2)
        fused = (gate_weights.unsqueeze(-1) * branches).sum(dim=-2)
        output = self.norm(fused + query)

        return {
            "output": output,
            "branch_weights": gate_weights,
            "selected_indices": sel_idx,
        }
