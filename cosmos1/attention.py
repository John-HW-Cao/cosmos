"""
Cosmos 1 Full Space-Time Self-Attention.

In Cosmos 1 every latent token (across *all* frames and spatial positions)
attends to every other token in a single global attention operation.

Notation used throughout:
    B  – batch size
    T' – number of latent frames   (T / temporal_compression)
    H' – latent height             (H / spatial_compression)
    W' – latent width              (W / spatial_compression)
    N  – sequence length           = T' × H' × W'
    D  – hidden dimension
    nh – number of attention heads
    hd – head dimension            = D / nh

Memory and compute
──────────────────
The attention matrix is (B × nh, N, N).  For a typical Cosmos 1 input
(57 × 40 × 64 = 145 920 tokens) this is enormous and requires gradient
checkpointing + FlashAttention in practice.

── Cosmos 2.5 change ────────────────────────────────────────────────────
Full space-time attention is replaced with ``ChunkedSpaceTimeAttention``
(see cosmos2/attention.py) that divides the token sequence into temporal
chunks and runs full attention *within* each chunk.  This gives O(C·N)
instead of O(N²) memory, where C is the chunk size.

Key code differences (Cosmos 1 → 2.5):
  REMOVED   : global softmax over all N tokens
  ADDED     : temporal chunking loop; cross-chunk KV sharing (optional)
  UNCHANGED : multi-head projection (W_q, W_k, W_v, W_o)
  UNCHANGED : 3-D RoPE injection
  UNCHANGED : QK-normalization (added to both for training stability)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.embeddings import RoPE3D
from ..shared.normalization import RMSNorm


class FullSpaceTimeAttention(nn.Module):
    """Global multi-head self-attention over the full space-time token grid.

    This is the Cosmos 1 attention module.  Every one of the N = T'×H'×W'
    tokens attends to every other token in a single softmax computation.

    Args:
        hidden_dim : Model hidden dimension D.
        num_heads  : Number of attention heads nh.
        head_dim   : Per-head dimension hd (D must equal nh × hd).
        rope       : Pre-built RoPE3D instance shared across all layers.
        dropout    : Attention dropout probability (0 in inference).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        rope: RoPE3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim == num_heads * head_dim, (
            f"hidden_dim ({hidden_dim}) must equal num_heads ({num_heads}) "
            f"× head_dim ({head_dim})"
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope = rope
        self.dropout = dropout

        # QKV and output projections — identical in Cosmos 1 and 2.5.
        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # QK-normalization for training stability — kept in both generations.
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run full space-time self-attention.

        Args:
            x    : (B, N, D)  — flattened latent token sequence.
            t_idx: (N,)        — temporal index of each token.
            h_idx: (N,)        — height  index of each token.
            w_idx: (N,)        — width   index of each token.

        Returns:
            (B, N, D) attended features.
        """
        B, N, D = x.shape
        nh, hd = self.num_heads, self.head_dim

        # ── Project to Q, K, V ─────────────────────────────────────────
        qkv = self.to_qkv(x)                          # (B, N, 3*D)
        q, k, v = qkv.reshape(B, N, 3, nh, hd).unbind(dim=2)
        # Shapes: (B, N, nh, hd)

        # Apply QK-norm before RoPE (Cosmos 1 convention).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, nh, N, hd) for attention computation.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Apply 3-D RoPE ─────────────────────────────────────────────
        # Cosmos 1 & 2.5 both use RoPE3D; the only difference is
        # max_frames (57 vs 121) set in the config.
        q, k = self.rope(q, k, t_idx, h_idx, w_idx)

        # ── Full global attention ──────────────────────────────────────
        # This is the expensive O(N²) step that Cosmos 2.5 replaces
        # with chunked attention (see cosmos2/attention.py).
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # (B, nh, N, hd) → (B, N, D)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)
