"""
Cosmos 2.5 Chunked Space-Time Attention.

Replaces the full O(N²) ``FullSpaceTimeAttention`` of Cosmos 1.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
REMOVED  : single global softmax over all N tokens (O(N²) memory)
ADDED    : temporal chunking loop — attention is computed independently
           within each chunk of ``chunk_size`` consecutive latent frames.
           Spatial tokens within a chunk fully attend to each other;
           tokens in different chunks do NOT attend directly.
ADDED    : optional *cross-chunk KV sharing* — each chunk can optionally
           attend to a small summary of the previous chunk's key/values,
           giving a limited form of long-range temporal context without
           paying the full O(N²) cost.
UNCHANGED: multi-head projections (W_q, W_k, W_v, W_o)
UNCHANGED: 3-D RoPE injection (max_frames enlarged in config)
UNCHANGED: QK-normalization

Memory complexity:
    Cosmos 1 : O(N²)            where N = T' × H' × W'
    Cosmos 2.5: O(C × H'W' × N) where C = chunk_size (≪ T')
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.embeddings import RoPE3D
from ..shared.normalization import RMSNorm


class ChunkedSpaceTimeAttention(nn.Module):
    """Temporal-chunked multi-head self-attention for Cosmos 2.5.

    The latent token sequence is split into non-overlapping temporal chunks
    of size ``chunk_size`` (measured in latent frames).  Full attention is
    run *within* each chunk.  Optionally the last K key/value tokens from
    the previous chunk are appended to the current chunk's KV set, providing
    a memory of recent context.

    Args:
        hidden_dim      : Model hidden dimension.
        num_heads       : Number of attention heads.
        head_dim        : Per-head dimension.
        rope            : Pre-built RoPE3D instance.
        chunk_size      : Latent frames per chunk (temporal_chunk_size in cfg).
        prev_kv_frames  : Number of previous-chunk frames to include in KV.
                          0 = no cross-chunk sharing (default).
        dropout         : Attention dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        rope: RoPE3D,
        chunk_size: int = 8,
        prev_kv_frames: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim == num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope = rope
        self.chunk_size = chunk_size          # ADDED vs Cosmos 1
        self.prev_kv_frames = prev_kv_frames  # ADDED vs Cosmos 1
        self.dropout = dropout

        # These projections are identical to Cosmos 1 — NO CHANGE.
        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run chunked space-time self-attention.

        Args:
            x    : (B, N, D) — flattened latent token sequence.
            t_idx: (N,) — temporal index of each token.
            h_idx: (N,) — height  index of each token.
            w_idx: (N,) — width   index of each token.

        Returns:
            (B, N, D) attended features.

        ── Key structural difference from Cosmos 1 ──────────────────────
        Cosmos 1 runs a single:
            attn_weights = softmax(Q @ K.T / scale)   # over all N tokens
            out = attn_weights @ V

        Cosmos 2.5 loops over temporal chunks:
            for each chunk c:
                select tokens belonging to frames in chunk c
                optionally append previous-chunk KV summary
                run local softmax within the chunk
        ─────────────────────────────────────────────────────────────────
        """
        B, N, D = x.shape
        nh, hd = self.num_heads, self.head_dim

        # ── Project to Q, K, V (UNCHANGED vs Cosmos 1) ────────────────
        qkv = self.to_qkv(x)
        q, k, v = qkv.reshape(B, N, 3, nh, hd).unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)  # (B, nh, N, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Apply 3-D RoPE (UNCHANGED vs Cosmos 1) ────────────────────
        q, k = self.rope(q, k, t_idx, h_idx, w_idx)

        # ── ADDED: Temporal chunking (not present in Cosmos 1) ─────────
        # Determine unique temporal frames and partition into chunks.
        max_t = int(t_idx.max().item()) + 1
        num_chunks = (max_t + self.chunk_size - 1) // self.chunk_size
        out = torch.zeros_like(q)  # will be filled chunk-by-chunk

        prev_k: torch.Tensor | None = None
        prev_v: torch.Tensor | None = None

        for c in range(num_chunks):
            t_start = c * self.chunk_size
            t_end   = min((c + 1) * self.chunk_size, max_t)

            # Mask of tokens belonging to this temporal chunk.
            chunk_mask = (t_idx >= t_start) & (t_idx < t_end)  # (N,)
            chunk_idx  = chunk_mask.nonzero(as_tuple=False).squeeze(1)

            if chunk_idx.numel() == 0:
                continue

            q_c = q[:, :, chunk_idx]   # (B, nh, Nc, hd)
            k_c = k[:, :, chunk_idx]
            v_c = v[:, :, chunk_idx]

            # Optionally prepend previous-chunk KV context.
            if self.prev_kv_frames > 0 and prev_k is not None:
                # Take the last ``prev_kv_frames`` frames from previous chunk KV.
                kv_ctx_k = prev_k[:, :, -self.prev_kv_frames * (chunk_idx.shape[0] // self.chunk_size):]
                kv_ctx_v = prev_v[:, :, -self.prev_kv_frames * (chunk_idx.shape[0] // self.chunk_size):]
                k_c = torch.cat([kv_ctx_k, k_c], dim=2)
                v_c = torch.cat([kv_ctx_v, v_c], dim=2)

            # Local attention within this chunk.
            # REMOVED global softmax from Cosmos 1; replaced by local:
            attn_w = torch.softmax(
                torch.matmul(q_c, k_c.transpose(-2, -1)) * self.scale,
                dim=-1,
            )
            if self.training and self.dropout > 0:
                attn_w = F.dropout(attn_w, p=self.dropout)

            out_c = torch.matmul(attn_w, v_c)   # (B, nh, Nc, hd)
            # Scatter back into the full output tensor.
            out[:, :, chunk_idx] = out_c

            # Save KV for next chunk's cross-chunk context.
            prev_k = k[:, :, chunk_idx].detach()
            prev_v = v[:, :, chunk_idx].detach()

        # ── Merge heads (UNCHANGED vs Cosmos 1) ───────────────────────
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)
