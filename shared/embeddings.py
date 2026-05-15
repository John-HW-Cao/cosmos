"""
Positional embedding strategies shared by Cosmos 1 and Cosmos 2.5.

Cosmos 1  uses sinusoidal embeddings for time steps and 3-D RoPE for
          spatial + temporal positions inside the DiT blocks.

Cosmos 2.5 keeps 3-D RoPE but extends the temporal component to support
           longer sequences (up to 30 s) without re-training existing
           spatial components.  See cosmos2/attention.py for the usage.

── What is RoPE? ────────────────────────────────────────────────────────
Rotary Position Embedding (Su et al., 2021) encodes position by rotating
query/key vectors in 2-D subspaces.  Unlike absolute embeddings, RoPE is
injected at every attention layer which improves length generalization.
─────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmbed(nn.Module):
    """Classic sinusoidal embedding for scalar positions (e.g. noise level t).

    Used identically in Cosmos 1 and Cosmos 2.5 for noise-level conditioning.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scalar time steps.

        Args:
            t: Shape (B,), values in [0, 1].

        Returns:
            Shape (B, dim).
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]            # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half, negate second half, swap."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RoPE2D(nn.Module):
    """2-D Rotary Position Embedding (height × width).

    Used for spatial tokens inside each video frame.

    Cosmos 1 & 2.5 share this class without modification.
    """

    def __init__(self, head_dim: int, max_h: int = 64, max_w: int = 64) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-D RoPE"
        half_dim = head_dim // 2  # split equally between H and W

        # Build frequency tables for height and width independently.
        freqs_h = self._build_freqs(half_dim, max_h)  # (max_h, half_dim/2)
        freqs_w = self._build_freqs(half_dim, max_w)
        self.register_buffer("freqs_h", freqs_h)
        self.register_buffer("freqs_w", freqs_w)

    @staticmethod
    def _build_freqs(dim: int, max_len: int) -> torch.Tensor:
        # Standard RoPE: use dim/2 distinct frequency bands, each applied twice
        # (once to the first half, once to the second half via rotate_half).
        # arange(0, dim, 2) picks every other index → shape (dim/2,)
        half = dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # (half,)
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)  # (max_len, half)
        return torch.cat([angles, angles], dim=-1)  # (max_len, dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2-D RoPE to queries and keys.

        Args:
            q, k:   (B, heads, N, head_dim) query / key tensors.
            h_idx:  (N,) height index for each spatial token.
            w_idx:  (N,) width  index for each spatial token.

        Returns:
            Rotated q and k tensors of the same shape.
        """
        cos_h = self.freqs_h[h_idx].cos()  # (N, half_dim/2)
        sin_h = self.freqs_h[h_idx].sin()
        cos_w = self.freqs_w[w_idx].cos()
        sin_w = self.freqs_w[w_idx].sin()

        # Concatenate H and W cosine/sine along the head_dim axis.
        cos = torch.cat([cos_h, cos_w], dim=-1)  # (N, head_dim)
        sin = torch.cat([sin_h, sin_w], dim=-1)

        # Broadcast to (B, heads, N, head_dim).
        cos = cos[None, None]
        sin = sin[None, None]

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


class RoPE3D(nn.Module):
    """3-D Rotary Position Embedding (temporal × height × width).

    Used for the full space-time token sequence in both Cosmos 1 and 2.5.

    ── Cosmos 1 vs Cosmos 2.5 ──────────────────────────────────────────
    Cosmos 1  : max_frames=57  (≈4 s at 14 fps after 4× temporal pooling)
    Cosmos 2.5: max_frames=121 (≈30 s) — only this constructor argument
                changes; the rest of the implementation is identical.
                See cosmos1/config.py and cosmos2/config.py respectively.
    ────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        head_dim: int,
        max_frames: int = 57,
        max_h: int = 40,
        max_w: int = 64,
    ) -> None:
        super().__init__()
        # Split head_dim as evenly as possible across T, H, W axes.
        # We require head_dim to be even so each axis can use the
        # rotate_half trick (needs even sub-dimensions).
        assert head_dim % 2 == 0, "head_dim must be even for RoPE3D"
        dim_t = head_dim // 3
        dim_h = head_dim // 3
        # Remainder goes to W so that dim_t + dim_h + dim_w == head_dim
        dim_w = head_dim - dim_t - dim_h
        # Ensure each sub-dimension is even for rotate_half
        dim_t = dim_t - (dim_t % 2)
        dim_h = dim_h - (dim_h % 2)
        dim_w = head_dim - dim_t - dim_h  # absorb any remainder into W
        if dim_w % 2 != 0:
            # Shift one unit from dim_h to keep dim_w even
            dim_h -= 1
            dim_w += 1
        self._dim_t = dim_t
        self._dim_h = dim_h
        self._dim_w = dim_w

        freqs_t = self._build_freqs(dim_t, max_frames)
        freqs_h = self._build_freqs(dim_h, max_h)
        freqs_w = self._build_freqs(dim_w, max_w)
        self.register_buffer("freqs_t", freqs_t)
        self.register_buffer("freqs_h", freqs_h)
        self.register_buffer("freqs_w", freqs_w)

    @staticmethod
    def _build_freqs(dim: int, max_len: int) -> torch.Tensor:
        half = dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # (half,)
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)           # (max_len, half)
        return torch.cat([angles, angles], dim=-1)       # (max_len, dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 3-D RoPE to queries and keys.

        Args:
            q, k:               (B, heads, N, head_dim).
            t_idx, h_idx, w_idx: (N,) index tensors for each token.

        Returns:
            Rotated q and k tensors of the same shape.
        """
        cos_t = self.freqs_t[t_idx].cos()   # (N, dim_t)
        sin_t = self.freqs_t[t_idx].sin()
        cos_h = self.freqs_h[h_idx].cos()   # (N, dim_h)
        sin_h = self.freqs_h[h_idx].sin()
        cos_w = self.freqs_w[w_idx].cos()   # (N, dim_w)
        sin_w = self.freqs_w[w_idx].sin()

        # Apply RoPE independently to each axis's sub-dimension, then concat.
        # Split q/k into T, H, W portions along head_dim.
        q_t = q[..., :self._dim_t]
        q_h = q[..., self._dim_t:self._dim_t + self._dim_h]
        q_w = q[..., self._dim_t + self._dim_h:]
        k_t = k[..., :self._dim_t]
        k_h = k[..., self._dim_t:self._dim_t + self._dim_h]
        k_w = k[..., self._dim_t + self._dim_h:]

        cos_t = cos_t[None, None]  # (1, 1, N, dim_t)
        sin_t = sin_t[None, None]
        cos_h = cos_h[None, None]
        sin_h = sin_h[None, None]
        cos_w = cos_w[None, None]
        sin_w = sin_w[None, None]

        q_rot = torch.cat([
            q_t * cos_t + _rotate_half(q_t) * sin_t,
            q_h * cos_h + _rotate_half(q_h) * sin_h,
            q_w * cos_w + _rotate_half(q_w) * sin_w,
        ], dim=-1)
        k_rot = torch.cat([
            k_t * cos_t + _rotate_half(k_t) * sin_t,
            k_h * cos_h + _rotate_half(k_h) * sin_h,
            k_w * cos_w + _rotate_half(k_w) * sin_w,
        ], dim=-1)
        return q_rot, k_rot
