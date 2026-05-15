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

── Multimodal RoPE variants ─────────────────────────────────────────────
Standard 3-D RoPE (RoPE3D) splits head_dim into three contiguous sub-bands
and assigns each one to a positional axis: [T...T | H...H | W...W].

Two improved variants from "Revisiting Multimodal Positional Encoding in
Vision-Language Models" (Huang et al., ICLR 2026 / arXiv:2510.23095):

  MRoPEInterleave3D – interleaves T/H/W assignments across channels as
    [T H W T H W ...], so every axis accesses the full frequency spectrum
    rather than a narrow sub-band.  Used by default in Qwen3-VL.

  MHRoPE3D – assigns whole attention heads to axes instead of channel
    segments: heads 0..n_t-1 encode temporal, n_t..n_t+n_h-1 encode
    height, and the rest encode width.  Each head gets a full-spectrum
    RoPE for its one axis.

Use ``build_rope3d`` to select the variant via a string key.
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
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        # Split head_dim as evenly as possible across T, H, W axes.
        # Each sub-dimension must be even for the rotate_half trick.
        assert head_dim % 2 == 0, "head_dim must be even for RoPE3D"
        # Compute the largest even number ≤ head_dim // 3 for T and H axes.
        dim_t = (head_dim // 3) & ~1   # floor to even
        dim_h = (head_dim // 3) & ~1   # floor to even
        # W absorbs the remainder; head_dim is even and dim_t, dim_h are even,
        # so dim_w = head_dim - dim_t - dim_h is automatically even.
        dim_w = head_dim - dim_t - dim_h
        assert dim_w > 0 and dim_w % 2 == 0, (
            f"Invalid RoPE3D split: dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}"
        )
        self._dim_t = dim_t
        self._dim_h = dim_h
        self._dim_w = dim_w

        freqs_t = self._build_freqs(dim_t, max_frames, theta)
        freqs_h = self._build_freqs(dim_h, max_h, theta)
        freqs_w = self._build_freqs(dim_w, max_w, theta)
        self.register_buffer("freqs_t", freqs_t)
        self.register_buffer("freqs_h", freqs_h)
        self.register_buffer("freqs_w", freqs_w)

    @staticmethod
    def _build_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (half,)
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


# ---------------------------------------------------------------------------
# MRoPE-Interleave 3-D
# ---------------------------------------------------------------------------

class MRoPEInterleave3D(nn.Module):
    """3-D RoPE with interleaved T/H/W frequency allocation (MRoPE-Interleave).

    Proposed in "Revisiting Multimodal Positional Encoding in Vision-Language
    Models" (Huang et al., ICLR 2026 / arXiv:2510.23095) and adopted as the
    default positional encoding in Qwen3-VL and Qwen3.5.

    Standard 3-D RoPE (RoPE3D above) splits head_dim into three contiguous
    sub-bands:  [T…T | H…H | W…W].  Each axis is therefore confined to a
    narrow slice of the frequency spectrum, causing *spectral imbalance*:
    for example, the temporal axis only sees the lowest frequencies while
    spatial axes get higher-frequency bands.

    MRoPE-Interleave fixes this by interleaving the axis assignments across
    the channel (frequency-pair) dimension:

        channel pair index:  0  1  2  3  4  5  6  7  8  …
        axis assignment:     T  H  W  T  H  W  T  H  W  …

    Every axis now spans the full frequency spectrum, giving richer and more
    balanced positional representations.

    All three axes share a single ``inv_freq`` table of size ``head_dim//2``
    built from the same base ``theta``.  Per-token angles differ only in
    which positional index (t, h, or w) is multiplied by the frequency.

    Args:
        head_dim:      Per-head feature dimension (must be even, ≥ 6).
        max_frames:    Maximum number of latent temporal frames.
        max_h:         Maximum latent height in tokens.
        max_w:         Maximum latent width in tokens.
        theta:         RoPE base frequency (default 10 000).
        mrope_section: (n_t, n_h, n_w) — number of frequency pairs assigned
                       to each axis within ``head_dim // 2``.  Must sum to
                       ``head_dim // 2``, with ``n_h`` and ``n_w`` small
                       enough that H/W channel indices stay within range
                       (``n_h * 3 - 2 < head_dim // 2`` and
                       ``n_w * 3 - 1 < head_dim // 2``).  Defaults to an
                       equal split (``n_h = n_w = (head_dim // 2) // 3``,
                       ``n_t`` absorbs the remainder).
    """

    def __init__(
        self,
        head_dim: int,
        max_frames: int = 57,
        max_h: int = 40,
        max_w: int = 64,
        theta: float = 10000.0,
        mrope_section: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        half = head_dim // 2

        if mrope_section is None:
            n_h = n_w = half // 3
            n_t = half - n_h - n_w
            mrope_section = (n_t, n_h, n_w)

        n_t, n_h, n_w = mrope_section
        assert n_t + n_h + n_w == half, (
            f"mrope_section {mrope_section} must sum to head_dim//2={half}"
        )
        assert n_t > 0 and n_h > 0 and n_w > 0, (
            "all mrope_section counts must be positive"
        )
        # Ensure interleaved H/W indices stay within [0, half).
        assert n_h * 3 - 2 < half, (
            f"n_h={n_h} too large: H channel {n_h * 3 - 2} >= half={half}"
        )
        assert n_w * 3 - 1 < half, (
            f"n_w={n_w} too large: W channel {n_w * 3 - 1} >= half={half}"
        )
        self._mrope_section = mrope_section

        # One shared inv_freq of size head_dim//2 for all three axes.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )  # (half,)

        # Precompute angle lookup tables (position × frequency) per axis.
        self.register_buffer(
            "freqs_t",
            torch.outer(torch.arange(max_frames, dtype=torch.float), inv_freq),
        )  # (max_frames, half)
        self.register_buffer(
            "freqs_h",
            torch.outer(torch.arange(max_h, dtype=torch.float), inv_freq),
        )  # (max_h, half)
        self.register_buffer(
            "freqs_w",
            torch.outer(torch.arange(max_w, dtype=torch.float), inv_freq),
        )  # (max_w, half)

        # Interleaved channel-index sets (registered as non-persistent buffers
        # so they are moved with the module but not saved to checkpoints).
        #   H channels: 1, 4, 7, …, 3*(n_h-1)+1   →  n_h entries
        #   W channels: 2, 5, 8, …, 3*(n_w-1)+2   →  n_w entries
        #   T channels: everything else             →  n_t entries
        self.register_buffer(
            "h_ch", torch.arange(1, n_h * 3, 3, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "w_ch", torch.arange(2, n_w * 3, 3, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MRoPE-Interleave to queries and keys.

        Args:
            q, k:              (B, heads, N, head_dim).
            t_idx, h_idx, w_idx: (N,) position index per token on each axis.

        Returns:
            Rotated (q, k) of the same shape.
        """
        # Full-spectrum angle lookup per axis; shape (N, half).
        angles = self.freqs_t[t_idx].clone()                  # base: all T
        angles[:, self.h_ch] = self.freqs_h[h_idx][:, self.h_ch]  # overwrite H slots
        angles[:, self.w_ch] = self.freqs_w[w_idx][:, self.w_ch]  # overwrite W slots

        # Duplicate for the rotate-half formula → (N, head_dim).
        angles = torch.cat([angles, angles], dim=-1)
        cos = angles.cos()[None, None]  # (1, 1, N, head_dim) — broadcast over B, heads
        sin = angles.sin()[None, None]

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# MHRoPE 3-D (Multi-Head RoPE)
# ---------------------------------------------------------------------------

class MHRoPE3D(nn.Module):
    """3-D RoPE with head-wise axis allocation (Multi-Head RoPE / MHRoPE).

    Proposed in "Revisiting Multimodal Positional Encoding in Vision-Language
    Models" (Huang et al., ICLR 2026 / arXiv:2510.23095).

    Standard 3-D RoPE (RoPE3D) splits head_dim into sub-bands per axis, so
    every head mixes T/H/W rotations within a single feature vector.  MHRoPE
    instead allocates *whole attention heads* to positional axes:

        head 0 … n_t-1          → temporal (T) axis
        head n_t … n_t+n_h-1   → height   (H) axis
        head n_t+n_h … end      → width    (W) axis

    Each head receives the full ``head_dim``-wide RoPE for its one axis,
    maximising frequency utilisation and avoiding spectral imbalance.
    All axes share the same ``inv_freq`` table; only the position index
    multiplied per token differs.

    The ``forward`` signature is identical to RoPE3D, so no changes to the
    attention modules are required.

    Args:
        head_dim:      Per-head feature dimension (must be even).
        num_heads:     Total number of attention heads (= num_kv_heads for MHA).
        max_frames:    Maximum number of latent temporal frames.
        max_h:         Maximum latent height in tokens.
        max_w:         Maximum latent width in tokens.
        theta:         RoPE base frequency (default 10 000).
        mrope_section: (n_t, n_h, n_w) — number of heads assigned to each
                       axis.  Must sum to ``num_heads``.  Defaults to an
                       equal split (n_h = n_w = num_heads // 3, n_t absorbs
                       the remainder).
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        max_frames: int = 57,
        max_h: int = 40,
        max_w: int = 64,
        theta: float = 10000.0,
        mrope_section: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"

        if mrope_section is None:
            n_h = n_w = num_heads // 3
            n_t = num_heads - n_h - n_w
            mrope_section = (n_t, n_h, n_w)

        n_t, n_h, n_w = mrope_section
        assert n_t + n_h + n_w == num_heads, (
            f"mrope_section {mrope_section} must sum to num_heads={num_heads}"
        )
        assert n_t > 0 and n_h > 0 and n_w > 0, (
            "all mrope_section counts must be positive"
        )
        self._mrope_section = mrope_section

        # Shared inv_freq for all axes, size head_dim//2.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )  # (half,)

        # Angle lookup tables per axis (all same shape, same frequencies).
        self.register_buffer(
            "freqs_t",
            torch.outer(torch.arange(max_frames, dtype=torch.float), inv_freq),
        )  # (max_frames, half)
        self.register_buffer(
            "freqs_h",
            torch.outer(torch.arange(max_h, dtype=torch.float), inv_freq),
        )  # (max_h, half)
        self.register_buffer(
            "freqs_w",
            torch.outer(torch.arange(max_w, dtype=torch.float), inv_freq),
        )  # (max_w, half)

        # head_axis[i] ∈ {0, 1, 2} indicates which axis head i encodes.
        head_axis = torch.tensor(
            [0] * n_t + [1] * n_h + [2] * n_w, dtype=torch.long
        )
        self.register_buffer("head_axis", head_axis, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MHRoPE to queries and keys.

        Each head is rotated by the full-spectrum RoPE of its assigned axis.

        Args:
            q, k:              (B, heads, N, head_dim).
            t_idx, h_idx, w_idx: (N,) position index per token on each axis.

        Returns:
            Rotated (q, k) of the same shape.
        """
        # Full-spectrum angle lookup per axis; shape (N, half).
        a_t = self.freqs_t[t_idx]  # (N, half)
        a_h = self.freqs_h[h_idx]
        a_w = self.freqs_w[w_idx]

        # Stack axes for indexing: (3, N, half).
        axis_angles = torch.stack([a_t, a_h, a_w], dim=0)

        # Gather per-head angles: (num_heads, N, half) → (num_heads, N, head_dim).
        head_angles = axis_angles[self.head_axis]
        head_angles = torch.cat([head_angles, head_angles], dim=-1)

        # Broadcast cos/sin: (1, num_heads, N, head_dim).
        cos = head_angles.cos().unsqueeze(0)
        sin = head_angles.sin().unsqueeze(0)

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rope3d(
    rope_type: str,
    head_dim: int,
    num_heads: int,
    max_frames: int,
    max_h: int,
    max_w: int,
    theta: float = 10000.0,
    mrope_section: tuple | None = None,
) -> nn.Module:
    """Construct a 3-D RoPE module by name.

    Args:
        rope_type:     One of ``"standard"``, ``"mrope_interleave"``,
                       or ``"mhrope"``.
        head_dim:      Per-head feature dimension.
        num_heads:     Number of attention heads (only used for ``"mhrope"``).
        max_frames:    Maximum temporal extent in latent frames.
        max_h:         Maximum height extent in latent tokens.
        max_w:         Maximum width extent in latent tokens.
        theta:         RoPE base frequency.
        mrope_section: Axis split for MRoPEInterleave3D (freq-pair counts) or
                       MHRoPE3D (head counts).  ``None`` = auto-split.

    Returns:
        An ``nn.Module`` with signature
        ``forward(q, k, t_idx, h_idx, w_idx) -> (q_rot, k_rot)``.
    """
    if rope_type == "standard":
        return RoPE3D(head_dim, max_frames, max_h, max_w, theta)
    elif rope_type == "mrope_interleave":
        return MRoPEInterleave3D(
            head_dim=head_dim,
            max_frames=max_frames,
            max_h=max_h,
            max_w=max_w,
            theta=theta,
            mrope_section=mrope_section,
        )
    elif rope_type == "mhrope":
        return MHRoPE3D(
            head_dim=head_dim,
            num_heads=num_heads,
            max_frames=max_frames,
            max_h=max_h,
            max_w=max_w,
            theta=theta,
            mrope_section=mrope_section,
        )
    else:
        raise ValueError(
            f"Unknown rope_type {rope_type!r}. "
            "Choose 'standard', 'mrope_interleave', or 'mhrope'."
        )
