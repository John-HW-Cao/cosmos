"""
Cosmos 1 Diffusion Transformer (DiT).

The DiT is the core denoising network.  It operates entirely in the
compressed latent space produced by the ContinuousVideoTokenizer.

Architecture summary
────────────────────
1. Patchify latent tokens (the tokenizer already patchified to latent;
   an optional extra linear projection is applied here).
2. Add time-step conditioning via sinusoidal embedding + MLP.
3. Add text cross-attention conditioning (T5 sequence embeddings).
4. Run N × DiTBlock (self-attention + FFN, modulated by adaLN-Zero).
5. Un-patchify to predict the noise / velocity field.

Each DiTBlock contains:
    AdaLNZero   → modulate input before MSA
    FullSpaceTimeAttention  (Cosmos 1)   or
    ChunkedSpaceTimeAttention (Cosmos 2.5 — see cosmos2/dit.py)
    FeedForward (SwiGLU)
    Residual connections with gating (gate_msa, gate_mlp from adaLN-Zero)

── Cosmos 2.5 changes ──────────────────────────────────────────────────
1. CHANGED  : FullSpaceTimeAttention → ChunkedSpaceTimeAttention
2. CHANGED  : AdaLNZero → AdaLNZeroMultiModal  (adds image/video token
              conditioning path alongside text + time)
3. ADDED    : CrossAttention for T5/VLM embeddings (Cosmos 1 uses simple
              concatenation; Cosmos 2.5 uses dedicated cross-attn blocks)
4. CHANGED  : hidden_dim 4096 → 2048 (2B variant) or 4096 (14B variant)
5. REMOVED  : task-specific output heads (Text2W, Image2W, Video2W);
              replaced by a single unified head (unified_conditioning=True)
6. ADDED    : RL fine-tuning hooks (reward weighting in loss)

See cosmos2/dit.py for the complete adapted implementation.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.activations import SwiGLU
from ..shared.normalization import RMSNorm, AdaLNZero
from ..shared.embeddings import SinusoidalPosEmbed, RoPE3D
from .attention import FullSpaceTimeAttention
from .config import DiTConfig


# ---------------------------------------------------------------------------
# Feed-Forward Block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU feed-forward network.

    Cosmos 1 & 2.5 use the same FFN structure.
    The output dimension doubles the expansion to account for the
    gate branch being halved by SwiGLU.

    hidden_dim → (hidden_dim × mlp_ratio × 2) → SwiGLU → hidden_dim
    """

    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        # Expand by 2× to feed into SwiGLU (which halves the dim).
        self.fc1 = nn.Linear(hidden_dim, 2 * inner, bias=False)
        self.act = SwiGLU()
        self.fc2 = nn.Linear(inner, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# Cosmos 1 DiT Block
# ---------------------------------------------------------------------------

class Cosmos1DiTBlock(nn.Module):
    """Single Cosmos 1 DiT block.

    Residual structure::

        x  ──► adaLN-Zero (MSA branch) ──► FullSpaceTimeAttn ──► gate_msa ──► + ──►
        x  ──► adaLN-Zero (MLP branch) ──► FeedForward        ──► gate_mlp ──► + ──►

    The gate_* tensors (from adaLN-Zero) implement the zero-init trick.

    ── Cosmos 2.5 adaptation ────────────────────────────────────────────
    Replace FullSpaceTimeAttention with ChunkedSpaceTimeAttention.
    Replace AdaLNZero with AdaLNZeroMultiModal.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        cond_dim: int,
        mlp_ratio: float,
        rope: RoPE3D,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = FullSpaceTimeAttention(hidden_dim, num_heads, head_dim, rope, dropout)
        self.ff = FeedForward(hidden_dim, mlp_ratio)
        self.adaln = AdaLNZero(hidden_dim, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, N, D) latent token sequence.
            cond : (B, cond_dim) conditioning vector (time + text pooled).
            t/h/w_idx: (N,) positional index tensors.

        Returns:
            (B, N, D) updated token sequence.
        """
        # adaLN-Zero returns (x_mod_msa, gate_msa, x_norm, shift_mlp, scale_mlp, gate_mlp)
        x_mod, gate_msa, x_norm, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, cond)

        # Self-attention branch (full space-time in Cosmos 1)
        attn_out = self.attn(x_mod, t_idx, h_idx, w_idx)
        x = x + gate_msa.tanh() * attn_out

        # Feed-forward branch
        x_ff_in = x_norm * (1 + scale_mlp) + shift_mlp
        ff_out = self.ff(x_ff_in)
        x = x + gate_mlp.tanh() * ff_out
        return x


# ---------------------------------------------------------------------------
# Time & Condition Embedder
# ---------------------------------------------------------------------------

class TimeEmbedder(nn.Module):
    """Embeds the scalar noise level t into a conditioning vector.

    sinusoidal → 2-layer MLP → cond_dim

    Identical in Cosmos 1 and Cosmos 2.5.
    """

    def __init__(self, time_embed_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.sin_embed = SinusoidalPosEmbed(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_embed(t))


# ---------------------------------------------------------------------------
# Cosmos 1 DiT
# ---------------------------------------------------------------------------

class Cosmos1DiT(nn.Module):
    """Cosmos 1 Diffusion Transformer.

    Operates on latent tokens z produced by ContinuousVideoTokenizer.
    Predicts the velocity field v(z_t, t, c) for Rectified Flow training.

    Task variants in Cosmos 1
    ──────────────────────────
    The model has three task-specific output heads:
        text2world  : conditioned on text only
        image2world : conditioned on text + first-frame image
        video2world : conditioned on text + conditioning video frames

    Each head is a separate linear layer applied after the shared trunk.
    The pipeline selects the correct head at runtime.

    ── Cosmos 2.5 adaptation ────────────────────────────────────────────
    All three task-specific heads are REMOVED and replaced with a single
    unified head.  Conditioning is handled by the new multi-modal adaLN-Zero
    module.  See cosmos2/dit.py → Cosmos2DiT.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.cfg = cfg

        # ── Positional embedding ──────────────────────────────────────
        # Cosmos 1: max_frames=57.  Cosmos 2.5 sets max_frames=121.
        self.rope = RoPE3D(
            head_dim=cfg.head_dim,
            max_frames=cfg.max_frames,  # ← config-driven
            max_h=cfg.max_h,
            max_w=cfg.max_w,
        )

        # ── Input projection (latent channels → hidden_dim) ───────────
        from .config import TokenizerConfig
        latent_ch = TokenizerConfig().latent_channels  # 16 by default
        self.latent_proj = nn.Linear(latent_ch, D, bias=True)

        # ── Time embedding ────────────────────────────────────────────
        self.time_embed = TimeEmbedder(cfg.time_embed_dim, cfg.cond_dim)

        # ── Text cross-attention (Cosmos 1: lightweight cross-attn) ───
        # In Cosmos 1 the T5 sequence is injected via a simple cross-
        # attention applied once before the main DiT blocks.
        # Cosmos 2.5 integrates text cross-attention inside every block.
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=cfg.num_heads,
            kdim=cfg.cond_dim,
            vdim=cfg.cond_dim,
            batch_first=True,
        )
        self.text_norm = RMSNorm(D)

        # ── DiT Blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Cosmos1DiTBlock(
                hidden_dim=D,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                cond_dim=cfg.cond_dim,
                mlp_ratio=cfg.mlp_ratio,
                rope=self.rope,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # ── Output head(s) ────────────────────────────────────────────
        # Cosmos 1 has separate heads per task.
        # Cosmos 2.5 replaces all three with a single `out_proj`.
        self.norm_out = RMSNorm(D)
        if not cfg.unified_conditioning:
            # Cosmos 1: three task-specific heads.
            self.head_text2world  = nn.Linear(D, latent_ch, bias=True)
            self.head_image2world = nn.Linear(D, latent_ch, bias=True)
            self.head_video2world = nn.Linear(D, latent_ch, bias=True)
        else:
            # Cosmos 2.5: single unified head.
            self.out_proj = nn.Linear(D, latent_ch, bias=True)

        # Zero-initialize output heads for training stability.
        self._zero_init_output_heads()

    def _zero_init_output_heads(self) -> None:
        if hasattr(self, "out_proj"):
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            for head in [self.head_text2world,
                         self.head_image2world,
                         self.head_video2world]:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------

    def _make_position_indices(
        self, T: int, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build flattened positional index tensors for a T×H×W grid."""
        t = torch.arange(T, device=device)
        h = torch.arange(H, device=device)
        w = torch.arange(W, device=device)
        # meshgrid → flatten
        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")
        return grid_t.flatten(), grid_h.flatten(), grid_w.flatten()

    # ------------------------------------------------------------------

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        task: str = "text2world",
        image_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field v(z_t, t, condition).

        Args:
            z_t      : Noisy latent (B, C, T', H', W').
            t        : Noise level  (B,)  in [0, 1].
            text_seq : T5 sequence embeddings (B, L, cond_dim).
            text_pool: T5 pooled embedding    (B, cond_dim).
            task     : One of "text2world" | "image2world" | "video2world".
                       Selects the output head (Cosmos 1 only).
            image_cond: Optional first-frame latent for image2world / video2world.

        Returns:
            Predicted velocity (B, C, T', H', W').

        ── Cosmos 2.5 adaptation ─────────────────────────────────────
        The ``task`` argument and all per-task branches are removed.
        ``image_cond`` is absorbed into the multi-modal adaLN-Zero
        conditioning.  See cosmos2/dit.py → Cosmos2DiT.forward().
        ──────────────────────────────────────────────────────────────
        """
        B, C, T_l, H_l, W_l = z_t.shape
        N = T_l * H_l * W_l

        # ── Flatten and project latent tokens ─────────────────────────
        x = z_t.permute(0, 2, 3, 4, 1).reshape(B, N, C)  # (B, N, C)
        x = self.latent_proj(x)                            # (B, N, D)

        # Optionally concat image/video conditioning tokens along N.
        if image_cond is not None:
            B2, C2, T2, H2, W2 = image_cond.shape
            cond_tokens = image_cond.permute(0, 2, 3, 4, 1).reshape(B2, -1, C2)
            cond_tokens = self.latent_proj(cond_tokens)
            # Prepend conditioning tokens (Cosmos 1 strategy).
            x = torch.cat([cond_tokens, x], dim=1)
            N_total = x.shape[1]
        else:
            N_total = N

        # ── Positional indices ─────────────────────────────────────────
        t_idx, h_idx, w_idx = self._make_position_indices(T_l, H_l, W_l, z_t.device)
        if image_cond is not None:
            # Extend indices for prepended conditioning tokens.
            tc, hc, wc = self._make_position_indices(T2, H2, W2, z_t.device)
            t_idx = torch.cat([tc, t_idx])
            h_idx = torch.cat([hc, h_idx])
            w_idx = torch.cat([wc, w_idx])

        # ── Conditioning vector (adaLN-Zero input) ────────────────────
        time_emb = self.time_embed(t)        # (B, cond_dim)
        cond = time_emb + text_pool          # combine time + text pool

        # ── Text cross-attention (Cosmos 1: once, before DiT blocks) ──
        x = x + self.text_cross_attn(
            query=self.text_norm(x),
            key=text_seq,
            value=text_seq,
        )[0]

        # ── DiT Blocks ────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, cond, t_idx, h_idx, w_idx)

        # ── Output head selection (Cosmos 1 task-specific) ────────────
        x = self.norm_out(x)
        if hasattr(self, "out_proj"):
            # Cosmos 2.5 unified head (should not be reached from Cosmos1DiT,
            # but kept for completeness if unified_conditioning=True is set).
            v = self.out_proj(x)
        else:
            head = {
                "text2world" : self.head_text2world,
                "image2world": self.head_image2world,
                "video2world": self.head_video2world,
            }[task]
            v = head(x)

        # Strip conditioning tokens if they were prepended.
        if image_cond is not None:
            cond_len = T2 * H2 * W2
            v = v[:, cond_len:]

        # Reshape back to (B, C, T', H', W').
        v = v.reshape(B, T_l, H_l, W_l, C).permute(0, 4, 1, 2, 3)
        return v
