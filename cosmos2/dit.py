"""
Cosmos 2.5 Diffusion Transformer (DiT).

Adapted from cosmos1/dit.py.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
1. CHANGED  : FullSpaceTimeAttention → ChunkedSpaceTimeAttention
              The attention module swap is the single most impactful change.
              It reduces peak attention memory from O(N²) to O(C·N).

2. CHANGED  : AdaLNZero → AdaLNZeroMultiModal
              A second input projection is added to also condition on the
              pooled latent embedding of the conditioning image/video.

3. CHANGED  : text cross-attention placement
              Cosmos 1 applied a single cross-attention before block 0.
              Cosmos 2.5 applies it every ``cross_attn_every_n_layers``
              blocks (default: every block), giving the text signal
              continuous reinforcement throughout the network.

4. CHANGED  : three task-specific output heads → single unified head
              head_text2world, head_image2world, head_video2world are all
              REMOVED and replaced by a single ``out_proj``.

5. CHANGED  : hidden_dim 4096 → 2048 (2B variant)
              head_dim    256  → 128

6. CHANGED  : conditioning vector construction
              Cosmos 1: cond = time_emb + text_pool
              Cosmos 2.5: cond = time_emb + text_pool + visual_pool
              (visual_pool comes from the encoded conditioning latent)

7. REMOVED  : ``task`` argument from forward()
8. REMOVED  : conditional token prepending strategy
              (conditioning via adaLN instead of token concat)
9. ADDED    : ``visual_pool`` conditioning branch

UNCHANGED   : FeedForward (SwiGLU), TimeEmbedder, DiTBlock residual structure,
              zero-init of output heads, latent_proj, norm_out.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..shared.activations import SwiGLU
from ..shared.normalization import RMSNorm
from ..shared.embeddings import SinusoidalPosEmbed, build_rope3d
from .attention import ChunkedSpaceTimeAttention
from .config import DiTConfig


# ---------------------------------------------------------------------------
# Feed-Forward Block (UNCHANGED from Cosmos 1)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU FFN — identical to cosmos1/dit.py.  No changes needed."""

    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1  = nn.Linear(hidden_dim, 2 * inner, bias=False)
        self.act  = SwiGLU()
        self.fc2  = nn.Linear(inner, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# ADDED: Multi-Modal AdaLN-Zero
# ---------------------------------------------------------------------------

class AdaLNZeroMultiModal(nn.Module):
    """Adaptive Layer Norm with multi-modal conditioning.

    ADDED in Cosmos 2.5.  Extends AdaLNZero from cosmos1/normalization.py
    by accepting a *visual* conditioning vector in addition to the combined
    (time + text) vector.

    The two vectors are summed after independent linear projections before
    computing the 6 modulation parameters.

    Cosmos 1 used:
        cond = time_emb + text_pool  →  single projection

    Cosmos 2.5 uses:
        cond = proj_A(time_emb + text_pool) + proj_B(visual_pool)
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm   = RMSNorm(dim)
        self.proj_a = nn.Linear(cond_dim, 6 * dim, bias=True)  # text+time branch
        self.proj_b = nn.Linear(cond_dim, 6 * dim, bias=True)  # visual branch (ADDED)
        nn.init.zeros_(self.proj_a.weight)
        nn.init.zeros_(self.proj_a.bias)
        nn.init.zeros_(self.proj_b.weight)
        nn.init.zeros_(self.proj_b.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond_text: torch.Tensor,
        cond_visual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (B, N, D)
            cond_text   : (B, cond_dim) — time + text pool
            cond_visual : (B, cond_dim) — visual pool (image/video), optional

        Returns:
            Same 6-tuple as AdaLNZero.
        """
        params = self.proj_a(cond_text)
        if cond_visual is not None:
            # ADDED: merge visual conditioning.
            params = params + self.proj_b(cond_visual)
        params = params.unsqueeze(1)           # (B, 1, 6*D)
        chunks = params.chunk(6, dim=-1)

        x_normed = self.norm(x)
        x_modulated = x_normed * (1 + chunks[1]) + chunks[0]
        return (x_modulated, chunks[2],
                x_normed,    chunks[3], chunks[4], chunks[5])


# ---------------------------------------------------------------------------
# Time Embedder (UNCHANGED from Cosmos 1)
# ---------------------------------------------------------------------------

class TimeEmbedder(nn.Module):
    """sinusoidal → MLP — identical to cosmos1/dit.py."""

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
# Cosmos 2.5 DiT Block
# ---------------------------------------------------------------------------

class Cosmos2DiTBlock(nn.Module):
    """Single Cosmos 2.5 DiT block.

    ── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────
    CHANGED  : FullSpaceTimeAttention → ChunkedSpaceTimeAttention
    CHANGED  : AdaLNZero → AdaLNZeroMultiModal
    ADDED    : in-block cross-attention for text (``apply_cross_attn`` flag)
    UNCHANGED: FeedForward, residual gating structure
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        cond_dim: int,
        mlp_ratio: float,
        rope: nn.Module,
        chunk_size: int = 8,
        apply_cross_attn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # CHANGED: ChunkedSpaceTimeAttention (was FullSpaceTimeAttention)
        self.attn = ChunkedSpaceTimeAttention(
            hidden_dim, num_heads, head_dim, rope, chunk_size, dropout=dropout
        )
        self.ff   = FeedForward(hidden_dim, mlp_ratio)
        # CHANGED: AdaLNZeroMultiModal (was AdaLNZero)
        self.adaln = AdaLNZeroMultiModal(hidden_dim, cond_dim)

        # ADDED: per-block text cross-attention (conditional).
        self.apply_cross_attn = apply_cross_attn
        if apply_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                kdim=cond_dim,
                vdim=cond_dim,
                batch_first=True,
            )
            self.cross_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond_text: torch.Tensor,
        cond_visual: torch.Tensor | None,
        text_seq: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> torch.Tensor:
        # CHANGED: AdaLNZeroMultiModal accepts cond_text AND cond_visual.
        x_mod, gate_msa, x_norm, shift_mlp, scale_mlp, gate_mlp = \
            self.adaln(x, cond_text, cond_visual)

        # CHANGED: ChunkedSpaceTimeAttention
        attn_out = self.attn(x_mod, t_idx, h_idx, w_idx)
        x = x + gate_msa.tanh() * attn_out

        # ADDED: per-block text cross-attention
        if self.apply_cross_attn:
            x = x + self.cross_attn(
                query=self.cross_norm(x),
                key=text_seq,
                value=text_seq,
            )[0]

        x_ff_in = x_norm * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp.tanh() * self.ff(x_ff_in)
        return x


# ---------------------------------------------------------------------------
# Visual Pool Projector (ADDED in Cosmos 2.5)
# ---------------------------------------------------------------------------

class VisualPoolProjector(nn.Module):
    """Project the conditioning image/video latent to a single pool vector.

    ADDED in Cosmos 2.5.  Not present in Cosmos 1.

    In Cosmos 1, conditioning tokens were *prepended* to the latent sequence
    (a form of in-context conditioning).  In Cosmos 2.5 the conditioning
    latent is instead *pooled* and injected via adaLN-Zero, which is cheaper
    and avoids increasing N.

    The projector flattens and mean-pools the conditioning latent (any shape),
    then maps it to the conditioning dimension.
    """

    def __init__(self, latent_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_channels, cond_dim)
        self.norm = RMSNorm(cond_dim)

    def forward(self, cond_latent: torch.Tensor) -> torch.Tensor:
        """Pool conditioning latent to a single vector.

        Args:
            cond_latent: (B, C, T', H', W') — encoded conditioning latent.

        Returns:
            (B, cond_dim) visual pool vector.
        """
        B, C = cond_latent.shape[:2]
        # Flatten spatial-temporal dims and mean-pool.
        flat = cond_latent.reshape(B, C, -1).mean(dim=-1)  # (B, C)
        return self.norm(self.proj(flat))


# ---------------------------------------------------------------------------
# Cosmos 2.5 DiT
# ---------------------------------------------------------------------------

class Cosmos2DiT(nn.Module):
    """Cosmos 2.5 Diffusion Transformer.

    Key differences from Cosmos1DiT
    ─────────────────────────────────
    1. Uses ChunkedSpaceTimeAttention (cheaper).
    2. Multi-modal adaLN-Zero conditioning (text + time + visual).
    3. Per-block text cross-attention.
    4. Single unified output head (no task argument).
    5. Conditioning latent pooled via VisualPoolProjector (not prepended).
    6. Supports longer sequences (max_frames=121 via config).
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.cfg = cfg

        # ── Positional embedding ──────────────────────────────────────
        # CHANGED: max_frames 57 → 121  (config-driven, class body unchanged)
        self.rope = build_rope3d(
            rope_type=cfg.rope_type,
            head_dim=cfg.head_dim,
            num_heads=cfg.num_heads,
            max_frames=cfg.max_frames,
            max_h=cfg.max_h,
            max_w=cfg.max_w,
            mrope_section=cfg.mrope_section,
        )

        # ── Input projection (UNCHANGED) ──────────────────────────────
        from .config import TokenizerConfig
        latent_ch = TokenizerConfig().latent_channels
        self.latent_proj = nn.Linear(latent_ch, D, bias=True)

        # ── Time embedding (UNCHANGED) ────────────────────────────────
        self.time_embed = TimeEmbedder(cfg.time_embed_dim, cfg.cond_dim)

        # ── ADDED: Visual pool projector ──────────────────────────────
        # Not present in Cosmos 1.
        self.visual_pool = VisualPoolProjector(latent_ch, cfg.cond_dim)

        # ── DiT Blocks ────────────────────────────────────────────────
        # CHANGED: Cosmos2DiTBlock (uses ChunkedSpaceTimeAttention + per-block cross-attn)
        self.blocks = nn.ModuleList([
            Cosmos2DiTBlock(
                hidden_dim=D,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                cond_dim=cfg.cond_dim,
                mlp_ratio=cfg.mlp_ratio,
                rope=self.rope,
                chunk_size=cfg.temporal_chunk_size,
                apply_cross_attn=(i % cfg.cross_attn_every_n_layers == 0),
                dropout=cfg.dropout,
            )
            for i in range(cfg.num_layers)
        ])

        # ── CHANGED: Single unified output head (was 3 task heads) ────
        self.norm_out = RMSNorm(D)
        self.out_proj = nn.Linear(D, latent_ch, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------

    def _make_position_indices(
        self, T: int, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.arange(T, device=device)
        h = torch.arange(H, device=device)
        w = torch.arange(W, device=device)
        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")
        return grid_t.flatten(), grid_h.flatten(), grid_w.flatten()

    # ------------------------------------------------------------------

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        # REMOVED: task argument — no longer needed (unified head)
        # CHANGED: image_cond is pooled, not prepended
        image_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field v(z_t, t, condition).

        Args:
            z_t       : Noisy latent (B, C, T', H', W').
            t         : Noise level  (B,) in [0, 1].
            text_seq  : VLM sequence embeddings (B, L, cond_dim).
            text_pool : VLM pooled embedding    (B, cond_dim).
            image_cond: Optional conditioning latent (B, C, T'', H'', W'').
                        CHANGED: pooled via VisualPoolProjector (not prepended).

        Returns:
            Predicted velocity (B, C, T', H', W').

        ── Cosmos 1 → 2.5 diff (forward pass) ──────────────────────────
        REMOVED: task selection (was: head = {task: self.head_<task>}[task])
        REMOVED: token prepending of image_cond
        ADDED  : visual_pool = self.visual_pool(image_cond)
        CHANGED: cond_text = time_emb + text_pool  (same formula)
                 cond_visual = visual_pool          (new)
        CHANGED: block.forward(x, cond_text, cond_visual, text_seq, ...)
                 was: block.forward(x, cond, t/h/w_idx)
        ─────────────────────────────────────────────────────────────────
        """
        B, C, T_l, H_l, W_l = z_t.shape
        N = T_l * H_l * W_l

        # ── Flatten and project latent tokens (UNCHANGED) ─────────────
        x = z_t.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        x = self.latent_proj(x)

        # ── Positional indices (UNCHANGED) ────────────────────────────
        t_idx, h_idx, w_idx = self._make_position_indices(T_l, H_l, W_l, z_t.device)

        # ── Conditioning vectors ──────────────────────────────────────
        time_emb   = self.time_embed(t)               # (B, cond_dim)
        cond_text  = time_emb + text_pool             # UNCHANGED formula

        # ADDED: visual pool conditioning (not present in Cosmos 1)
        cond_visual: torch.Tensor | None = None
        if image_cond is not None:
            cond_visual = self.visual_pool(image_cond)  # (B, cond_dim)

        # ── DiT Blocks ────────────────────────────────────────────────
        # CHANGED: pass cond_visual and text_seq per block
        for block in self.blocks:
            x = block(x, cond_text, cond_visual, text_seq, t_idx, h_idx, w_idx)

        # ── Unified output head (CHANGED: was 3 task heads) ───────────
        x = self.norm_out(x)
        v = self.out_proj(x)

        # Reshape back to (B, C, T', H', W').
        v = v.reshape(B, T_l, H_l, W_l, C).permute(0, 4, 1, 2, 3)
        return v
