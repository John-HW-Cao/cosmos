"""
Cosmos 1 Continuous Video Tokenizer (CV8x8x8).

Architecture
────────────
The tokenizer is a 3-D convolutional encoder–decoder with residual
blocks and optional spatial self-attention at low resolutions.

Compression axes:
    temporal : 4×  (causal 3-D conv with stride 4 along time)
    spatial  : 8×  (successive 2-D spatial downsampling)

A KL-regularized continuous latent space is used (no codebook).
The encoder outputs a mean and log-variance; during training the
reparameterisation trick is applied.  At inference time the mean
is used directly (deterministic encoding).

── Cosmos 2.5 changes ──────────────────────────────────────────────────
1. spatial_compression : 8 → 16
   The spatial stride doubles.  This halves the number of latent tokens
   along each spatial axis, making the DiT cheaper to run.
   In code: change the number of downsampling stages from 3 to 4
   (or use a larger stride in the final spatial downsampling layer).

2. Encoder base_channels: 128 → 192
   Cosmos 2.5's tokenizer has a wider channel budget to compensate for
   the higher compression and maintain reconstruction quality.

3. Channel multipliers: (1,2,4,8) → (1,2,4,4,8)
   An additional resolution level is inserted to handle the extra 2×
   spatial compression gracefully.

See cosmos2/tokenizer.py for the updated implementation.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TokenizerConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """3-D Residual Block with GroupNorm.

    Used in both the encoder and decoder at every resolution level.
    Identical in Cosmos 1 and Cosmos 2.5 (no changes needed here).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpatialAttnBlock(nn.Module):
    """Single-head spatial self-attention over H×W at a fixed resolution.

    Applied only at the lowest-resolution level of the tokenizer
    encoder / decoder.  Cosmos 2.5 keeps this unchanged.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — apply attention independently per frame.
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        h = self.norm(x_flat)
        qkv = self.qkv(h).reshape(B * T, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))  # (B*T, C, H*W)
        out = self.proj(out.reshape(B * T, C, H, W))
        return (x_flat + out).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


class CausalTemporalDownsample(nn.Module):
    """Causal 3-D conv that downsamples the temporal axis by `stride_t`.

    "Causal" means the convolution only looks at past frames so the model
    can be used in a streaming / autoregressive setting.

    Cosmos 1 : stride_t=4  (temporal compression = 4×)
    Cosmos 2.5: same        (temporal compression unchanged)
    """

    def __init__(self, channels: int, stride_t: int = 4) -> None:
        super().__init__()
        kernel_t = stride_t * 2 - 1          # kernel covers `stride_t` past frames
        self.pad_t = kernel_t - 1            # causal padding amount
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_t, 1, 1),
            stride=(stride_t, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad the time dimension on the left (causal) only.
        x = F.pad(x, (0, 0, 0, 0, self.pad_t, 0))
        return self.conv(x)


class SpatialDownsample(nn.Module):
    """2× spatial downsampling via strided 2-D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialUpsample(nn.Module):
    """2× spatial upsampling via nearest-neighbour + conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = x.reshape(B, T, C, H * 2, W * 2).permute(0, 2, 1, 3, 4)
        return self.conv(x)


class CausalTemporalUpsample(nn.Module):
    """Causal temporal upsampling (inverse of CausalTemporalDownsample)."""

    def __init__(self, channels: int, stride_t: int = 4) -> None:
        super().__init__()
        self.stride_t = stride_t
        self.conv = nn.ConvTranspose3d(
            channels, channels,
            kernel_size=(stride_t, 1, 1),
            stride=(stride_t, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder and Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Cosmos 1 video encoder.

    Down-samples: temporal 4× + spatial 8×.
    Outputs (mean, log_var) for the KL latent.
    """

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        mults = cfg.channel_multipliers  # e.g. (1, 2, 4, 8)
        n_res = cfg.num_res_blocks
        attn_res = set(cfg.attn_resolutions)

        # Initial projection
        self.conv_in = nn.Conv3d(cfg.in_channels, ch, 3, padding=1)

        # ── Downsampling stages ─────────────────────────────────────────
        # Cosmos 1: 3 spatial-downsampling stages (8×) + 1 temporal stage.
        # Cosmos 2.5 adds a 4th spatial stage (→16×) and adjusts
        #   channel_multipliers to (1,2,4,4,8) in its config.
        # In this Encoder the number of stages is driven entirely by
        #   len(channel_multipliers) - 1, so the adaptation is automatic
        #   when the config is updated.
        self.down_blocks = nn.ModuleList()
        in_ch = ch
        current_res = 64  # starting spatial resolution of latent grid
        for i, mult in enumerate(mults):
            out_ch = ch * mult
            stage = nn.ModuleList()
            for _ in range(n_res):
                stage.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                if current_res in attn_res:
                    stage.append(SpatialAttnBlock(out_ch))
            self.down_blocks.append(stage)
            if i < len(mults) - 1:
                # Spatial downsampling between stages.
                self.down_blocks.append(nn.ModuleList([SpatialDownsample(in_ch)]))
                current_res //= 2

        # Temporal downsampling (causal, 4×) — happens after spatial.
        self.temporal_down = CausalTemporalDownsample(in_ch, cfg.temporal_compression)

        # Middle block (bottleneck)
        self.mid_block1 = ResBlock(in_ch, in_ch)
        self.mid_attn = SpatialAttnBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch)

        # Output projection to latent (mean + log_var)
        self.norm_out = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv_out = nn.Conv3d(in_ch, 2 * cfg.latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video to KL latent.

        Args:
            x: (B, 3, T, H, W) raw video in [-1, 1].

        Returns:
            mean, log_var each of shape (B, latent_channels, T/4, H/8, W/8).
        """
        h = self.conv_in(x)

        for block_group in self.down_blocks:
            for layer in block_group:
                h = layer(h)

        h = self.temporal_down(h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        mean, log_var = h.chunk(2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        return mean, log_var


class Decoder(nn.Module):
    """Cosmos 1 video decoder.

    Mirror of Encoder: up-samples temporal 4× + spatial 8×.
    """

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        mults = cfg.channel_multipliers[::-1]  # reversed for upsampling
        n_res = cfg.num_res_blocks
        attn_res = set(cfg.attn_resolutions)

        # Start from the highest-multiplier channel count.
        in_ch = ch * cfg.channel_multipliers[-1]
        self.conv_in = nn.Conv3d(cfg.latent_channels, in_ch, 3, padding=1)

        # Temporal upsampling first (inverse of encoder).
        self.temporal_up = CausalTemporalUpsample(in_ch, cfg.temporal_compression)

        # Middle block
        self.mid_block1 = ResBlock(in_ch, in_ch)
        self.mid_attn = SpatialAttnBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch)

        # ── Upsampling stages ──────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        current_res = 8  # starts at lowest resolution
        for i, mult in enumerate(mults):
            out_ch = ch * mult
            stage = nn.ModuleList()
            for _ in range(n_res + 1):
                stage.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                if current_res in attn_res:
                    stage.append(SpatialAttnBlock(out_ch))
            self.up_blocks.append(stage)
            if i < len(mults) - 1:
                self.up_blocks.append(nn.ModuleList([SpatialUpsample(in_ch)]))
                current_res *= 2

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv_out = nn.Conv3d(in_ch, cfg.in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video.

        Args:
            z: (B, latent_channels, T/4, H/8, W/8)

        Returns:
            (B, 3, T, H, W) in [-1, 1].
        """
        h = self.conv_in(z)
        h = self.temporal_up(h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block_group in self.up_blocks:
            for layer in block_group:
                h = layer(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ContinuousVideoTokenizer(nn.Module):
    """Cosmos 1 Continuous Video Tokenizer (CV8x8x8).

    Wraps Encoder + Decoder and exposes encode / decode helpers.

    Compression summary:
        temporal: 4×   spatial: 8×8   →  total 256× per spatial-temporal patch

    Usage::

        tokenizer = ContinuousVideoTokenizer(TokenizerConfig())
        z_mean, z_logvar = tokenizer.encode(video)  # deterministic mean
        video_recon      = tokenizer.decode(z_mean)

    ── Cosmos 2.5 adaptation ────────────────────────────────────────────
    Replace ``TokenizerConfig(spatial_compression=8)`` with
    ``cosmos2.config.TokenizerConfig(spatial_compression=16)`` and use
    ``cosmos2.tokenizer.ContinuousVideoTokenizer`` which inherits this
    class but overrides the channel budget.  See cosmos2/tokenizer.py.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, cfg: TokenizerConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = TokenizerConfig()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor, sample: bool = False
    ) -> torch.Tensor:
        """Encode raw video to latent.

        Args:
            x:      (B, 3, T, H, W) in [-1, 1].
            sample: If True, sample from N(mean, exp(0.5*log_var));
                    otherwise return the mean (used at inference).

        Returns:
            Latent tensor (B, latent_channels, T', H', W').
        """
        mean, log_var = self.encoder(x)
        if sample:
            std = (0.5 * log_var).exp()
            return mean + std * torch.randn_like(std)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to video.

        Args:
            z: (B, latent_channels, T', H', W').

        Returns:
            (B, 3, T, H, W) in [-1, 1].
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-decode pass for training (returns recon + KL terms).

        Returns:
            recon   : reconstructed video
            mean    : encoder mean
            log_var : encoder log-variance
        """
        mean, log_var = self.encoder(x)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(mean)
        recon = self.decoder(z)
        return recon, mean, log_var

    @property
    def compression_t(self) -> int:
        return self.cfg.temporal_compression

    @property
    def compression_s(self) -> int:
        return self.cfg.spatial_compression
