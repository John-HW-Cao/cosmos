"""
Cosmos 2.5 hyper-parameter configuration.

This file is directly derived from cosmos1/config.py.
Every line that differs from Cosmos 1 is annotated with:
    # CHANGED  – value was updated
    # ADDED    – new field not present in Cosmos 1
    # REMOVED  – field is no longer used (kept as comment for clarity)

Unchanged fields are reproduced verbatim to make the diff self-contained.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TokenizerConfig:
    """Cosmos 2.5 video tokenizer configuration.

    ── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────
    CHANGED : spatial_compression  8  → 16
              Doubles the spatial compression ratio, halving the number of
              latent tokens per frame and making the DiT 4× cheaper
              (N_tokens = T' × (H/16) × (W/16) vs T' × (H/8) × (W/8)).

    CHANGED : base_channels        128 → 192
              Wider channel budget compensates for higher compression.

    CHANGED : channel_multipliers  (1,2,4,8) → (1,2,4,4,8)
              An extra resolution stage is inserted to bridge the 2× extra
              downsampling without a quality drop.

    UNCHANGED : in_channels, latent_channels, temporal_compression,
                num_res_blocks, attn_resolutions
    ─────────────────────────────────────────────────────────────────────
    """

    in_channels: int = 3
    latent_channels: int = 16
    temporal_compression: int = 4
    spatial_compression: int = 16             # CHANGED (was 8 in Cosmos 1)
    base_channels: int = 192                  # CHANGED (was 128 in Cosmos 1)
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4, 8)  # CHANGED (was (1,2,4,8))
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)


@dataclass
class DiTConfig:
    """Cosmos 2.5 Diffusion Transformer configuration.

    ── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────
    CHANGED : hidden_dim          4096 → 2048  (2B variant; 14B keeps 4096)
              The 2B edge variant reduces the hidden dimension to make the
              model deployable on edge GPUs.

    CHANGED : num_layers          28   → 24
              Fewer layers for the 2B variant.

    CHANGED : num_heads           16   → 16   (unchanged, shown for clarity)

    CHANGED : head_dim            256  → 128  (matches hidden_dim / num_heads)

    CHANGED : cond_dim            4096 → 5120
              CosmosReason VLM produces 5120-d embeddings vs T5's 4096-d.

    CHANGED : max_frames          57   → 121
              Supports up to ~30 s of video at 14 fps (after 4× pooling).
              Only the RoPE3D ``max_frames`` constructor argument changes.

    CHANGED : attention_type      "full" → "chunked"
              Full O(N²) attention is replaced with chunked attention.
              See cosmos2/attention.py.

    CHANGED : unified_conditioning False → True
              Single model head replaces three task-specific heads.

    ADDED   : temporal_chunk_size 8
              Number of latent frames per attention chunk in
              ChunkedSpaceTimeAttention.

    ADDED   : cross_attn_every_n_layers 1
              Text cross-attention is applied every N DiT blocks
              (Cosmos 1 applied it once before the first block).

    UNCHANGED : mlp_ratio, dropout, time_embed_dim
    ─────────────────────────────────────────────────────────────────────
    """

    hidden_dim: int = 2048                    # CHANGED (was 4096 for 7B/14B)
    num_layers: int = 24                      # CHANGED (was 28)
    num_heads: int = 16                       # UNCHANGED
    head_dim: int = 128                       # CHANGED (was 256; = hidden_dim/num_heads); must be even for RoPE3D
    mlp_ratio: float = 4.0                    # UNCHANGED
    dropout: float = 0.0                      # UNCHANGED

    # Conditioning
    cond_dim: int = 5120                      # CHANGED (was 4096; matches VLM dim)
    time_embed_dim: int = 256                 # UNCHANGED

    # Spatial-temporal extent
    max_frames: int = 121                     # CHANGED (was 57; supports 30 s videos)
    max_h: int = 40                           # UNCHANGED
    max_w: int = 64                           # UNCHANGED

    # Attention configuration
    attention_type: str = "chunked"           # CHANGED (was "full")
    temporal_chunk_size: int = 8              # ADDED   (no equivalent in Cosmos 1)

    # Cross-attention cadence
    cross_attn_every_n_layers: int = 1        # ADDED   (was always 1 pre-block in C1)

    # Unified head
    unified_conditioning: bool = True         # CHANGED (was False in Cosmos 1)


@dataclass
class Cosmos25Config:
    """Top-level Cosmos 2.5 configuration.

    ── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────
    CHANGED : num_inference_steps  35 → 20   (distilled model)
    CHANGED : guidance_scale       7.0 → 6.0 (distilled models need lower CFG)
    ADDED   : use_rl_posttraining  True
    UNCHANGED : seed, learning_rate
    ─────────────────────────────────────────────────────────────────────
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)

    num_inference_steps: int = 20             # CHANGED (was 35 in Cosmos 1)
    guidance_scale: float = 6.0              # CHANGED (was 7.0 in Cosmos 1)
    seed: int = 42                            # UNCHANGED

    learning_rate: float = 1e-4               # UNCHANGED
    use_rl_posttraining: bool = True          # ADDED   (False in Cosmos 1)
    distillation_loss_weight: float = 0.5     # ADDED   (no equivalent in Cosmos 1)
