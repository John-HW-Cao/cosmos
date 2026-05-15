"""
Cosmos 1 hyper-parameter configuration.

Every architectural constant for Cosmos 1 is defined here so that the
changes made in Cosmos 2.5 are easy to diff in cosmos2/config.py.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TokenizerConfig:
    """Configuration for the Cosmos 1 Continuous Video Tokenizer (CV8x8x8).

    The tokenizer compresses raw video along three axes:
        temporal : 4× (every 4 input frames → 1 latent frame)
        spatial_h: 8× (height dimension)
        spatial_w: 8× (width dimension)

    Total compression: 4 × 8 × 8 = 256×  (spatial-temporal)

    ── Cosmos 2.5 change ────────────────────────────────────────────────
    Cosmos 2.5 uses a 4×16×16 tokenizer (spatial compression doubles to
    16×).  See cosmos2/config.py → TokenizerConfig for the update.
    ─────────────────────────────────────────────────────────────────────
    """

    in_channels: int = 3          # RGB input
    latent_channels: int = 16     # latent space channels
    temporal_compression: int = 4
    spatial_compression: int = 8  # ← Cosmos 2.5 changes this to 16
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)


@dataclass
class DiTConfig:
    """Configuration for the Cosmos 1 Diffusion Transformer.

    ── Key Cosmos 1 architectural choices ──────────────────────────────
    • hidden_dim    : 4096  (Cosmos 1 – 7B parameter variant)
    • num_layers    : 28
    • num_heads     : 16
    • head_dim      : 256
    • attention     : FullSpaceTimeAttention — all tokens attend to all
                      tokens in a single pass (O(N²) memory).
    • text_encoder  : T5-XXL (4096-d pooled text embedding)
    • cond_dim      : 4096  (matches hidden_dim; fed through adaLN-Zero)
    • max_frames    : 57    (≈ 4 s video at 14 fps after 4× pooling)
    • max_h / max_w : 40 / 64 spatial latent tokens (320×512 native)

    ── Cosmos 2.5 changes ──────────────────────────────────────────────
    See cosmos2/config.py → DiTConfig for the full diff.  Summary:
      • hidden_dim  reduced to 2048 (2B variant) or kept 4096 (14B)
      • attention   replaced with ChunkedSpaceTimeAttention
      • text_encoder replaced with CosmosReasonEncoder (VLM-based)
      • cond_dim    increased to 5120 (VLM output dimension)
      • max_frames  increased to 121 (≈ 30 s video)
      • unified_conditioning = True  (single model for all tasks)
    ─────────────────────────────────────────────────────────────────────
    """

    hidden_dim: int = 4096
    num_layers: int = 28
    num_heads: int = 16
    head_dim: int = 256  # must be divisible by 6 for 3-D RoPE (T+H+W each get head_dim/3)
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Conditioning
    cond_dim: int = 4096   # ← Cosmos 2.5 changes to 5120
    time_embed_dim: int = 256

    # Spatial-temporal extent
    max_frames: int = 57   # ← Cosmos 2.5 changes to 121
    max_h: int = 40
    max_w: int = 64

    # Attention type name — referenced in DiT factory functions.
    attention_type: str = "full"  # ← Cosmos 2.5 changes to "chunked"

    # Cosmos 1: separate task heads; 2.5 merges into one unified model.
    unified_conditioning: bool = False  # ← Cosmos 2.5 sets this to True


@dataclass
class Cosmos1Config:
    """Top-level Cosmos 1 configuration bundling tokenizer and DiT."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)

    # Inference defaults
    num_inference_steps: int = 35   # ← Cosmos 2.5 reduces to 20 (distilled)
    guidance_scale: float = 7.0
    seed: int = 42

    # Training defaults
    learning_rate: float = 1e-4
    use_rl_posttraining: bool = False  # ← Cosmos 2.5 sets this to True
