"""
Cosmos 2.5 Continuous Video Tokenizer (4×16×16).

Adapted from cosmos1/tokenizer.py.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
CHANGED : spatial_compression  8 → 16
          Achieved by adding one extra SpatialDownsample / SpatialUpsample
          stage.  The Encoder and Decoder loops already handle variable
          numbers of stages via ``channel_multipliers``; only the config
          changes:
              Cosmos 1 : channel_multipliers = (1, 2, 4, 8)     → 3 spatial stages
              Cosmos 2.5: channel_multipliers = (1, 2, 4, 4, 8)  → 4 spatial stages

CHANGED : base_channels  128 → 192
          Wider channel budget per stage.

UNCHANGED : temporal_compression (4×)
            Encoder/Decoder class bodies are completely unchanged.
            The change is purely config-driven.

This module re-exports ``ContinuousVideoTokenizer`` under the alias
``ContinuousVideoTokenizerV2`` so that users can see at a glance which
version they are using.  The class itself only overrides ``__init__``
to inject the Cosmos 2.5 TokenizerConfig.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from ..cosmos1.tokenizer import ContinuousVideoTokenizer
from .config import TokenizerConfig as TokenizerConfig25


class ContinuousVideoTokenizerV2(ContinuousVideoTokenizer):
    """Cosmos 2.5 video tokenizer (4×16×16 compression).

    Inherits *all* implementation from Cosmos 1's ContinuousVideoTokenizer.
    The only difference is the default config:

        Cosmos 1 : TokenizerConfig(spatial_compression=8,  base_channels=128,
                                   channel_multipliers=(1,2,4,8))
        Cosmos 2.5: TokenizerConfig(spatial_compression=16, base_channels=192,
                                    channel_multipliers=(1,2,4,4,8))

    This demonstrates that the Cosmos 1 → 2.5 tokenizer adaptation is
    purely a configuration change, not a code change, because the Encoder
    and Decoder were already written to handle variable numbers of stages.

    Usage::

        tokenizer = ContinuousVideoTokenizerV2()
        z = tokenizer.encode(video)   # (B, 16, T/4, H/16, W/16)
        recon = tokenizer.decode(z)
    """

    def __init__(self, cfg: TokenizerConfig25 | None = None) -> None:
        # Pass Cosmos 2.5 config; Cosmos 1 parent does the rest.
        super().__init__(cfg=cfg or TokenizerConfig25())
