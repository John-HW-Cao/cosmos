"""
Shared building blocks used by both Cosmos 1 and Cosmos 2.5.

These modules contain primitives (activations, normalizations, embeddings)
that are identical—or nearly so—across both generations.  Where Cosmos 2.5
deviates from Cosmos 1, the change is documented in the cosmos2/ override.
"""

from .activations import SwiGLU, GELU
from .normalization import RMSNorm, AdaLNZero
from .embeddings import SinusoidalPosEmbed, RoPE2D, RoPE3D

__all__ = [
    "SwiGLU",
    "GELU",
    "RMSNorm",
    "AdaLNZero",
    "SinusoidalPosEmbed",
    "RoPE2D",
    "RoPE3D",
]
