"""
Cosmos: NVIDIA World Foundation Model implementations.

This package contains reference implementations of:

    cosmos1  — Cosmos 1.0 video generation (DiT + CV8x8x8 tokenizer)
    cosmos2  — Cosmos 2.5 video generation (adapted from Cosmos 1)
    shared   — Building blocks shared by both generations

Quick-start
───────────
from cosmos.cosmos1.pipeline import Text2WorldPipeline
from cosmos.cosmos2.pipeline import UnifiedPipeline

# Cosmos 1 — separate pipelines per task
pipe1 = Text2WorldPipeline()

# Cosmos 2.5 — single unified pipeline
pipe2 = UnifiedPipeline()
"""

from . import cosmos1
from . import cosmos2
from . import shared

__version__ = "0.1.0"
__all__ = ["cosmos1", "cosmos2", "shared"]
