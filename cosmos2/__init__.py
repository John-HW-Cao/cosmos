"""
Cosmos 2.5 video generation model.

Adapted from Cosmos 1 with the following key changes:

    config          – updated spatial compression (8→16), longer videos,
                      distilled inference (35→20 steps), unified conditioning
    tokenizer       – 4×16×16 compression; wider channel budget
    attention       – chunked space-time attention (O(C·N) vs O(N²))
    dit             – unified DiT with multi-modal adaLN-Zero and single head
    text_encoder    – Cosmos Reason VLM encoder (5120-d) replacing T5-XXL
    flow_matching   – distilled rectified flow with RL reward weighting
    pipeline        – single UnifiedPipeline replacing three separate ones

Every module documents exactly what changed relative to the Cosmos 1
counterpart.  Look for the "── Cosmos 1 → 2.5 diff ──" sections.
"""

from .config import Cosmos25Config
from .tokenizer import ContinuousVideoTokenizerV2
from .attention import ChunkedSpaceTimeAttention
from .dit import Cosmos2DiT
from .text_encoder import CosmosReasonEncoder
from .flow_matching import DistilledRectifiedFlow
from .pipeline import UnifiedPipeline

__all__ = [
    "Cosmos25Config",
    "ContinuousVideoTokenizerV2",
    "ChunkedSpaceTimeAttention",
    "Cosmos2DiT",
    "CosmosReasonEncoder",
    "DistilledRectifiedFlow",
    "UnifiedPipeline",
]
