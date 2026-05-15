"""
Activation functions shared by Cosmos 1 and Cosmos 2.5.

Both generations use SwiGLU inside feed-forward blocks.
Cosmos 2.5 retains the same activations (no changes needed here).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation: x * swish(gate).

    Used in the feed-forward blocks of the Diffusion Transformer for
    both Cosmos 1 and Cosmos 2.5.

    Formula:
        SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)

    Reference: Noam Shazeer, "GLU Variants Improve Transformer" (2020).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along the last dimension: half goes through SiLU, half is gate.
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class GELU(nn.Module):
    """Standard GELU activation.

    Provided as a drop-in alternative for ablation experiments.
    Not used in the default Cosmos 1 or 2.5 configurations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
