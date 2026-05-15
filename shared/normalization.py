"""
Normalization layers shared by Cosmos 1 and Cosmos 2.5.

Both generations use RMSNorm as the primary layer normalization.
AdaLNZero (adaptive layer norm with zero-init) is the conditioning
mechanism in the DiT blocks.

Cosmos 2.5 extends AdaLNZero with an additional *scale_shift* branch
for multi-modal conditioning; that extension lives in cosmos2/dit.py.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used in every transformer block in both Cosmos 1 and Cosmos 2.5.

    Unlike LayerNorm, RMSNorm omits the mean-centering step and only
    scales by the root mean square of the activations, which is slightly
    cheaper and empirically equivalent in quality.

    Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable per-channel scale (no bias — omitted intentionally).
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over the last dimension.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with Zero Initialization (adaLN-Zero).

    Introduced in the DiT paper (Peebles & Xie, 2022), adaLN-Zero injects
    time-step (and optionally text) conditioning into every transformer
    block by predicting six scalar parameters:

        shift_msa, scale_msa, gate_msa,   # modulate self-attention
        shift_mlp, scale_mlp, gate_mlp    # modulate feed-forward block

    The gate_* terms are initialized to zero so that residual branches
    start as identity at the beginning of training.

    ── Cosmos 1 vs Cosmos 2.5 ──────────────────────────────────────────
    Cosmos 1  : conditions on *time embedding* + *text pooled embedding*.
    Cosmos 2.5: conditions on *time* + *text* + *image/video* embeddings
                (multi-modal adaLN); the number of output parameters
                stays the same but the input projection is wider.
                See cosmos2/dit.py → AdaLNZeroMultiModal for the update.
    ────────────────────────────────────────────────────────────────────
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        # Projects conditioning vector to 6 modulation parameters.
        self.proj = nn.Linear(cond_dim, 6 * dim, bias=True)
        # Zero-initialize projection weights and biases so gates start at 0.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the 6 modulation parameters for the current block.

        Args:
            x:    Input feature tensor  (B, T, D).
            cond: Conditioning vector   (B, cond_dim).

        Returns:
            Tuple of (shift_msa, scale_msa, gate_msa,
                      shift_mlp, scale_mlp, gate_mlp).
            All tensors have shape (B, 1, D).
        """
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 6*D)
        chunks = params.chunk(6, dim=-1)        # 6 × (B, 1, D)
        x_normed = self.norm(x)
        # Apply scale and shift to the normalized input.
        x_modulated = x_normed * (1 + chunks[1]) + chunks[0]
        return (x_modulated, chunks[2],   # msa branch
                x_normed,    chunks[3], chunks[4], chunks[5])  # mlp branch
