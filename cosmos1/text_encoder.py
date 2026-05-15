"""
Cosmos 1 T5-XXL Text Encoder.

Text prompts are encoded with a frozen T5-XXL (4096-d) model.  The encoder
produces a sequence of token embeddings that are injected into the DiT via
cross-attention; a pooled (mean-aggregated) embedding is also extracted and
fed into the adaLN-Zero conditioning module.

── Cosmos 2.5 change ────────────────────────────────────────────────────
T5-XXL is replaced with **Cosmos Reason 1**, an NVIDIA-internal vision-
language model (VLM) that produces 5120-dimensional token embeddings.

Key code differences (Cosmos 1 → 2.5):
  REMOVED  : T5 encoder loading (``transformers.T5EncoderModel``)
  REMOVED  : ``hidden_size = 4096`` (T5-XXL dimension)
  ADDED    : CosmosReasonEncoder wrapper (cosmos2/text_encoder.py)
  ADDED    : ``hidden_size = 5120`` (VLM output dimension)
  CHANGED  : cond_dim in DiTConfig 4096 → 5120
  UNCHANGED: text projection head (linear dim → hidden_dim)
  UNCHANGED: frozen-encoder logic (both are frozen during DiT training)

The pooled embedding used for adaLN-Zero conditioning is the mean of
non-padding tokens in both Cosmos 1 and 2.5.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn


class T5TextEncoder(nn.Module):
    """Frozen T5-XXL text encoder for Cosmos 1.

    Wraps HuggingFace ``T5EncoderModel`` and exposes a simple
    ``encode(text_tokens, attention_mask) → (sequence, pooled)`` API.

    In training, this module is frozen (``requires_grad = False``).
    The downstream DiT only learns a linear projection head on top of
    the T5 embeddings.

    Args:
        model_name : HuggingFace model identifier for T5.
                     Default matches the Cosmos 1 paper ("google/t5-v1_1-xxl").
        hidden_dim : Target hidden dimension of the DiT.  A linear layer
                     projects T5's 4096-d output to this dimension.
        max_length : Maximum tokenized prompt length.
    """

    T5_HIDDEN: int = 4096  # T5-XXL output dimension

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-xxl",
        hidden_dim: int = 4096,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._encoder: nn.Module | None = None  # lazy-loaded

        # Project T5's 4096-d embeddings to the DiT's hidden_dim.
        # When hidden_dim == T5_HIDDEN (default Cosmos 1), this is an
        # identity-like linear.
        self.proj = nn.Linear(self.T5_HIDDEN, hidden_dim, bias=False)

    # ------------------------------------------------------------------
    # Lazy model loading (avoids importing heavy transformers at import-time)
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        """Load T5 encoder and freeze its weights."""
        try:
            from transformers import T5EncoderModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for T5TextEncoder. "
                "Install it with: pip install transformers"
            ) from exc
        self._encoder = T5EncoderModel.from_pretrained(self.model_name)
        for p in self._encoder.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokenized text to embeddings.

        Args:
            input_ids     : (B, L) integer token IDs.
            attention_mask: (B, L) binary mask (1 = real token, 0 = pad).

        Returns:
            seq_emb  : (B, L, hidden_dim) per-token embeddings (projected).
            pool_emb : (B, hidden_dim) mean-pooled embedding over real tokens.
        """
        if self._encoder is None:
            self._load_encoder()

        # Run frozen T5 encoder.
        with torch.inference_mode():
            outputs = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # last_hidden_state: (B, L, 4096)
        seq = outputs.last_hidden_state

        # Project to DiT hidden_dim.
        seq_emb = self.proj(seq)  # (B, L, hidden_dim)

        # Mean-pool over non-padding tokens.
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pool_emb = (seq_emb * mask).sum(1) / mask.sum(1).clamp(min=1)

        return seq_emb, pool_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(input_ids, attention_mask)
