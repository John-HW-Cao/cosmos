"""
Cosmos 2.5 text encoder: Cosmos Reason VLM.

Adapted from cosmos1/text_encoder.py.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
REMOVED  : T5EncoderModel (HuggingFace google/t5-v1_1-xxl)
REMOVED  : T5_HIDDEN = 4096
ADDED    : CosmosReasonEncoder — wraps NVIDIA's Cosmos Reason 1 VLM
ADDED    : VLM_HIDDEN = 5120   (VLM output dimension)
CHANGED  : proj linear: 4096 → 5120  (matches the larger VLM embedding)
CHANGED  : cond_dim in DiTConfig: 4096 → 5120

Why the change?
    Cosmos Reason 1 is a vision-language model that jointly processes text
    and visual inputs, producing better-grounded prompts for physical AI
    scenarios (robotics, simulation, AV).  Its richer 5120-d representation
    provides more semantic information to the adaLN-Zero conditioning module
    than T5 alone.

UNCHANGED : frozen-encoder logic
UNCHANGED : mean-pooling over non-padding tokens
UNCHANGED : encode() / forward() API signature (except hidden_dim default)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CosmosReasonEncoder(nn.Module):
    """Frozen Cosmos Reason 1 VLM text encoder for Cosmos 2.5.

    Cosmos Reason 1 is a vision-language model that produces higher-quality
    text embeddings than T5-XXL, particularly for physically grounded
    descriptions.  Its output dimension is 5120 (vs T5's 4096).

    The API is intentionally identical to ``T5TextEncoder`` so that
    existing training code only needs to swap out the encoder object.

    Args:
        model_name : HuggingFace / local path to the Cosmos Reason model.
        hidden_dim : Target DiT hidden dimension.  A linear projection maps
                     VLM_HIDDEN (5120) to this dimension.
        max_length : Maximum tokenized prompt length.
    """

    VLM_HIDDEN: int = 5120  # CHANGED from T5_HIDDEN = 4096 in Cosmos 1

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason1-7B",
        hidden_dim: int = 2048,               # CHANGED default (was 4096)
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._encoder: nn.Module | None = None

        # Project VLM's 5120-d output to the DiT's hidden_dim.
        # CHANGED: in_features 4096 → 5120 (VLM output dimension)
        self.proj = nn.Linear(self.VLM_HIDDEN, hidden_dim, bias=False)

    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        """Load Cosmos Reason VLM and freeze weights.

        CHANGED vs Cosmos 1: loads an AutoModel (VLM) rather than T5EncoderModel.
        The rest of the freezing logic is UNCHANGED.
        """
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for CosmosReasonEncoder. "
                "Install it with: pip install transformers"
            ) from exc

        # CHANGED: AutoModel instead of T5EncoderModel
        self._encoder = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # Freeze all parameters — identical logic as Cosmos 1.
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

        API is UNCHANGED vs T5TextEncoder.encode():
            input_ids, attention_mask → (seq_emb, pool_emb)

        Internal change: output dimension 4096 → 5120.

        Args:
            input_ids     : (B, L) integer token IDs.
            attention_mask: (B, L) binary mask.

        Returns:
            seq_emb  : (B, L, hidden_dim) per-token embeddings (projected).
            pool_emb : (B, hidden_dim) mean-pooled embedding.
        """
        if self._encoder is None:
            self._load_encoder()

        with torch.inference_mode():
            outputs = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Use the last hidden state: (B, L, VLM_HIDDEN)
        # CHANGED: last_hidden_state dim is 5120 (was 4096 in Cosmos 1)
        seq = outputs.last_hidden_state

        seq_emb = self.proj(seq)  # (B, L, hidden_dim)

        # Mean-pool — UNCHANGED logic from Cosmos 1.
        mask = attention_mask.unsqueeze(-1).float()
        pool_emb = (seq_emb * mask).sum(1) / mask.sum(1).clamp(min=1)

        return seq_emb, pool_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(input_ids, attention_mask)
