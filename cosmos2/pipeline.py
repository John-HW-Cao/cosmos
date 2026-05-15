"""
Cosmos 2.5 Unified Pipeline.

Adapted from cosmos1/pipeline.py.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
REMOVED  : Text2WorldPipeline, Image2WorldPipeline, Video2WorldPipeline
           All three task-specific pipeline classes are replaced by a
           single UnifiedPipeline.

ADDED    : UnifiedPipeline — dispatches based on which conditioning inputs
           are present (text only / text+image / text+video).

CHANGED  : Pipeline.run() no longer accepts a ``task`` argument.
           The pipeline auto-detects the task from the inputs:
               no image_cond, no video_cond → text-to-world
               image_cond provided           → image-to-world
               video_cond provided           → video-to-world

CHANGED  : Conditioning mechanism for image/video
           Cosmos 1 : prepends conditioning tokens to the sequence
                      (handled inside the DiT forward pass).
           Cosmos 2.5: passes the conditioning latent through
                       VisualPoolProjector → adaLN-Zero injection.
           The pipeline now simply passes ``image_cond`` to the DiT
           without any token manipulation.

CHANGED  : text encoder CosmosReasonEncoder (was T5TextEncoder)
CHANGED  : model        Cosmos2DiT          (was Cosmos1DiT)
CHANGED  : scheduler    DistilledRectifiedFlow (was RectifiedFlow)
CHANGED  : scheduler.sample() call — no ``task`` argument

UNCHANGED: _encode_text(), _decode_latent(), _latent_shape()
           GenerationOutput dataclass (task field now always "unified")
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import Cosmos25Config
from .tokenizer import ContinuousVideoTokenizerV2
from .dit import Cosmos2DiT
from .text_encoder import CosmosReasonEncoder
from .flow_matching import DistilledRectifiedFlow


# ---------------------------------------------------------------------------
# Output type (unchanged from Cosmos 1 except task default)
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """Result of a Cosmos 2.5 generation call."""

    video: torch.Tensor     # (B, 3, T, H, W) decoded video in [-1, 1]
    latents: torch.Tensor   # (B, C, T', H', W') final latent z_0
    task: str               # auto-detected task string


# ---------------------------------------------------------------------------
# CHANGED: single UnifiedPipeline (was 3 separate classes in Cosmos 1)
# ---------------------------------------------------------------------------

class UnifiedPipeline:
    """Cosmos 2.5 unified video generation pipeline.

    Handles Text-to-World, Image-to-World, and Video-to-World generation
    through a single interface.  The appropriate conditioning path is
    selected automatically based on which arguments are provided.

    Cosmos 1 had three separate classes for these tasks.
    Cosmos 2.5 merges them here.

    Usage::

        pipe = UnifiedPipeline()

        # Text-to-world (auto-detected)
        out = pipe.run(prompt_ids=..., prompt_mask=..., T=121, H=320, W=512)

        # Image-to-world (auto-detected)
        out = pipe.run(prompt_ids=..., prompt_mask=..., image_cond=image, ...)

        # Video-to-world (auto-detected)
        out = pipe.run(prompt_ids=..., prompt_mask=..., video_cond=video, ...)
    """

    def __init__(
        self,
        cfg: Cosmos25Config | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or Cosmos25Config()
        self.device = torch.device(device)

        # CHANGED: ContinuousVideoTokenizerV2 (4×16×16, was 4×8×8)
        self.tokenizer    = ContinuousVideoTokenizerV2(self.cfg.tokenizer).to(self.device)
        # CHANGED: CosmosReasonEncoder (was T5TextEncoder)
        self.text_encoder = CosmosReasonEncoder(
            hidden_dim=self.cfg.dit.cond_dim
        ).to(self.device)
        # CHANGED: Cosmos2DiT (was Cosmos1DiT)
        self.dit          = Cosmos2DiT(self.cfg.dit).to(self.device)
        # CHANGED: DistilledRectifiedFlow (was RectifiedFlow); 35→20 steps
        self.scheduler    = DistilledRectifiedFlow(
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        )

        # Freeze tokenizer and text encoder — UNCHANGED logic.
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers (UNCHANGED from Cosmos 1 _BasePipeline)
    # ------------------------------------------------------------------

    def _encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_encoder.encode(input_ids, attention_mask)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(z)

    def _latent_shape(self, T: int, H: int, W: int, B: int = 1) -> tuple[int, ...]:
        C  = self.cfg.tokenizer.latent_channels
        Tp = T // self.cfg.tokenizer.temporal_compression
        # CHANGED: spatial_compression 8 → 16
        Hp = H // self.cfg.tokenizer.spatial_compression
        Wp = W // self.cfg.tokenizer.spatial_compression
        return (B, C, Tp, Hp, Wp)

    # ------------------------------------------------------------------
    # CHANGED: single run() method (was three separate .run() methods)
    # ------------------------------------------------------------------

    def run(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        neg_prompt_ids: Optional[torch.Tensor] = None,
        neg_prompt_mask: Optional[torch.Tensor] = None,
        # CHANGED: image_cond / video_cond replace separate pipeline classes
        image_cond: Optional[torch.Tensor] = None,
        video_cond: Optional[torch.Tensor] = None,
        T: int = 121,           # CHANGED default (was 57; supports ~30 s)
        H: int = 320,
        W: int = 512,
        B: int = 1,
        seed: Optional[int] = None,
    ) -> GenerationOutput:
        """Generate a video using the unified Cosmos 2.5 model.

        Args:
            prompt_ids/mask    : Tokenized text prompt.
            neg_prompt_ids/mask: Tokenized negative prompt for CFG.
            image_cond         : Reference image latent (B, C, 1, H', W').
                                 If provided → image-to-world mode.
            video_cond         : Conditioning video latent (B, C, T'', H', W').
                                 If provided → video-to-world mode.
            T, H, W            : Desired output video dimensions.
            B                  : Batch size.
            seed               : Random seed.

        Returns:
            GenerationOutput with decoded video.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids  = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)

        text_seq, text_pool = self._encode_text(prompt_ids, prompt_mask)

        # Unconditional embeddings for CFG — UNCHANGED logic.
        text_seq_u = text_pool_u = None
        if neg_prompt_ids is not None:
            neg_ids  = neg_prompt_ids.to(self.device)
            neg_mask = neg_prompt_mask.to(self.device)
            text_seq_u, text_pool_u = self._encode_text(neg_ids, neg_mask)

        # ── CHANGED: unified conditioning (no task-specific branches) ──
        # Determine which conditioning modality to use.
        cond_latent: Optional[torch.Tensor] = None
        if image_cond is not None:
            # Image-to-world: encode the reference image.
            cond_latent = self.tokenizer.encode(image_cond.to(self.device))
            task_str = "image2world"
        elif video_cond is not None:
            # Video-to-world: encode the conditioning prefix video.
            cond_latent = self.tokenizer.encode(video_cond.to(self.device))
            task_str = "video2world"
        else:
            task_str = "text2world"

        latent_shape = self._latent_shape(T, H, W, B)

        # CHANGED: scheduler.sample() — no ``task`` argument (unified model)
        z0 = self.scheduler.sample(
            model=self.dit,
            shape=latent_shape,
            text_seq=text_seq,
            text_pool=text_pool,
            text_seq_uncond=text_seq_u,
            text_pool_uncond=text_pool_u,
            # REMOVED: task="text2world" / "image2world" / "video2world"
            image_cond=cond_latent,           # unified conditioning argument
            device=self.device,
        )

        video = self._decode_latent(z0)
        return GenerationOutput(video=video, latents=z0, task=task_str)
