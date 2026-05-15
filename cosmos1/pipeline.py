"""
Cosmos 1 inference pipelines.

Cosmos 1 exposes **three separate** pipeline classes, one per task:

    Text2WorldPipeline  — text prompt → video
    Image2WorldPipeline — text + reference image → video
    Video2WorldPipeline — text + conditioning video frames → video extension

This mirrors the Cosmos 1 model design where three task-specific output
heads live inside a shared DiT backbone.

── Cosmos 2.5 change ────────────────────────────────────────────────────
All three pipelines are collapsed into a single ``UnifiedPipeline`` that
dispatches based on which inputs are provided.  The ``task`` string and
per-task head selection are removed entirely.

Key code differences (Cosmos 1 → 2.5):
  REMOVED   : Text2WorldPipeline, Image2WorldPipeline, Video2WorldPipeline
  ADDED     : UnifiedPipeline (see cosmos2/pipeline.py)
  CHANGED   : Pipeline.run() no longer accepts ``task`` argument
  CHANGED   : Conditioning (image / video) handled by multi-modal adaLN-Zero
              instead of token prepending
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import Cosmos1Config
from .tokenizer import ContinuousVideoTokenizer
from .dit import Cosmos1DiT
from .text_encoder import T5TextEncoder
from .flow_matching import RectifiedFlow


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """Result of a Cosmos 1 generation call."""

    video: torch.Tensor          # (B, 3, T, H, W) decoded video in [-1, 1]
    latents: torch.Tensor        # (B, C, T', H', W') final latent z_0
    task: str                    # "text2world" | "image2world" | "video2world"


class _BasePipeline:
    """Internal base class shared by all Cosmos 1 pipelines.

    Holds references to the frozen tokenizer and text encoder, and the
    trainable DiT + flow matching modules.
    """

    def __init__(
        self,
        cfg: Cosmos1Config | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or Cosmos1Config()
        self.device = torch.device(device)

        self.tokenizer    = ContinuousVideoTokenizer(self.cfg.tokenizer).to(self.device)
        self.text_encoder = T5TextEncoder(
            hidden_dim=self.cfg.dit.cond_dim
        ).to(self.device)
        self.dit          = Cosmos1DiT(self.cfg.dit).to(self.device)
        self.scheduler    = RectifiedFlow(
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        )

        # Freeze tokenizer and text encoder.
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def _encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_encoder.encode(input_ids, attention_mask)

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(z)

    def _latent_shape(self, T: int, H: int, W: int, B: int = 1) -> tuple[int, ...]:
        C = self.cfg.tokenizer.latent_channels
        Tp = T  // self.cfg.tokenizer.temporal_compression
        Hp = H  // self.cfg.tokenizer.spatial_compression
        Wp = W  // self.cfg.tokenizer.spatial_compression
        return (B, C, Tp, Hp, Wp)


# ---------------------------------------------------------------------------
# Text2World Pipeline
# ---------------------------------------------------------------------------

class Text2WorldPipeline(_BasePipeline):
    """Cosmos 1 Text-to-Video generation pipeline.

    Generates a video from a text prompt alone.

    Usage::

        pipeline = Text2WorldPipeline()
        output = pipeline.run(
            prompt_ids=...,
            prompt_mask=...,
            T=57, H=320, W=512,
        )
        video = output.video  # (1, 3, 57, 320, 512)

    ── Cosmos 2.5 adaptation ─────────────────────────────────────────
    This class is removed.  Use cosmos2.pipeline.UnifiedPipeline with
    only text inputs.
    ─────────────────────────────────────────────────────────────────
    """

    def run(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        neg_prompt_ids: Optional[torch.Tensor] = None,
        neg_prompt_mask: Optional[torch.Tensor] = None,
        T: int = 57,
        H: int = 320,
        W: int = 512,
        B: int = 1,
        seed: Optional[int] = None,
    ) -> GenerationOutput:
        """Generate a video from a text prompt.

        Args:
            prompt_ids/mask    : Tokenized text prompt.
            neg_prompt_ids/mask: Tokenized negative prompt for CFG.
            T, H, W            : Desired video dimensions (before tokenizer compression).
            B                  : Batch size.
            seed               : Random seed.

        Returns:
            GenerationOutput with decoded video tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids   = prompt_ids.to(self.device)
        prompt_mask  = prompt_mask.to(self.device)

        text_seq, text_pool = self._encode_text(prompt_ids, prompt_mask)

        # Unconditional embeddings for CFG.
        text_seq_u = text_pool_u = None
        if neg_prompt_ids is not None:
            neg_ids  = neg_prompt_ids.to(self.device)
            neg_mask = neg_prompt_mask.to(self.device)
            text_seq_u, text_pool_u = self._encode_text(neg_ids, neg_mask)

        latent_shape = self._latent_shape(T, H, W, B)
        z0 = self.scheduler.sample(
            model=self.dit,
            shape=latent_shape,
            text_seq=text_seq,
            text_pool=text_pool,
            text_seq_uncond=text_seq_u,
            text_pool_uncond=text_pool_u,
            task="text2world",    # ← removed in Cosmos 2.5
            device=self.device,
        )
        video = self._decode_latent(z0)
        return GenerationOutput(video=video, latents=z0, task="text2world")


# ---------------------------------------------------------------------------
# Image2World Pipeline
# ---------------------------------------------------------------------------

class Image2WorldPipeline(_BasePipeline):
    """Cosmos 1 Image-to-Video generation pipeline.

    Generates a video conditioned on a reference image and a text prompt.

    ── Cosmos 2.5 adaptation ─────────────────────────────────────────
    This class is removed.  Use cosmos2.pipeline.UnifiedPipeline with
    an ``image_cond`` argument.
    ─────────────────────────────────────────────────────────────────
    """

    def run(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        image: torch.Tensor,
        neg_prompt_ids: Optional[torch.Tensor] = None,
        neg_prompt_mask: Optional[torch.Tensor] = None,
        T: int = 57,
        H: int = 320,
        W: int = 512,
        B: int = 1,
        seed: Optional[int] = None,
    ) -> GenerationOutput:
        """Generate a video from an image + text prompt.

        Args:
            prompt_ids/mask : Tokenized text prompt.
            image           : Reference image (B, 3, 1, H, W) in [-1, 1].
            T, H, W         : Desired video dimensions.

        Returns:
            GenerationOutput.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids  = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        image       = image.to(self.device)

        text_seq, text_pool = self._encode_text(prompt_ids, prompt_mask)

        # Encode the reference image to latent space.
        image_latent = self.tokenizer.encode(image)  # (B, C, 1, H', W')

        text_seq_u = text_pool_u = None
        if neg_prompt_ids is not None:
            neg_ids  = neg_prompt_ids.to(self.device)
            neg_mask = neg_prompt_mask.to(self.device)
            text_seq_u, text_pool_u = self._encode_text(neg_ids, neg_mask)

        latent_shape = self._latent_shape(T, H, W, B)
        z0 = self.scheduler.sample(
            model=self.dit,
            shape=latent_shape,
            text_seq=text_seq,
            text_pool=text_pool,
            text_seq_uncond=text_seq_u,
            text_pool_uncond=text_pool_u,
            task="image2world",   # ← task-specific head; removed in Cosmos 2.5
            image_cond=image_latent,
            device=self.device,
        )
        video = self._decode_latent(z0)
        return GenerationOutput(video=video, latents=z0, task="image2world")


# ---------------------------------------------------------------------------
# Video2World Pipeline
# ---------------------------------------------------------------------------

class Video2WorldPipeline(_BasePipeline):
    """Cosmos 1 Video-to-Video (continuation / extension) pipeline.

    Generates a video continuation conditioned on a prefix video + text.

    ── Cosmos 2.5 adaptation ─────────────────────────────────────────
    This class is removed.  Use cosmos2.pipeline.UnifiedPipeline with
    a ``video_cond`` argument.
    ─────────────────────────────────────────────────────────────────
    """

    def run(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        cond_video: torch.Tensor,
        neg_prompt_ids: Optional[torch.Tensor] = None,
        neg_prompt_mask: Optional[torch.Tensor] = None,
        T: int = 57,
        H: int = 320,
        W: int = 512,
        B: int = 1,
        seed: Optional[int] = None,
    ) -> GenerationOutput:
        """Generate a video continuation.

        Args:
            prompt_ids/mask: Tokenized text prompt.
            cond_video     : Conditioning prefix video (B, 3, T_c, H, W).
            T, H, W        : Desired video dimensions (generated portion).

        Returns:
            GenerationOutput.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids  = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        cond_video  = cond_video.to(self.device)

        text_seq, text_pool = self._encode_text(prompt_ids, prompt_mask)

        # Encode conditioning video to latent space.
        video_latent = self.tokenizer.encode(cond_video)

        text_seq_u = text_pool_u = None
        if neg_prompt_ids is not None:
            neg_ids  = neg_prompt_ids.to(self.device)
            neg_mask = neg_prompt_mask.to(self.device)
            text_seq_u, text_pool_u = self._encode_text(neg_ids, neg_mask)

        latent_shape = self._latent_shape(T, H, W, B)
        z0 = self.scheduler.sample(
            model=self.dit,
            shape=latent_shape,
            text_seq=text_seq,
            text_pool=text_pool,
            text_seq_uncond=text_seq_u,
            text_pool_uncond=text_pool_u,
            task="video2world",   # ← task-specific head; removed in Cosmos 2.5
            image_cond=video_latent,
            device=self.device,
        )
        video = self._decode_latent(z0)
        return GenerationOutput(video=video, latents=z0, task="video2world")
