"""
Cosmos 2.5 Distilled Rectified Flow.

Adapted from cosmos1/flow_matching.py.

── Cosmos 1 → 2.5 diff ──────────────────────────────────────────────────
CHANGED  : num_inference_steps  35 → 20
           The Cosmos 2.5 model undergoes consistency distillation so
           fewer ODE steps are needed while maintaining sample quality.

CHANGED  : guidance_scale  7.0 → 6.0
           Distilled models require a lower CFG scale (stronger guidance
           causes over-saturation with fewer denoising steps).

ADDED    : distillation_loss()
           Computes the consistency distillation training objective
           that enables the 35→20 step reduction.
           Loss = MSE(v_θ(z_t, t), v_teacher(z_t, t).detach())
           Not present in Cosmos 1 (returns 0.0 if teacher is None).

ADDED    : reward_weight support in loss()
           Cosmos 2.5 uses RL post-training to improve prompt alignment.
           A per-sample reward weight is used to up-weight high-reward
           trajectories in the flow matching loss.
           In Cosmos 1 reward_weight was always None (dead code path).

UNCHANGED: add_noise() — forward noising formula is identical
UNCHANGED: Euler and Heun ODE solvers
UNCHANGED: CFG inference logic
UNCHANGED: sample() API
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..cosmos1.flow_matching import RectifiedFlow  # reuse Cosmos 1 base


class DistilledRectifiedFlow(RectifiedFlow):
    """Rectified Flow with consistency distillation for Cosmos 2.5.

    Inherits all forward-process and ODE-solver logic from Cosmos 1's
    RectifiedFlow.  Adds a distillation loss and activates RL reward
    weighting in the base class loss() method.

    Args:
        num_inference_steps   : ODE solver steps. CHANGED 35 → 20.
        guidance_scale        : CFG scale.        CHANGED 7.0 → 6.0.
        distillation_weight   : Weight for distillation loss term.
                                ADDED (no equivalent in Cosmos 1).
        sigma_min             : Minimum noise level. UNCHANGED.
    """

    def __init__(
        self,
        num_inference_steps: int = 20,          # CHANGED (was 35)
        guidance_scale: float = 6.0,            # CHANGED (was 7.0)
        distillation_weight: float = 0.5,       # ADDED
        sigma_min: float = 0.002,               # UNCHANGED
    ) -> None:
        super().__init__(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sigma_min=sigma_min,
        )
        self.distillation_weight = distillation_weight  # ADDED

    # ------------------------------------------------------------------
    # UNCHANGED: add_noise() inherited from RectifiedFlow
    # UNCHANGED: Euler / Heun ODE solvers inherited from RectifiedFlow
    # UNCHANGED: sample() inherited from RectifiedFlow
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # CHANGED: loss() — now activates reward weighting path
    # ------------------------------------------------------------------

    def loss(
        self,
        model: nn.Module,
        z_0: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        # REMOVED: task argument (unified model has no task heads)
        image_cond: torch.Tensor | None = None,
        reward_weight: torch.Tensor | None = None,  # ADDED (now actively used)
    ) -> torch.Tensor:
        """Compute flow matching MSE loss with optional RL reward weighting.

        CHANGED vs Cosmos 1:
          • ``task`` argument removed (unified model)
          • ``reward_weight`` is actively used here (was dead code in C1)

        Args:
            model        : Cosmos2DiT instance.
            z_0          : Clean latents.
            t            : Noise levels.
            text_seq     : VLM sequence embeddings.
            text_pool    : VLM pooled embeddings.
            image_cond   : Optional conditioning latent.
            reward_weight: Per-sample RL weights (B,) ∈ [0, 2].
                           None = standard training (no RL weighting).

        Returns:
            Scalar loss.
        """
        z_t, target = self.add_noise(z_0, t)

        # CHANGED: no ``task`` argument passed to model forward
        pred = model(z_t, t, text_seq, text_pool, image_cond=image_cond)

        mse = (pred - target).pow(2)

        # CHANGED: reward_weight is now used (was always None in Cosmos 1)
        if reward_weight is not None:
            w = reward_weight.view(-1, *([1] * (target.ndim - 1)))
            mse = mse * w

        return mse.mean()

    # ------------------------------------------------------------------
    # ADDED: Consistency distillation loss
    # ------------------------------------------------------------------

    def distillation_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        z_0: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        image_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consistency distillation training objective.

        ADDED in Cosmos 2.5.  Not present in Cosmos 1.

        Trains the student to match a frozen teacher model's predictions,
        enabling the student to generate quality samples with fewer steps.

        L_distill = || v_student(z_t, t, c) - v_teacher(z_t, t, c) ||²

        The teacher is assumed to be a copy of the pre-trained Cosmos 2.5
        (or even Cosmos 1) model loaded with frozen weights.

        Args:
            student_model : Trainable Cosmos2DiT (gradient flows here).
            teacher_model : Frozen reference model (detached).
            z_0           : Clean latents.
            t             : Noise levels.
            text_seq      : VLM sequence embeddings.
            text_pool     : VLM pooled embeddings.
            image_cond    : Optional conditioning latent.

        Returns:
            Scalar distillation loss.
        """
        z_t, _ = self.add_noise(z_0, t)

        # Student prediction (gradients flow).
        v_student = student_model(z_t, t, text_seq, text_pool, image_cond=image_cond)

        # Teacher prediction (no gradients).
        with torch.no_grad():
            v_teacher = teacher_model(z_t, t, text_seq, text_pool, image_cond=image_cond)

        return (v_student - v_teacher.detach()).pow(2).mean()

    # ------------------------------------------------------------------
    # ADDED: Combined training loss (flow matching + distillation)
    # ------------------------------------------------------------------

    def combined_loss(
        self,
        student_model: nn.Module,
        z_0: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        image_cond: torch.Tensor | None = None,
        reward_weight: torch.Tensor | None = None,
        teacher_model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined flow matching + distillation loss for Cosmos 2.5.

        ADDED in Cosmos 2.5.  Cosmos 1 only uses the flow matching loss.

        Returns:
            dict with keys "fm_loss", "distill_loss", "total_loss".
        """
        fm = self.loss(student_model, z_0, t, text_seq, text_pool,
                       image_cond=image_cond, reward_weight=reward_weight)

        if teacher_model is not None:
            distill = self.distillation_loss(
                student_model, teacher_model, z_0, t, text_seq, text_pool, image_cond
            )
        else:
            distill = torch.tensor(0.0, device=z_0.device)

        total = fm + self.distillation_weight * distill
        return {"fm_loss": fm, "distill_loss": distill, "total_loss": total}
