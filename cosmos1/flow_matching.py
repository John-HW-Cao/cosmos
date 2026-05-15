"""
Cosmos 1 Rectified Flow (Flow Matching) noise schedule and training loss.

Rectified Flow (Liu et al., 2022) trains a neural network to predict the
straight-line velocity field that maps noise → data:

    z_t = (1 - t) * z_0 + t * epsilon,    t ∈ [0, 1]
    v*  = epsilon - z_0                   (target velocity)

At inference time the ODE  dz/dt = v_θ(z_t, t)  is solved with a simple
Euler solver (or a more accurate Heun / DPM++ solver).

── Cosmos 2.5 changes ──────────────────────────────────────────────────
1. CHANGED  : num_inference_steps: 35 → 20  (distilled model is 2× faster)
2. ADDED    : Consistency distillation loss term (weighted alongside
              the standard flow matching MSE loss).
3. ADDED    : reward_weight parameter for RL post-training step.
4. UNCHANGED: Forward (noising) formula — straight-line interpolation.
5. UNCHANGED: Euler / Heun ODE solvers.
6. UNCHANGED: Classifier-free guidance (CFG) at inference.

See cosmos2/flow_matching.py for the adapted implementation.
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RectifiedFlow(nn.Module):
    """Rectified Flow noise schedule for Cosmos 1.

    Handles:
      • noise sampling (forward process)
      • training loss computation
      • inference ODE solving (Euler and Heun)
      • classifier-free guidance

    Args:
        num_inference_steps : ODE solver steps at inference.
                              Cosmos 1 default: 35.
                              Cosmos 2.5 reduces to 20 (distilled).
        guidance_scale      : CFG scale.  0.0 = unconditional;
                              default 7.0 for text2world.
    """

    def __init__(
        self,
        num_inference_steps: int = 35,   # ← Cosmos 2.5 changes to 20
        guidance_scale: float = 7.0,
        sigma_min: float = 0.002,
    ) -> None:
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # Forward process (noising)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy latent z_t = (1-t)·z_0 + t·ε.

        Args:
            z_0  : Clean latent (B, C, T', H', W').
            t    : Noise level  (B,) in [sigma_min, 1].
            noise: Optional pre-sampled Gaussian noise; if None, sampled here.

        Returns:
            z_t   : Noisy latent  (B, C, T', H', W').
            target: Target velocity = ε - z_0.
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        # Broadcast t to (B, 1, 1, 1, 1) for element-wise multiply.
        t_b = t.view(-1, *([1] * (z_0.ndim - 1)))
        z_t = (1 - t_b) * z_0 + t_b * noise
        target = noise - z_0
        return z_t, target

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(
        self,
        model: nn.Module,
        z_0: torch.Tensor,
        t: torch.Tensor,
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        task: str = "text2world",
        image_cond: torch.Tensor | None = None,
        reward_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the flow matching MSE loss.

        L = E_t [ || v_θ(z_t, t, c) - (ε - z_0) ||² ]

        Args:
            model        : Cosmos1DiT instance.
            z_0          : Clean latents.
            t            : Noise levels sampled from U[sigma_min, 1].
            text_seq     : T5 sequence embeddings.
            text_pool    : T5 pooled embedding.
            task         : Task name for head selection.
            image_cond   : Optional conditioning latent.
            reward_weight: Per-sample RL weight (B,).  Not used in Cosmos 1
                           (always None); added for Cosmos 2.5 compatibility.
                           See cosmos2/flow_matching.py.

        Returns:
            Scalar loss.
        """
        z_t, target = self.add_noise(z_0, t)
        pred = model(z_t, t, text_seq, text_pool, task=task, image_cond=image_cond)

        # Pixel-wise MSE over all latent dimensions.
        mse = (pred - target).pow(2).mean()

        # ── RL reward weighting (Cosmos 2.5 addition) ─────────────────
        # In Cosmos 1 reward_weight is always None so this branch is dead.
        # Cosmos 2.5 uses reward_weight to up-weight high-reward samples
        # from the RL post-training phase.  See cosmos2/flow_matching.py.
        if reward_weight is not None:
            mse = (mse * reward_weight.view(-1, *([1] * (target.ndim - 1)))).mean()

        return mse

    # ------------------------------------------------------------------
    # Inference ODE solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        text_seq: torch.Tensor,
        text_pool: torch.Tensor,
        text_seq_uncond: torch.Tensor | None = None,
        text_pool_uncond: torch.Tensor | None = None,
        task: str = "text2world",
        image_cond: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        solver: str = "euler",
    ) -> torch.Tensor:
        """Solve the flow-matching ODE to generate a clean latent.

        Args:
            model           : Cosmos1DiT in eval mode.
            shape           : Output latent shape (B, C, T', H', W').
            text_seq/pool   : Conditional text embeddings.
            text_*_uncond   : Unconditional embeddings for CFG.
                              If None, CFG is disabled.
            task            : Task name.
            image_cond      : Optional conditioning latent.
            device          : Target device.
            solver          : "euler" (first-order) or "heun" (second-order).

        Returns:
            Clean latent z_0 (B, C, T', H', W').
        """
        use_cfg = (
            text_seq_uncond is not None
            and self.guidance_scale > 1.0
        )

        # Start from pure noise at t=1.
        z = torch.randn(shape, device=device)
        dt = 1.0 / self.num_inference_steps
        # Time steps from t=1 (noise) → t=sigma_min (clean).
        ts = torch.linspace(1.0, self.sigma_min, self.num_inference_steps + 1,
                            device=device)

        model.eval()
        for i in range(self.num_inference_steps):
            t_cur = ts[i].expand(shape[0])

            if use_cfg:
                # Classifier-free guidance: run model twice.
                v_cond   = model(z, t_cur, text_seq,        text_pool,
                                 task=task, image_cond=image_cond)
                v_uncond = model(z, t_cur, text_seq_uncond, text_pool_uncond,
                                 task=task, image_cond=image_cond)
                v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                v = model(z, t_cur, text_seq, text_pool,
                          task=task, image_cond=image_cond)

            if solver == "euler":
                z = z + (-dt) * v
            elif solver == "heun":
                # Heun (corrector step).
                z_mid = z + (-dt) * v
                t_next = ts[i + 1].expand(shape[0])
                if use_cfg:
                    v2_cond   = model(z_mid, t_next, text_seq,        text_pool,
                                      task=task, image_cond=image_cond)
                    v2_uncond = model(z_mid, t_next, text_seq_uncond, text_pool_uncond,
                                      task=task, image_cond=image_cond)
                    v2 = v2_uncond + self.guidance_scale * (v2_cond - v2_uncond)
                else:
                    v2 = model(z_mid, t_next, text_seq, text_pool,
                               task=task, image_cond=image_cond)
                z = z + (-dt) * 0.5 * (v + v2)
            else:
                raise ValueError(f"Unknown solver: {solver!r}")

        return z
