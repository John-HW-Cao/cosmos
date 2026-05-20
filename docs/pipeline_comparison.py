"""
Generate a multi-panel comparison diagram for Cosmos-Transfer1 vs Cosmos-Transfer2.5.

Based on the official repositories:
  https://github.com/nvidia-cosmos/cosmos-transfer1
  https://github.com/nvidia-cosmos/cosmos-transfer2.5

Run::

    python docs/pipeline_comparison.py

Outputs ``docs/pipeline_comparison.png``.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

# ── colour palette ───────────────────────────────────────────────────────────
T1_COLOR    = "#2E6FAD"   # Transfer1 – deep blue
T25_COLOR   = "#C95F1A"   # Transfer2.5 – burnt orange
SHARED_COLOR = "#3A8A3A"  # Shared / unchanged – green
NEW_COLOR   = "#8B44A8"   # New in Transfer2.5 – purple
BG_COLOR    = "#F5F7FA"
HDR_COLOR   = "#1E2D40"
ARROW_COLOR = "#444444"

# ── helpers ──────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, color, fontsize=8, text_color="white",
         style="round,pad=0.08", alpha=1.0, lw=1.2):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=lw,
        edgecolor="#FFFFFF", facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            zorder=4, multialignment="center")


def _arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=2)


def _side_arrow(ax, x1, y, x2, color=ARROW_COLOR):
    """Horizontal arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2), zorder=2)


def _cell(ax, x, y, w, h, text, facecolor, fontsize=8, bold=False,
          text_color="#111111"):
    rect = FancyBboxPatch(
        (x, y), w, h - 0.002,
        boxstyle="square,pad=0", linewidth=0.4,
        edgecolor="#CCCCCC", facecolor=facecolor,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=text_color,
            transform=ax.transAxes, multialignment="center", clip_on=False)


# ════════════════════════════════════════════════════════════════════════════
# Figure layout
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 28), facecolor=BG_COLOR)
fig.suptitle(
    "Cosmos-Transfer1  vs  Cosmos-Transfer2.5\nPipeline Architecture Comparison",
    fontsize=18, fontweight="bold", y=0.992, color=HDR_COLOR,
)

gs = GridSpec(
    3, 2,
    figure=fig,
    hspace=0.32, wspace=0.10,
    top=0.970, bottom=0.02,
    left=0.03, right=0.98,
    height_ratios=[2.4, 1.55, 1.0],
)

ax_t1   = fig.add_subplot(gs[0, 0])
ax_t25  = fig.add_subplot(gs[0, 1])
ax_tbl  = fig.add_subplot(gs[1, :])
ax_bar  = fig.add_subplot(gs[2, :])


# ════════════════════════════════════════════════════════════════════════════
# Panel A – Cosmos-Transfer1 architecture flow
# (source: github.com/nvidia-cosmos/cosmos-transfer1)
# ════════════════════════════════════════════════════════════════════════════

ax = ax_t1
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 10)
ax.set_ylim(0, 26)
ax.axis("off")
ax.set_title(
    "Cosmos-Transfer1\n(7B  ·  EDM SDE  ·  Cosmos1 VAE)",
    fontsize=12, fontweight="bold", color=T1_COLOR, pad=6,
)

# ── Inputs ──────────────────────────────────────────────────────────────────
_box(ax, 2.8, 25.0, 4.6, 0.9, "Text Prompt  +  Control Video\n(edge / vis / depth / seg / keypoint / lidar…)",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 24.5, 2.8, 23.7)

# ── Text encoder ─────────────────────────────────────────────────────────
_box(ax, 2.8, 23.2, 4.4, 0.9, "T5 Text Encoder  (frozen)\nDropout 0.2  →  text embeddings",
     SHARED_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 22.7, 2.8, 21.9)

# ── Tokenizer ────────────────────────────────────────────────────────────
_box(ax, 2.8, 21.4, 4.4, 0.9,
     "Cosmos1 VAE Tokenizer\n8×8×8 spatio-temporal  ·  16 ch\nlatent: [16, 24, 44, 80]",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 20.9, 2.8, 20.1)

# ── Noise schedule ────────────────────────────────────────────────────────
_box(ax, 2.8, 19.6, 4.4, 0.8,
     "EDM SDE Noise Schedule\nσ_min=0.0002  σ_max=80  ·  35 steps",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 19.2, 2.8, 18.4)

# ── ControlNet encoder ───────────────────────────────────────────────────
# Side branch: control input hint MLP
_box(ax, 7.4, 18.1, 3.8, 0.8,
     "Control Hint MLP\n[16→…→4096]  SiLU\n(per modality, latent space)",
     T1_COLOR, fontsize=7)
_side_arrow(ax, 5.7, 18.1, 5.3, T1_COLOR)

_box(ax, 2.8, 17.8, 4.4, 1.0,
     "ControlNet Encoder\n(first 14 of 28 DiT blocks  ·  O(N²))\nzero-linear residuals → base DiT",
     T1_COLOR, fontsize=7.5)

ax.text(1.05, 17.1,
        "• MultiControlNet: separate encoder\n  per modality → sum residuals",
        fontsize=6.8, color=T1_COLOR, style="italic")
_arrow(ax, 2.8, 17.3, 2.8, 16.5)

# ── Base DiT ─────────────────────────────────────────────────────────────
_box(ax, 2.8, 15.8, 4.4, 1.2,
     "GeneralDIT  (7B  ·  28 blocks)\nFA-CA-MLP format  ·  32 heads\nhidden=4096  ·  rope3d (fixed)\nAdaLN-LoRA dim=256",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 15.2, 2.8, 14.4)

# ── Video conditioning ────────────────────────────────────────────────────
_box(ax, 7.4, 15.8, 3.8, 0.8,
     "Video Cond via Channel-Concat\ncondition_video_input_mask\n+  video_cond_bool  flag",
     T1_COLOR, fontsize=7)
_side_arrow(ax, 5.7, 15.8, 5.3, T1_COLOR)

# ── Output head ───────────────────────────────────────────────────────────
_box(ax, 2.8, 13.9, 4.4, 0.8,
     "3 Task Output Heads\ntext2world / image2world / video2world",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 13.5, 2.8, 12.7)

# ── Classifier-free guidance ─────────────────────────────────────────────
_box(ax, 2.8, 12.2, 4.4, 0.8,
     "CFG + EDM Sampler\n35 steps  (or 1-step DMD2 distilled)",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 11.8, 2.8, 11.0)

_box(ax, 2.8, 10.5, 4.4, 0.8,
     "Cosmos1 VAE Decoder",
     T1_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 10.1, 2.8, 9.3)

ax.text(2.8, 9.0, "Output Video  (720p  ·  ~93 frames)",
        ha="center", fontsize=8.5, color=HDR_COLOR, fontweight="bold")

# ── Control modalities legend ─────────────────────────────────────────────
ax.text(0.25, 8.2,
        "Supported control signals:\nedge · vis · depth · seg\nkeypoint · upscale\nhdmap · lidar",
        fontsize=7.5, color=T1_COLOR,
        bbox=dict(boxstyle="round", facecolor="#D6E8F8", alpha=0.85))

ax.text(0.25, 5.8,
        "Video cond: channel-concat in latent\n(condition_video_input_mask)\n"
        "Pose cond: condition_video_pose\n"
        "VRAM: ~80 GB  (720p, H100)",
        fontsize=7.2, color="#333333",
        bbox=dict(boxstyle="round", facecolor="#EEF4FF", alpha=0.85))


# ════════════════════════════════════════════════════════════════════════════
# Panel B – Cosmos-Transfer2.5 architecture flow
# (source: github.com/nvidia-cosmos/cosmos-transfer2.5)
# ════════════════════════════════════════════════════════════════════════════

ax = ax_t25
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 10)
ax.set_ylim(0, 26)
ax.axis("off")
ax.set_title(
    "Cosmos-Transfer2.5\n(2B  ·  Rectified Flow  ·  WAN 2.1 VAE)",
    fontsize=12, fontweight="bold", color=T25_COLOR, pad=6,
)

# ── Inputs ───────────────────────────────────────────────────────────────
_box(ax, 2.8, 25.0, 4.6, 0.9,
     "Text Prompt  +  Control Video  +  (Image Context)\n(edge / vis / depth / seg  ·  spatiotemporal mask)",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 24.5, 2.8, 23.7)

# ── Text encoder (T5 standard + Qwen2.5-VL optional) ─────────────────────
_box(ax, 2.8, 23.2, 4.4, 0.9,
     "T5 Text Encoder  (frozen, standard)\n+ Qwen2.5-VL-7B  (reason1 path)",
     T25_COLOR, fontsize=7.5)

# ── SigLip2 image context (NEW) ───────────────────────────────────────────
_box(ax, 7.6, 23.2, 3.6, 0.9,
     "SigLip2 Image Encoder\n(NEW)  img_latent_dim=1024\ndropout=0.5",
     NEW_COLOR, fontsize=7)
_side_arrow(ax, 5.7, 23.2, 6.0, NEW_COLOR)

_arrow(ax, 2.8, 22.7, 2.8, 21.9)

# ── Tokenizer ─────────────────────────────────────────────────────────────
_box(ax, 2.8, 21.4, 4.4, 0.9,
     "WAN 2.1 VAE Tokenizer\n4×8×8 spatio-temporal  ·  16 ch\nsupports torch.compile + ctx parallelism",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 20.9, 2.8, 20.1)

# ── Noise schedule ────────────────────────────────────────────────────────
_box(ax, 2.8, 19.6, 4.4, 0.8,
     "Rectified Flow  (linear trajectory)\nlow_sigma_threshold=0.05  ·  35 steps\n(or 4-step DMD2 distilled)",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 19.2, 2.8, 18.4)

# ── VACE Control (side branch) ────────────────────────────────────────────
_box(ax, 7.6, 18.1, 3.6, 1.0,
     "VACE Control Input\n(latent concat + mask channel)\nup to 8 modalities  ·  641 ch max",
     T25_COLOR, fontsize=7)
_side_arrow(ax, 5.7, 18.1, 6.0, T25_COLOR)

_box(ax, 2.8, 17.8, 4.4, 1.0,
     "VACE-style Control Encoder\n4 spaced ControlEncoderDiTBlocks\nat layers {0, 7, 14, 21} of 28\nafter_proj: Linear(4×2048→2048)",
     T25_COLOR, fontsize=7.5)

ax.text(0.5, 17.1,
        "• Multi-branch: 4 separate encoder lists\n  fused by after_proj per control layer",
        fontsize=6.8, color=T25_COLOR, style="italic")
_arrow(ax, 2.8, 17.3, 2.8, 16.5)

# ── Base DiT ─────────────────────────────────────────────────────────────
_box(ax, 2.8, 15.6, 4.4, 1.4,
     "MinimalV4LVGControlVaceDiT  (2B)\nSelf-Attn  →  Image-CrossAttn (SigLip2)\n→  Text-CrossAttn  →  MLP\n28 blocks  ·  16 heads  ·  hidden=2048\nrope3d (learnable)  ·  Sparse Attn (SAC)",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 14.9, 2.8, 14.1)

# ── ControlAwareDiTBlock note ─────────────────────────────────────────────
ax.text(0.5, 14.55,
        "All 28 blocks are ControlAwareDiTBlock;\nonly 4 have active VACE control injection",
        fontsize=6.8, color=T25_COLOR, style="italic")

# ── Single unified output head ─────────────────────────────────────────────
_box(ax, 2.8, 13.6, 4.4, 0.8,
     "Single Unified Output Head\n(no task argument)",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 13.2, 2.8, 12.4)

# ── Sampler ───────────────────────────────────────────────────────────────
_box(ax, 2.8, 11.9, 4.4, 0.8,
     "CFG + Rectified Flow Sampler\n35 steps  (or 4-step DMD2 distilled)",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 11.5, 2.8, 10.7)

_box(ax, 2.8, 10.2, 4.4, 0.8,
     "WAN 2.1 VAE Decoder",
     T25_COLOR, fontsize=7.5)
_arrow(ax, 2.8, 9.8, 2.8, 9.0)

ax.text(2.8, 8.75, "Output Video  (720p  ·  ~93 frames)",
        ha="center", fontsize=8.5, color=HDR_COLOR, fontweight="bold")

# ── Control modalities legend ─────────────────────────────────────────────
ax.text(0.25, 8.0,
        "Supported control signals:\nedge · vis · depth · seg\n(+ experimental: inpaint · hdmap_bbox)\n"
        "AUTO_MULTIVIEW  ·  ROBOT_MULTIVIEW",
        fontsize=7.5, color=T25_COLOR,
        bbox=dict(boxstyle="round", facecolor="#FDEBD0", alpha=0.85))

ax.text(0.25, 5.5,
        "Video cond: use_video_condition flag\n"
        "Image cond: SigLip2 cross-attention (NEW)\n"
        "Spatiotemporal mask per modality (SAM2)\n"
        "VRAM: ~65 GB  (720p, H100)",
        fontsize=7.2, color="#333333",
        bbox=dict(boxstyle="round", facecolor="#FFF3EA", alpha=0.85))


# ════════════════════════════════════════════════════════════════════════════
# Panel C – Comparison table
# ════════════════════════════════════════════════════════════════════════════

ax = ax_tbl
ax.set_facecolor(BG_COLOR)
ax.axis("off")
ax.set_title("Component-by-Component Comparison", fontsize=13,
             fontweight="bold", color=HDR_COLOR, pad=6)

rows = [
    # (Category, Transfer1, Transfer2.5, changed?)
    ("Base model size",
     "7B  (GeneralDIT / FADITV2)\nhidden=4096  ·  28 blocks  ·  32 heads",
     "2B released  (MinimalV4LVGControlVaceDiT)\nhidden=2048  ·  28 blocks  ·  16 heads\n(14B variant in code: hidden=5120, 36 blocks)",
     True),
    ("Diffusion type",
     "EDM SDE\nσ_min=0.0002, σ_max=80\np_mean=0, p_std=1",
     "Rectified Flow  (linear trajectory)\nlow_sigma_threshold=0.05",
     True),
    ("VAE tokenizer",
     "Cosmos1 VAE\nShared Joint Image-Video\n8×8×8 compression  ·  16 ch",
     "WAN 2.1 VAE\n4×8×8 compression  ·  16 ch\ntorch.compile + context parallelism",
     True),
    ("Text encoder",
     "T5  (frozen, dropout=0.2)\nstandard text embeddings",
     "T5  (standard)  +  Qwen2.5-VL-7B-Instruct\n(reason1 path: 28 layers, hidden=3584)\nmean-pool across layers",
     True),
    ("Image context",
     "None",
     "SigLip2 vision encoder  (NEW)\nimg_latent_dim=1024, dropout=0.5\nI2VCrossAttentionFull inside each DiT block",
     True),
    ("Attention style",
     "Full self-attention  O(N²)\n(FA-CA-MLP per block format)",
     "Sparse Attention  (SACConfig)\nminimal_a2a backend\nSelf → Image-CrossAttn → Text-CrossAttn → MLP",
     True),
    ("Positional embedding",
     "rope3d  (fixed, non-learnable)",
     "rope3d  (learnable, pos_emb_learnable=True)",
     True),
    ("AdaLN variant",
     "AdaLN-LoRA  (adaln_lora_dim=256)",
     "AdaLN-LoRA  (adaln_lora_dim=256)  — unchanged",
     False),
    ("ControlNet type",
     "Classic ControlNet encoder\nFirst 14 of 28 blocks copied\nzero-linear residuals per block",
     "VACE-style control encoder\n4 spaced ControlEncoderDiTBlocks\nat layers {0, 7, 14, 21}",
     True),
    ("Control input encoding",
     "Control Hint MLP  [16→…→4096]  SiLU\nper modality  (7-layer MLP in latent space)",
     "Direct latent channel concat  +  mask ch\nVACE: up to (16+64)×8+1 = 641 channels",
     True),
    ("Multi-control fusion",
     "MultiControlNet:\nseparate encoder per modality\nsum of residuals",
     "num_control_branches=4\n4 separate ControlEncoderDiTBlock lists\nafter_proj: Linear(4×2048 → 2048)",
     True),
    ("Video conditioning",
     "condition_video_input_mask\nchannel-concat to DiT input\nvideo_cond_bool flag",
     "use_video_condition flag\n+ SigLip2 for image-to-video context",
     True),
    ("Control modalities",
     "edge, vis, depth, seg\nkeypoint, upscale, hdmap, lidar",
     "edge, vis, depth, seg\n(+ experimental: inpaint, hdmap_bbox)\nAUTO_MULTIVIEW  ·  ROBOT_MULTIVIEW",
     True),
    ("Output heads",
     "3 task-specific heads\ntext2world / image2world / video2world",
     "1 unified head\n(no task argument in forward())",
     True),
    ("Inference steps",
     "35 steps  (EDM)\n1-step distilled  (DMD2, edge only)",
     "35 steps  (Rectified Flow)\n4-step distilled  (DMD2, edge)",
     True),
    ("Inference config",
     "Hydra + Python dicts",
     "Pydantic + JSON spec files",
     True),
    ("VRAM  (720p, H100)",
     "~80 GB",
     "~65 GB",
     True),
]

col_widths = [0.16, 0.38, 0.38, 0.07]
col_xs     = [0.005, 0.167, 0.549, 0.930]
headers    = ["Component", "Cosmos-Transfer1", "Cosmos-Transfer2.5", "Changed?"]

total_rows = len(rows) + 1
row_h = 0.960 / (total_rows + 0.4)

# Header
for cx, cw, hdr in zip(col_xs, col_widths, headers):
    _cell(ax, cx, 1.0 - row_h, cw, row_h, hdr, HDR_COLOR,
          fontsize=9, bold=True, text_color="white")

# Data rows
for ri, (comp, t1_val, t25_val, changed) in enumerate(rows):
    y = 1.0 - row_h * (ri + 2)
    row_bg      = "#FFFFFF" if ri % 2 == 0 else "#F0F5FB"
    changed_bg  = "#FDDCCA" if changed else "#D6EFD6"
    changed_txt = "YES" if changed else "no"
    _cell(ax, col_xs[0], y, col_widths[0], row_h, comp,     row_bg,    fontsize=7.8, bold=True)
    _cell(ax, col_xs[1], y, col_widths[1], row_h, t1_val,   "#E6F2FF", fontsize=7.2)
    _cell(ax, col_xs[2], y, col_widths[2], row_h, t25_val,  "#FFF2E8", fontsize=7.2)
    _cell(ax, col_xs[3], y, col_widths[3], row_h, changed_txt, changed_bg, fontsize=8, bold=True,
          text_color=(T25_COLOR if changed else SHARED_COLOR))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# Panel D – Quantitative bar comparison
# ════════════════════════════════════════════════════════════════════════════

ax = ax_bar
ax.set_facecolor(BG_COLOR)
ax.set_title("Key Quantitative Parameters", fontsize=13,
             fontweight="bold", color=HDR_COLOR, pad=4)

metrics = [
    # (label,              T1 value, T2.5 value, unit suffix)
    ("Model\nparams",        7,     2,     "B"),
    ("DiT hidden\ndim",      4096,  2048,  ""),
    ("DiT blocks",           28,    28,    ""),
    ("Active ctrl\nblocks",  14,    4,     ""),
    ("VAE spat.\ncompress",  8,     8,     "×"),
    ("VAE temp.\ncompress",  8,     4,     "×"),
    ("Max infer\nsteps",     35,    35,    ""),
    ("Distilled\nsteps",     1,     4,     ""),
    ("VRAM 720p\n(GB)",      80,    65,    ""),
    ("Control\nmodalities",  8,     4,     ""),
    ("Output\nheads",        3,     1,     ""),
    ("Pipeline\nclasses",    3,     1,     ""),
]

n = len(metrics)
x = np.arange(n)
bar_w = 0.36

vals_t1  = np.array([m[1] for m in metrics], dtype=float)
vals_t25 = np.array([m[2] for m in metrics], dtype=float)

bars1 = ax.bar(x - bar_w / 2, vals_t1,  bar_w, color=T1_COLOR,  label="Transfer1",   alpha=0.85, zorder=3)
bars2 = ax.bar(x + bar_w / 2, vals_t25, bar_w, color=T25_COLOR, label="Transfer2.5", alpha=0.85, zorder=3)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
ax.set_ylabel("Value  (log scale)", fontsize=9)
ax.legend(fontsize=11, loc="upper right",
          handles=[
              plt.Rectangle((0,0),1,1, color=T1_COLOR,  alpha=0.85, label="Transfer1"),
              plt.Rectangle((0,0),1,1, color=T25_COLOR, alpha=0.85, label="Transfer2.5"),
          ])
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.set_facecolor(BG_COLOR)
for sp in ["top","right"]:
    ax.spines[sp].set_visible(False)

# Annotate
for bar, val, unit in zip(list(bars1) + list(bars2),
                           list(vals_t1) + list(vals_t25),
                           [m[3] for m in metrics] * 2):
    disp = f"{int(val)}{unit}" if val == int(val) else f"{val}{unit}"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            disp, ha="center", va="bottom", fontsize=7.5, color="#222222")

# Source credit
fig.text(0.5, 0.005,
         "Sources: github.com/nvidia-cosmos/cosmos-transfer1  ·  github.com/nvidia-cosmos/cosmos-transfer2.5",
         ha="center", fontsize=7.5, color="#888888", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────

out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "pipeline_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {out_path}")
