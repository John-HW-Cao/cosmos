"""
Generate a multi-panel comparison diagram for Cosmos 1 vs Cosmos 2.5 pipelines.

Run::

    python docs/pipeline_comparison.py

Outputs ``docs/pipeline_comparison.png``.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

# ── colour palette ───────────────────────────────────────────────────────────
C1_COLOR   = "#4A90D9"   # Cosmos 1  – blue
C25_COLOR  = "#E87722"   # Cosmos 2.5 – orange
SAME_COLOR = "#6CB26C"   # Unchanged  – green
BG_COLOR   = "#F7F9FC"
ARROW_COLOR = "#555555"

# ── helpers ──────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, color, fontsize=8, text_color="white",
         style="round,pad=0.1", alpha=1.0):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style,
        linewidth=1.2,
        edgecolor="white",
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=4, wrap=True,
            multialignment="center")


def _arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
        zorder=2,
    )


# ── Figure layout ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 22), facecolor=BG_COLOR)
fig.suptitle(
    "Cosmos 1  vs  Cosmos 2.5 — Pipeline Architecture Comparison",
    fontsize=17, fontweight="bold", y=0.985, color="#222222",
)

gs = GridSpec(
    3, 2,
    figure=fig,
    hspace=0.38,
    wspace=0.12,
    top=0.96,
    bottom=0.03,
    left=0.04,
    right=0.97,
    height_ratios=[1.9, 1.0, 1.3],
)

ax_c1   = fig.add_subplot(gs[0, 0])   # Cosmos 1 architecture flow
ax_c25  = fig.add_subplot(gs[0, 1])   # Cosmos 2.5 architecture flow
ax_tbl  = fig.add_subplot(gs[1, :])   # Side-by-side component table
ax_diff = fig.add_subplot(gs[2, :])   # Bar / radar comparison


# ════════════════════════════════════════════════════════════════════════════
# Panel A – Cosmos 1 architecture flow
# ════════════════════════════════════════════════════════════════════════════

ax = ax_c1
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")
ax.set_title("Cosmos 1 Pipeline", fontsize=13, fontweight="bold", color=C1_COLOR, pad=6)

# --- three task pipelines shown stacked, then merge into shared backbone ----
# Inputs
pipe_labels = ["Text2WorldPipeline", "Image2WorldPipeline", "Video2WorldPipeline"]
pipe_ys     = [17.5, 15.5, 13.5]
for label, py in zip(pipe_labels, pipe_ys):
    _box(ax, 2.5, py, 4.0, 1.1, label, C1_COLOR, fontsize=8)

# Shared backbone label
ax.text(2.5, 12.2, "3 separate pipeline classes\n(one per task)",
        ha="center", va="center", fontsize=7.5, color=C1_COLOR,
        style="italic")

# Arrow down to "T5-XXL Text Encoder"
_arrow(ax, 2.5, 12.9, 2.5, 11.8)

_box(ax, 2.5, 11.2, 4.4, 1.0, "T5-XXL Text Encoder\n(frozen, 4096-d)", SAME_COLOR, fontsize=8)

_arrow(ax, 2.5, 10.7, 2.5, 9.8)

_box(ax, 2.5, 9.2, 4.4, 1.0, "CV8×8×8 Tokenizer\n(spatial 8×, total 256×)", C1_COLOR, fontsize=8)

_arrow(ax, 2.5, 8.7, 2.5, 7.8)

_box(ax, 2.5, 7.2, 4.4, 1.0, "RoPE3D Pos. Embed.\n(max_frames=57)", SAME_COLOR, fontsize=8)

_arrow(ax, 2.5, 6.7, 2.5, 5.8)

_box(ax, 2.5, 5.2, 4.4, 1.2,
     "Cosmos1DiT (28 blocks)\nFullSpaceTimeAttn  O(N²)\nAdaLN-Zero (text+time)",
     C1_COLOR, fontsize=7.5)

# Text cross-attn (once, before blocks)
_box(ax, 7.5, 5.2, 4.0, 0.9, "Text Cross-Attn\n(once before blocks)", C1_COLOR, fontsize=7.5)
_arrow(ax, 5.5, 5.2, 5.7, 5.2)

_arrow(ax, 2.5, 4.6, 2.5, 3.7)

_box(ax, 2.5, 3.2, 4.4, 1.0,
     "3 task heads\nhead_text2world\nhead_image2world / head_video2world",
     C1_COLOR, fontsize=7.5)

_arrow(ax, 2.5, 2.7, 2.5, 1.8)

_box(ax, 2.5, 1.2, 4.4, 1.0, "RectifiedFlow Scheduler\n35 steps, cfg=7.0", C1_COLOR, fontsize=8)

_arrow(ax, 2.5, 0.7, 2.5, 0.15)
ax.text(2.5, 0.05, "Output Video  (T≤57 frames)", ha="center", fontsize=8,
        color="#333333", fontweight="bold")

# conditioning legend note
ax.text(0.2, 0.35,
        "Image/Video cond: token prepend\n(increases sequence length N)",
        fontsize=7, color=C1_COLOR, style="italic",
        bbox=dict(boxstyle="round", facecolor="#DDEEFF", alpha=0.7))


# ════════════════════════════════════════════════════════════════════════════
# Panel B – Cosmos 2.5 architecture flow
# ════════════════════════════════════════════════════════════════════════════

ax = ax_c25
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")
ax.set_title("Cosmos 2.5 Pipeline", fontsize=13, fontweight="bold", color=C25_COLOR, pad=6)

# Single unified pipeline
_box(ax, 2.5, 17.5, 4.0, 1.1,
     "UnifiedPipeline\n(single class, auto-detects task)",
     C25_COLOR, fontsize=8)

ax.text(2.5, 16.0, "task auto-detected from inputs\n(image_cond / video_cond / text-only)",
        ha="center", va="center", fontsize=7.5, color=C25_COLOR, style="italic")

_arrow(ax, 2.5, 15.4, 2.5, 14.5)

_box(ax, 2.5, 13.9, 4.4, 1.0,
     "CosmosReason VLM Encoder\n(frozen, 5120-d)", C25_COLOR, fontsize=8)

_arrow(ax, 2.5, 13.4, 2.5, 12.5)

_box(ax, 2.5, 11.9, 4.4, 1.0,
     "CV4×16×16 Tokenizer\n(spatial 16×, total 1024×)", C25_COLOR, fontsize=8)

# Visual Pool projector (side branch)
_box(ax, 7.5, 11.9, 3.8, 0.9,
     "VisualPoolProjector\n(pool cond → 5120-d vec)", C25_COLOR, fontsize=7.5)
_arrow(ax, 5.5, 11.9, 5.7, 11.9)

_arrow(ax, 2.5, 11.4, 2.5, 10.5)

_box(ax, 2.5, 9.9, 4.4, 1.0, "RoPE3D Pos. Embed.\n(max_frames=121)", SAME_COLOR, fontsize=8)

_arrow(ax, 2.5, 9.4, 2.5, 8.3)

_box(ax, 2.5, 7.6, 4.4, 1.4,
     "Cosmos2DiT (24 blocks)\nChunkedSpaceTimeAttn  O(C·N)\nAdaLNZeroMultiModal\n(text+time+visual)",
     C25_COLOR, fontsize=7.5)

# Per-block cross-attn
_box(ax, 7.5, 7.6, 3.8, 0.9,
     "Text Cross-Attn\n(every block)", C25_COLOR, fontsize=7.5)
_arrow(ax, 5.5, 7.6, 5.7, 7.6)

_arrow(ax, 2.5, 6.9, 2.5, 6.1)

_box(ax, 2.5, 5.5, 4.4, 1.0,
     "Single unified head\nout_proj (replaces 3 heads)",
     C25_COLOR, fontsize=8)

_arrow(ax, 2.5, 5.0, 2.5, 4.1)

_box(ax, 2.5, 3.5, 4.4, 1.0,
     "DistilledRectifiedFlow\n20 steps, cfg=6.0  (+RL loss)",
     C25_COLOR, fontsize=8)

_arrow(ax, 2.5, 3.0, 2.5, 2.0)
ax.text(2.5, 1.85, "Output Video  (T≤121 frames)", ha="center", fontsize=8,
        color="#333333", fontweight="bold")

ax.text(0.2, 0.35,
        "Image/Video cond: VisualPoolProjector\n→ adaLN-Zero (N unchanged)",
        fontsize=7, color=C25_COLOR, style="italic",
        bbox=dict(boxstyle="round", facecolor="#FFEEDD", alpha=0.7))


# ════════════════════════════════════════════════════════════════════════════
# Panel C – Component comparison table
# ════════════════════════════════════════════════════════════════════════════

ax = ax_tbl
ax.set_facecolor(BG_COLOR)
ax.axis("off")
ax.set_title("Component-by-Component Differences", fontsize=13,
             fontweight="bold", color="#333333", pad=4)

rows = [
    # (Component, Cosmos 1, Cosmos 2.5, changed?)
    ("Pipeline class(es)",       "3 separate\n(Text2World / Image2World / Video2World)",
                                  "1 unified\n(UnifiedPipeline)", True),
    ("Task dispatch",             "task= argument in run()",
                                  "auto-detected from inputs", True),
    ("Text encoder",              "T5-XXL  (4096-d)",
                                  "CosmosReason VLM  (5120-d)", True),
    ("Video tokenizer",           "CV8×8×8  (spatial 8×, total 256×)",
                                  "CV4×16×16  (spatial 16×, total 1024×)", True),
    ("Attention type",            "FullSpaceTimeAttention\nO(N²) memory",
                                  "ChunkedSpaceTimeAttention\nO(C·N) memory", True),
    ("Conditioning (image/video)","Token prepend → longer N",
                                  "VisualPoolProjector → adaLN-Zero\n(N unchanged)", True),
    ("Conditioning module",       "AdaLN-Zero\n(time + text_pool)",
                                  "AdaLNZeroMultiModal\n(time + text_pool + visual_pool)", True),
    ("Text cross-attention",      "Once, before block 0",
                                  "Inside every DiT block", True),
    ("Output heads",              "3 task heads\nhead_text2world / image2world / video2world",
                                  "1 unified head\n(out_proj)", True),
    ("DiT depth / hidden dim",    "28 layers, hidden=4096 (7B)",
                                  "24 layers, hidden=2048 (2B) / 4096 (14B)", True),
    ("Max video frames",          "57 frames (~4 s @ 14 fps)",
                                  "121 frames (~30 s @ 14 fps)", True),
    ("Noise scheduler",           "RectifiedFlow\n35 steps, guidance=7.0",
                                  "DistilledRectifiedFlow\n20 steps, guidance=6.0 + RL", True),
    ("FeedForward (SwiGLU)",      "SwiGLU FFN", "SwiGLU FFN (unchanged)", False),
    ("Tokenizer/text freeze",     "Frozen weights", "Frozen weights (unchanged)", False),
    ("RoPE variant",              "mrope_interleave", "mrope_interleave (unchanged)", False),
]

col_widths = [0.18, 0.36, 0.36, 0.08]
col_xs     = [0.01, 0.20, 0.57, 0.93]
headers    = ["Component", "Cosmos 1", "Cosmos 2.5", "Changed?"]

total_rows = len(rows) + 1  # +1 for header
row_h = 1.0 / (total_rows + 0.5)

def _cell(ax, x, y, w, h, text, facecolor, fontsize=8, bold=False):
    rect = FancyBboxPatch((x, y), w, h - 0.002,
                           boxstyle="square,pad=0",
                           linewidth=0.4, edgecolor="#CCCCCC",
                           facecolor=facecolor, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fw, transform=ax.transAxes,
            multialignment="center", clip_on=False)

# Header row
for xi, (cx, cw, hdr) in enumerate(zip(col_xs, col_widths, headers)):
    _cell(ax, cx, 1 - row_h, cw, row_h, hdr, "#2C3E50",
          fontsize=9, bold=True)
    ax.texts[-1].set_color("white")

# Data rows
for ri, (comp, c1_val, c25_val, changed) in enumerate(rows):
    y = 1 - row_h * (ri + 2)
    row_bg = "#FFFFFF" if ri % 2 == 0 else "#F2F6FB"
    changed_bg  = "#FDDCCA" if changed else "#D6EFD6"
    changed_txt = "✔" if changed else "—"
    _cell(ax, col_xs[0], y, col_widths[0], row_h, comp,        row_bg,      fontsize=8, bold=True)
    _cell(ax, col_xs[1], y, col_widths[1], row_h, c1_val,      "#EAF3FB",   fontsize=7.5)
    _cell(ax, col_xs[2], y, col_widths[2], row_h, c25_val,     "#FFF5EE",   fontsize=7.5)
    _cell(ax, col_xs[3], y, col_widths[3], row_h, changed_txt, changed_bg,  fontsize=9, bold=True)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# Panel D – Quantitative comparison bar chart
# ════════════════════════════════════════════════════════════════════════════

ax = ax_diff
ax.set_facecolor(BG_COLOR)
ax.set_title("Key Quantitative Differences", fontsize=13,
             fontweight="bold", color="#333333", pad=4)

metrics = [
    ("Spatial\ncompression",       8,   16,  "×"),
    ("Total video\ncompression",   256, 1024, "×"),
    ("Text encoder\ndim",          4096, 5120, ""),
    ("Cond. dim\n(cond_dim)",      4096, 5120, ""),
    ("DiT hidden\ndim (2B)",       4096, 2048, ""),
    ("DiT depth\n(layers)",        28,   24,  ""),
    ("Max frames",                 57,   121, ""),
    ("Inference\nsteps",           35,   20,  ""),
    ("Guidance\nscale",            7.0,  6.0, ""),
    ("Pipeline\nclasses",          3,    1,   ""),
    ("Output\nheads",              3,    1,   ""),
]

n = len(metrics)
x = np.arange(n)
bar_w = 0.35

# Normalise each pair so both bars are visible on log scale
vals_c1  = [m[1] for m in metrics]
vals_c25 = [m[2] for m in metrics]

bars1 = ax.bar(x - bar_w / 2, vals_c1,  bar_w, color=C1_COLOR,  label="Cosmos 1",   alpha=0.85, zorder=3)
bars2 = ax.bar(x + bar_w / 2, vals_c25, bar_w, color=C25_COLOR, label="Cosmos 2.5", alpha=0.85, zorder=3)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
ax.set_ylabel("Value  (log scale)", fontsize=9)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.set_facecolor(BG_COLOR)

# Annotate bars with their raw values
for bar, val, unit in zip(list(bars1) + list(bars2),
                           vals_c1 + vals_c25,
                           [m[3] for m in metrics] * 2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.12,
        f"{val}{unit}",
        ha="center", va="bottom", fontsize=7, color="#333333",
    )

# ── Save ─────────────────────────────────────────────────────────────────────

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "pipeline_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {out_path}")
