# Cosmos: NVIDIA World Foundation Models (Cosmos 1 & Cosmos 2.5)

Reference implementations of both Cosmos 1 and Cosmos 2.5 video generation models.
Every change from Cosmos 1 to Cosmos 2.5 is explicitly annotated in the code.

---

## Repository Layout

```
cosmos/
├── __init__.py
├── shared/                    # Building blocks shared by both generations
│   ├── activations.py         # SwiGLU, GELU
│   ├── normalization.py       # RMSNorm, AdaLNZero
│   └── embeddings.py          # SinusoidalPosEmbed, RoPE2D, RoPE3D
│
├── cosmos1/                   # Cosmos 1.0 implementation
│   ├── config.py              # Hyper-parameters (CV8x8x8, full attention, T5)
│   ├── tokenizer.py           # Continuous Video Tokenizer: 4×8×8 compression
│   ├── attention.py           # Full 3-D self-attention (O(N²))
│   ├── dit.py                 # Diffusion Transformer + 3 task-specific heads
│   ├── text_encoder.py        # Frozen T5-XXL encoder (4096-d)
│   ├── flow_matching.py       # Rectified Flow (35 inference steps)
│   └── pipeline.py            # Text2World / Image2World / Video2World pipelines
│
└── cosmos2/                   # Cosmos 2.5 — adapted from Cosmos 1
    ├── config.py              # Updated hyper-parameters (CV4x16x16, chunked attn, VLM)
    ├── tokenizer.py           # 4×16×16 compression (config-driven, no code change)
    ├── attention.py           # Chunked space-time attention (O(C·N))
    ├── dit.py                 # Unified DiT + multi-modal adaLN + single head
    ├── text_encoder.py        # Cosmos Reason VLM encoder (5120-d)
    ├── flow_matching.py       # Distilled Rectified Flow (20 steps) + RL weighting
    └── pipeline.py            # Single UnifiedPipeline replacing 3 task pipelines
```

---

## Architecture Overview

### Cosmos 1 (Cosmos-1.0)

```
Text Prompt ──► T5-XXL (frozen) ──► text_seq (B,L,4096) ──┐
                                  ──► text_pool (B,4096)    │
                                                            │  adaLN-Zero
Video ──► CV8x8x8 Tokenizer ──► z₀ (B,16,T/4,H/8,W/8)    │  conditioning
                                                            │
Noise ──► z_t = (1-t)·z₀ + t·ε ──►  DiT (28 blocks)  ◄───┘
                                      │  FullSpaceTimeAttn (O(N²))
                                      │  FeedForward (SwiGLU)
                                      │  3 task heads
                                      ▼
                              v(z_t, t, c)  ← velocity field
                                      │
                           Euler/Heun ODE (35 steps)
                                      │
                                      ▼
                              z₀ ──► CV8x8x8 Decoder ──► Video
```

**Three separate pipelines:**

| Pipeline | Task |
|----------|------|
| `Text2WorldPipeline` | text → video |
| `Image2WorldPipeline` | text + image → video |
| `Video2WorldPipeline` | text + video → video continuation |

---

### Cosmos 2.5

```
Text Prompt ──► Cosmos Reason VLM (frozen) ──► text_seq (B,L,5120) ──┐
                                             ──► text_pool (B,5120)    │
                                                                       │  adaLN-Zero
Image/Video ──► CV4x16x16 Tokenizer ──► cond_latent ──► VisualPool ──┤  (multi-modal)
                                                      (B,5120)         │
Noise ──► z_t ──►  DiT (24 blocks)  ◄──────────────────────────────────┘
                   │  ChunkedSpaceTimeAttn (O(C·N))
                   │  Per-block cross-attention (text)
                   │  Single unified head
                   ▼
           v(z_t, t, c)
                   │
        Euler/Heun ODE (20 steps, distilled)
                   │
                   ▼
           z₀ ──► CV4x16x16 Decoder ──► Video
```

**Single unified pipeline:**

```python
pipe = UnifiedPipeline()
out = pipe.run(prompt_ids=..., prompt_mask=...)             # text-to-world
out = pipe.run(..., image_cond=image)                       # image-to-world
out = pipe.run(..., video_cond=video)                       # video-to-world
```

---

## Cosmos 1 → Cosmos 2.5: Complete Change Guide

### 1. Video Tokenizer (`tokenizer.py`)

| Property | Cosmos 1 | Cosmos 2.5 |
|---|---|---|
| Spatial compression | **8×** | **16×** |
| Temporal compression | 4× | 4× (unchanged) |
| Total compression | 256× | 1024× |
| `base_channels` | 128 | 192 |
| `channel_multipliers` | `(1,2,4,8)` | `(1,2,4,4,8)` |

**Code change:** The `Encoder` and `Decoder` classes in Cosmos 1 were already
parameterized by `channel_multipliers`, so the adaptation is purely config-driven.
`ContinuousVideoTokenizerV2` inherits Cosmos 1's class and only overrides `__init__`
with the new config.

```python
# Cosmos 1 (cosmos1/tokenizer.py)
tokenizer = ContinuousVideoTokenizer(TokenizerConfig(spatial_compression=8))
z = tokenizer.encode(video)   # shape: (B, 16, T/4, H/8,  W/8)

# Cosmos 2.5 (cosmos2/tokenizer.py) — only config changes, class body unchanged
tokenizer = ContinuousVideoTokenizerV2()
z = tokenizer.encode(video)   # shape: (B, 16, T/4, H/16, W/16)
```

---

### 2. Attention Mechanism (`attention.py`)

| Property | Cosmos 1 | Cosmos 2.5 |
|---|---|---|
| Type | Full 3-D self-attention | Chunked space-time attention |
| Memory | O(N²) | O(C·N) |
| Class | `FullSpaceTimeAttention` | `ChunkedSpaceTimeAttention` |
| New param | — | `chunk_size=8` (latent frames per chunk) |

**What was removed:**
```python
# Cosmos 1 (cosmos1/attention.py)
# Single global softmax over ALL N tokens:
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
attn_weights = F.softmax(attn_weights, dim=-1)       # ← REMOVED in 2.5
out = torch.matmul(attn_weights, v)
```

**What replaced it:**
```python
# Cosmos 2.5 (cosmos2/attention.py)
# Loop over temporal chunks, run local attention within each chunk:
for c in range(num_chunks):
    chunk_mask = (t_idx >= t_start) & (t_idx < t_end)
    q_c, k_c, v_c = q[..., chunk_idx], k[..., chunk_idx], v[..., chunk_idx]
    attn_w = F.softmax(q_c @ k_c.T * scale, dim=-1)  # local attention ← ADDED
    out[..., chunk_idx] = attn_w @ v_c
```

The QKV projections, RoPE injection, and output projection are **unchanged**.

---

### 3. Text Encoder (`text_encoder.py`)

| Property | Cosmos 1 | Cosmos 2.5 |
|---|---|---|
| Model | T5-XXL | Cosmos Reason 1 (VLM) |
| Class | `T5TextEncoder` | `CosmosReasonEncoder` |
| Output dim | 4096 | 5120 |
| HF load call | `T5EncoderModel.from_pretrained(...)` | `AutoModel.from_pretrained(...)` |

**What changed:**
```python
# Cosmos 1 (cosmos1/text_encoder.py)
T5_HIDDEN: int = 4096
from transformers import T5EncoderModel
self._encoder = T5EncoderModel.from_pretrained(self.model_name)
self.proj = nn.Linear(4096, hidden_dim)   # 4096 → hidden_dim

# Cosmos 2.5 (cosmos2/text_encoder.py) — CHANGED
VLM_HIDDEN: int = 5120                    # CHANGED 4096 → 5120
from transformers import AutoModel
self._encoder = AutoModel.from_pretrained(self.model_name)
self.proj = nn.Linear(5120, hidden_dim)   # CHANGED 4096 → 5120
```

**What stayed the same:**
- frozen-weight logic
- mean-pooling over non-padding tokens
- `encode(input_ids, attention_mask) → (seq_emb, pool_emb)` API

---

### 4. Diffusion Transformer (`dit.py`)

#### 4a. Conditioning Module

| Property | Cosmos 1 | Cosmos 2.5 |
|---|---|---|
| Class | `AdaLNZero` | `AdaLNZeroMultiModal` |
| Inputs | time + text_pool | time + text_pool + visual_pool |
| Extra projection | — | `proj_b: Linear(cond_dim, 6*dim)` for visual |

```python
# Cosmos 1 (shared/normalization.py)
cond = time_emb + text_pool        # single conditioning vector
params = self.proj(cond)           # one projection

# Cosmos 2.5 (cosmos2/dit.py) — ADDED visual branch
cond_text   = time_emb + text_pool
cond_visual = self.visual_pool(image_cond)   # ADDED
params = self.proj_a(cond_text) + self.proj_b(cond_visual)
```

#### 4b. Cross-Attention Placement

| Cosmos 1 | Cosmos 2.5 |
|---|---|
| One cross-attention before block 0 | Cross-attention inside every N blocks |

```python
# Cosmos 1 (cosmos1/dit.py)
# Applied ONCE before the block loop:
x = x + self.text_cross_attn(query=self.text_norm(x), key=text_seq, value=text_seq)[0]
for block in self.blocks:
    x = block(x, cond, t_idx, h_idx, w_idx)

# Cosmos 2.5 (cosmos2/dit.py) — cross-attn is INSIDE each block:
for block in self.blocks:
    x = block(x, cond_text, cond_visual, text_seq, t_idx, h_idx, w_idx)
    # block internally calls self.cross_attn(...)
```

#### 4c. Output Heads

| Cosmos 1 | Cosmos 2.5 |
|---|---|
| `head_text2world`, `head_image2world`, `head_video2world` | `out_proj` (single) |

```python
# Cosmos 1 (cosmos1/dit.py) — task-specific heads
head = {"text2world": self.head_text2world, ...}[task]
v = head(x)

# Cosmos 2.5 (cosmos2/dit.py) — REMOVED task heads; single unified head
v = self.out_proj(x)
```

---

### 5. Flow Matching / Noise Schedule (`flow_matching.py`)

| Property | Cosmos 1 | Cosmos 2.5 |
|---|---|---|
| Class | `RectifiedFlow` | `DistilledRectifiedFlow` |
| Inference steps | 35 | **20** (distilled) |
| Guidance scale | 7.0 | **6.0** |
| Distillation loss | — | `distillation_loss()` ← ADDED |
| RL reward weighting | dead code | **actively used** ← CHANGED |

```python
# Cosmos 2.5 (cosmos2/flow_matching.py) — ADDED distillation loss
def distillation_loss(self, student, teacher, z_0, t, text_seq, text_pool):
    z_t, _ = self.add_noise(z_0, t)
    v_student = student(z_t, t, ...)
    with torch.no_grad():
        v_teacher = teacher(z_t, t, ...)
    return (v_student - v_teacher.detach()).pow(2).mean()
```

---

### 6. Pipeline (`pipeline.py`)

| Cosmos 1 | Cosmos 2.5 |
|---|---|
| 3 classes: `Text2WorldPipeline`, `Image2WorldPipeline`, `Video2WorldPipeline` | 1 class: `UnifiedPipeline` |
| `run(..., task="text2world")` | `run(...)` — task auto-detected |
| Conditioning via token prepending | Conditioning via `VisualPoolProjector` → adaLN |

```python
# Cosmos 1 (cosmos1/pipeline.py)
pipe1 = Text2WorldPipeline()
out = pipe1.run(prompt_ids, prompt_mask, T=57, H=320, W=512)

pipe2 = Image2WorldPipeline()
out = pipe2.run(prompt_ids, prompt_mask, image=img, T=57, H=320, W=512)

# Cosmos 2.5 (cosmos2/pipeline.py) — single pipeline
pipe = UnifiedPipeline()
out = pipe.run(prompt_ids, prompt_mask, T=121, H=320, W=512)           # text2world
out = pipe.run(prompt_ids, prompt_mask, image_cond=img, T=121, ...)    # image2world
out = pipe.run(prompt_ids, prompt_mask, video_cond=vid, T=121, ...)    # video2world
```

---

## Summary of All Changes: Cosmos 1 → Cosmos 2.5

| Component | File | Change |
|---|---|---|
| Tokenizer compression | `config.py` | `spatial_compression` 8 → 16 |
| Tokenizer channels | `config.py` | `base_channels` 128 → 192 |
| Tokenizer stages | `config.py` | `channel_multipliers` `(1,2,4,8)` → `(1,2,4,4,8)` |
| Attention class | `attention.py` | `FullSpaceTimeAttention` → `ChunkedSpaceTimeAttention` |
| Attention complexity | `attention.py` | O(N²) → O(C·N) |
| Text encoder model | `text_encoder.py` | T5-XXL (4096-d) → Cosmos Reason VLM (5120-d) |
| `cond_dim` | `config.py` | 4096 → 5120 |
| Conditioning module | `dit.py` | `AdaLNZero` → `AdaLNZeroMultiModal` |
| Visual conditioning | `dit.py` | token prepend → `VisualPoolProjector` |
| Cross-attention | `dit.py` | once before blocks → every N blocks |
| Output heads | `dit.py` | 3 task heads → 1 unified head |
| `max_frames` (RoPE) | `config.py` | 57 → 121 (supports ~30 s) |
| `hidden_dim` | `config.py` | 4096 (7B) → 2048 (2B) or 4096 (14B) |
| Inference steps | `flow_matching.py` | 35 → 20 (distilled) |
| Guidance scale | `flow_matching.py` | 7.0 → 6.0 |
| Distillation loss | `flow_matching.py` | — → `distillation_loss()` ADDED |
| RL reward weighting | `flow_matching.py` | dead code → actively used |
| Number of pipelines | `pipeline.py` | 3 → 1 (`UnifiedPipeline`) |
| `task` argument | `pipeline.py` | required → auto-detected / removed |

---

## Installation

```bash
pip install torch torchvision
pip install transformers   # for T5TextEncoder / CosmosReasonEncoder
```

---

## Quick Usage

```python
import torch
from cosmos.cosmos1.pipeline import Text2WorldPipeline
from cosmos.cosmos2.pipeline import UnifiedPipeline

# ── Cosmos 1 ────────────────────────────────────────────────────────
pipe1 = Text2WorldPipeline(device="cuda")

# Dummy tokenized prompt (replace with real tokenizer output)
prompt_ids  = torch.zeros(1, 32, dtype=torch.long)
prompt_mask = torch.ones(1, 32, dtype=torch.long)

out1 = pipe1.run(
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    T=57, H=320, W=512,
    seed=42,
)
print(out1.video.shape)   # (1, 3, 57, 320, 512)

# ── Cosmos 2.5 ──────────────────────────────────────────────────────
pipe2 = UnifiedPipeline(device="cuda")

out2 = pipe2.run(
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    T=121, H=320, W=512,   # supports longer videos
    seed=42,
)
print(out2.video.shape)   # (1, 3, 121, 320, 512)

# Image-to-world (Cosmos 2.5 only — unified pipeline)
image = torch.randn(1, 3, 1, 320, 512)
out3 = pipe2.run(
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    image_cond=image,
    T=121, H=320, W=512,
)
```

---

## Key References

- **Cosmos 1.0**: [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575)
- **Cosmos 2.5**: [Cosmos-Predict2.5: Improved World Simulation with Video Foundation Models](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/)
- **DiT**: Peebles & Xie, ["Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) (2022)
- **Rectified Flow**: Liu et al., ["Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"](https://arxiv.org/abs/2209.03003) (2022)
- **RoPE**: Su et al., ["RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) (2021)
- **RMSNorm**: Zhang & Sennrich, ["Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467) (2019)
