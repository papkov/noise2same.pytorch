# SwinIA v2 Design Document

## Code Audit, Literature Verification, and Implementation Roadmap

**Date:** 2026-03-06
**Branch:** `claude/swiniu-v2-audit-design-ckMTN`

---

## Table of Contents

1. [Code Audit: Bugs and Issues in Current SwinIA v1](#1-code-audit-bugs-and-issues-in-current-swinia-v1)
2. [Literature Verification and Corrections](#2-literature-verification-and-corrections)
3. [Design Assessment of Option A ("SwinIA v2 Pure")](#3-design-assessment-of-option-a)
4. [Implementation Phases](#4-implementation-phases)
5. [Risk Analysis](#5-risk-analysis)

---

## 1. Code Audit: Bugs and Issues in Current SwinIA v1

### 1.1 Critical Issues

#### BUG-1: SwinIA forward_masked bypasses masking entirely
**File:** `noise2same/model.py:216-217`
**Severity:** Critical (affects training correctness)

```python
if isinstance(self.net, SwinIA):
    return self.forward_whole(x, convolve, crops, full_size_image)  # mask ignored!
```

When SwinIA is used, `forward_masked()` calls `forward_whole()` without applying
any input mask. This means `out_mask` and `out_raw` are produced from **identical
inputs**. During training with dropout (attn_drop=0.05), the invariance loss
`||out_raw - out_mask||^2` measures only **dropout variance**, not meaningful
blind-spot invariance.

**Impact:** The Noise2Same invariance loss degenerates for SwinIA. With
`lambda_inv=2.0` (the default), significant gradient signal comes from a loss
term that is essentially random noise. This likely explains why "SwinIA v1 tried
Noise2Same training but it failed" (as noted in the proposal appendix reference).

**Recommendation:** This is a design choice (SwinIA uses architectural masking
rather than input masking), but the invariance loss computation needs to be
reconsidered. Either:
- Set `lambda_inv=0` when using SwinIA in Noise2Self mode
- Or implement a proper two-pass scheme for Noise2Same mode

#### BUG-2: Parallel encoder does not evolve query representations
**File:** `noise2same/backbone/swinia.py:332-342`
**Severity:** High (architectural limitation, not a code bug per se)

```python
mid = len(self.groups) // 2
for i, group in enumerate(self.groups):
    if i < mid:
        q_ = group(q, k, v)       # q_ saved, but q NOT updated
        shortcuts.append(q_)
    elif shortcuts:
        q = group(q, k, v)        # decoder updates q
        q = connect_shortcut(self.shortcut, q, shortcuts.pop())
    else:
        q = group(q, k, v)
```

During the encoder phase (`i < mid`), the query `q` is never updated. Each
encoder group independently receives the same positional embedding as query and
the same frozen K/V from input. The encoder groups are truly **parallel**, not
hierarchical. Only the decoder groups form a sequential chain.

This means:
- No hierarchical feature extraction in the encoder
- All encoder shortcuts are computed from the same initial representation
- The symmetric shuffle pattern `[1, 2, 4, 2, 1]` creates multi-scale views
  but these are independent, not refined

**Impact:** Limited representational capacity. TBSN's ablation confirms: their
sequential encoder with evolving Q/K/V achieves +0.59 dB over SwinIA's approach.

#### BUG-3: Non-standard attention scaling
**File:** `noise2same/backbone/swinia.py:73`
**Severity:** Medium

```python
self.scale = head_dim ** -0.5 / shuffle
```

The attention scale factor divides by `shuffle` (the patch size). With the default
config `shuffles=[1, 2, 4, 2, 1]`, the middle group divides by 4, making attention
logits 4x smaller than standard. This destabilizes training because:
- Softmax over smaller logits produces more uniform attention weights
- Different groups operate at fundamentally different attention "temperatures"

Standard transformers use `head_dim ** -0.5` only. The proposal correctly
identifies this issue.

### 1.2 Moderate Issues

#### BUG-4: Missing config array length validation
**File:** `noise2same/backbone/swinia.py:303`
**Severity:** Medium (silent misconfiguration)

```python
for i, (d, n, dl, sh) in enumerate(zip(depths, num_heads, dilations, shuffles)):
```

`zip()` silently truncates to the shortest array. If `depths` has 5 elements but
`shuffles` has 4, the model silently creates 4 groups instead of 5. No error or
warning is raised.

**Fix:**
```python
assert len(depths) == len(num_heads) == len(dilations) == len(shuffles), \
    f"Config array lengths must match: {len(depths)}, {len(num_heads)}, ..."
```

#### BUG-5: Scheduler can be None but save_model always calls state_dict()
**File:** `noise2same/trainer.py:277`
**Severity:** Medium (crashes checkpoint saving)

```python
torch.save({
    "model": self.inner_model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
    "scheduler": self.scheduler.state_dict(),  # crashes if None
}, ...)
```

**Fix:**
```python
"scheduler": self.scheduler.state_dict() if self.scheduler else None,
```

#### BUG-6: Evaluator does not pad inputs for SwinIA
**File:** `noise2same/evaluator.py:48-50`
**Severity:** Medium (inference-time errors)

```python
if isinstance(self.model.net, UNet):
    self.resizer = PadAndCropResizer(mode="reflect", div_n=2 ** self.model.net.depth)
elif isinstance(self.model.net, SwinIR):
    self.resizer = PadAndCropResizer(mode="reflect", div_n=self.model.net.window_size)
else:
    self.resizer = PadAndCropResizer(div_n=1)  # SwinIA falls here
```

SwinIA requires inputs divisible by `window_size * shuffle * dilation`, but the
evaluator uses `div_n=1`. This can cause shape mismatches at inference time for
non-standard image sizes.

**Fix:** Add SwinIA branch with appropriate `div_n`.

#### BUG-7: Wasted computation reversing frozen K/V
**File:** `noise2same/backbone/swinia.py:131, 224-225`
**Severity:** Low (performance, not correctness)

After attention, `key` and `value` are passed through `head_partition_reversed`,
`window_partition_reversed`, and `shift_image_reversed` even though they were
never modified. This is wasted computation. Since K/V are frozen from input and
reused across groups, they should be partitioned once and stored, or the reverse
operations should be skipped for K/V.

### 1.3 Code Quality Issues

| Location | Issue |
|----------|-------|
| `model.py:61` | TODO "understand stride" in DonutMask |
| `evaluator.py:121` | Comment `# СТЫД` ("shame") — unprofessional |
| `evaluator.py:83` | TODO "remove randomness" in masked evaluation |
| `trainer.py:89` | TODO "gradient accumulation" not implemented |
| `model.py:136` | TODO "parametrize" GaussianBlur kernel |
| Various | 20+ TODO comments indicating incomplete features |

### 1.4 Confirmed Design Decisions (Not Bugs)

- **Frozen K/V from input**: Intentional — provides structural prior, prevents
  latent collapse, biases toward feature-level denoising
- **SwinIA bypasses input masking**: Intentional — architectural blind-spot via
  attention mask makes input masking redundant
- **Loss computed over all pixels for SwinIA** (`mask=None`): Follows from the
  above — since all pixels are structurally blind-spotted, all contribute to loss
- **Symmetric shuffle pattern**: Creates multi-scale receptive fields matching
  encoder-decoder symmetry

---

## 2. Literature Verification and Corrections

### 2.1 Verified Papers

#### TBSN — CONFIRMED (AAAI 2025)
- **Actual title:** "Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising"
- **Authors:** Nage et al. | **arXiv:** 2404.07846
- **GitHub:** https://github.com/nagejacob/TBSN

**Verified claims:**
- M-WSA with even-coordinate masking: **CORRECT** — "the query token attends to
  spatial locations at even coordinates", M(i,j)=0 when xi-xj ≡ yi-yj ≡ 0 (mod 2)
- G-CSA (Grouped Channel Self-Attention): **CORRECT**
- Centrally-masked 3x3 conv as first layer: **CORRECT** — "TBSN follows dilated
  BSN to apply 3×3 centrally masked convolution at the first layer"
- 0.59 dB drop when replacing M-WSA with SwinIA: **CORRECT** — confirmed in
  Table 3 ablation
- Knowledge distillation (TBSN2UNet): **CORRECT** — distilled model achieves
  37.79 dB

**CORRECTION in proposal:** Section 6 states "TBSN gets ~34 dB on SIDD real-world
noise." This is **WRONG**. TBSN achieves **37.78 dB** on SIDD benchmark. The
34.10 dB figure belongs to UBSN (see below). This error significantly
misrepresents the current SOTA gap.

**Ablation details (Table 3, SIDD validation):**
- Base (dilated conv only): 36.90 dB
- + Dilated M-WSA: 37.17 dB (+0.27)
- + Dilated G-CSA: 37.55 dB (+0.65)
- Full TBSN (both): 37.71 dB (+0.81 total)

Note: G-CSA contributes **more** than M-WSA in isolation (+0.65 vs +0.27), which
suggests it should be prioritized alongside M-WSA in v2.

#### UBSN (not "U-BSN") — CONFIRMED (WACV 2025)
- **Actual title:** "Design Principles of Multi-Scale J-Invariant Networks for Self-Supervised Image Denoising"
- **Authors:** Yu et al.
- **PDF:** https://openaccess.thecvf.com/content/WACV2025/papers/Yu_Design_Principles_of_Multi-Scale_J-Invariant_Networks_for_Self-Supervised_Image_Denoising_WACV_2025_paper.pdf

**Verified claims:**
- Sequential Rule and Aggregation Rule theorems: **CORRECT** — formal theorems
  for multi-scale J-invariant network design
- Randomized PD (random stride from {2,3,4,5} + permutation): **CORRECT**
- Architecture uses blind-spot conv only at skip connections: **CORRECT**

**NOTE:** The paper is named "UBSN", not "U-BSN". The name "U-BSN" used in the
proposal does not appear in search results. The 34.10 dB / 1.38M params claim
could not be independently verified from web search alone — please cross-check
against the paper's Table directly.

#### AT-BSN — CONFIRMED (CVPR 2024)
- **Actual title:** "Exploring Efficient Asymmetric Blind-Spots for Self-Supervised Denoising in Real-World Scenarios"
- **Authors:** Chen et al.
- **GitHub:** https://github.com/hnmizuho/AT-BSN

**Verified claims:**
- Asymmetric blind-spots (different sizes for train vs inference): **CORRECT**
- Venue CVPR 2024: **CORRECT**

**SIDD results:**
- AT-BSN: 36.73 dB (benchmark), 36.80 dB (validation)
- AT-BSN (D, direct training): 37.77-37.78 dB (benchmark), 37.88 dB (validation)

#### AMSNet — CONFIRMED but WRONG VENUE
- **Actual title:** "Asymmetric Mask Scheme for Self-Supervised Real Image Denoising"
- **Authors:** Liao, Zheng, Zhong, Zhang, Ren
- **GitHub:** https://github.com/lll143653/amsnet

**CORRECTION:** The proposal says "AMSNet (arXiv 2024)". This is **WRONG** —
AMSNet was published at **ECCV 2024** (European Conference on Computer Vision),
a top-tier peer-reviewed venue. This significantly strengthens its credibility
versus being "just an arXiv preprint."

**Verified claims:**
- Input masking approach abandoning BSN constraints: **CORRECT** — "randomly
  masks some pixels via a 0,1 mask matrix M" at input level
- Any backbone can be used: **CORRECT**
- Multiple inference passes needed: **CORRECT**

**SIDD results:** AMSNet-P-E achieves **37.93 dB (validation) / 37.87 dB
(benchmark)** — making it competitive with TBSN.

#### SwinIA — CONFIRMED (WACV 2025)
- **Actual title:** "SwinIA: Self-Supervised Blind-Spot Image Denoising without Convolutions"
- **Authors:** Papkov et al. | **Venue:** WACV 2025
- **PDF:** https://openaccess.thecvf.com/content/WACV2025/papers/Papkov_SwinIA_Self-Supervised_Blind-Spot_Image_Denoising_without_Convolutions_WACV_2025_paper.pdf

### 2.2 Unverified / Partially Verified Papers

#### Rotation-Equivariant BSN (CVPR 2025)
- Could not independently verify from web search. CVPR 2025 papers may not be
  fully indexed yet. The claims about rotation-equivariant convolutions for BSN
  are plausible but should be verified against the actual paper.

#### Zero-Shot BSN via INR (CVPR 2025)
- Same situation — CVPR 2025 papers not yet fully indexed. Claims are plausible.

#### Complementary-BSN (2025)
- Mentioned in the proposal title but never described. Should be either detailed
  or removed.

### 2.3 Corrected SOTA Landscape (SIDD Benchmark)

| Method | Venue | SIDD (dB) | Params |
|--------|-------|-----------|--------|
| AMSNet-P-E | ECCV 2024 | 37.87 | - |
| TBSN2UNet | AAAI 2025 | 37.79 | ~4x fewer |
| TBSN | AAAI 2025 | 37.78 | - |
| AT-BSN (D) | CVPR 2024 | 37.77 | - |
| UBSN | WACV 2025 | 34.10* | 1.38M |
| SwinIA | WACV 2025 | - | - |

*UBSN's 34.10 needs direct verification. If correct, there's a ~3.7 dB gap
vs TBSN, suggesting UBSN trades quality for extreme efficiency.

### 2.4 Key Insight from Literature Verification

The current SOTA BSN performance on SIDD is ~37.8 dB (TBSN, AT-BSN). The
proposal's Section 6 estimated improvements should be recalibrated against
this number, not against "~34 dB":

- SwinIA v1 on BSD68 (σ=25): 30.01 dB — this is a **synthetic noise** result
- TBSN on SIDD: 37.78 dB — this is a **real-world noise** result

These numbers are not directly comparable. The expected improvement from v2
should be benchmarked separately for synthetic noise (BSD68) and real-world
noise (SIDD).

---

## 3. Design Assessment of Option A

### 3.1 Overall Assessment

Option A ("SwinIA v2 Pure") is **well-founded** and addresses the two critical
limitations identified by TBSN (frozen K/V, parallel encoder). The proposal
correctly identifies the key changes needed. Below are specific technical notes.

### 3.2 Design Corrections

#### 3.2.1 Cross-window communication claim
The proposal Section 3.1 states: "No cyclic shifting is needed as in original
Swin Transformer — the dilated pattern inherently provides cross-window
communication."

This is **INCORRECT**. Dilated attention within a fixed window does NOT provide
cross-window communication. The dilated mask only determines which positions
within the window are attended. Windows remain isolated without shifting.

The proposal then contradicts itself: "However, I'd recommend KEEPING the Swin
cyclic shift as well." This second recommendation is correct. **Cyclic shifting
is essential** for cross-window information flow and should be kept.

#### 3.2.2 G-CSA code has parameter confusion
The proposal's G-CSA code in Section 3.2 accepts both `num_groups` and
`num_heads` but uses `num_heads` only for the temperature parameter shape while
all computation is grouped by `num_groups`. The `num_heads` parameter is unused
in the actual attention computation. Either:
- Remove `num_heads` and use `num_groups` for temperature
- Or implement multi-head attention within each group

#### 3.2.3 Dilated mask interaction with shifted windows
When combining TBSN's dilated mask with Swin's shifted windows, the mask needs
careful handling. In shifted windows, the shift boundary creates regions that
should not cross-attend (handled by the shift mask). The dilated mask must be
applied **in addition to** the shift mask, not instead of it. The combined mask
should be:

```python
combined_mask = dilated_mask + shift_mask  # both use -inf for blocking
```

This is analogous to how the current SwinIA combines the diagonal mask with the
shift mask (line 126 in swinia.py).

#### 3.2.4 Blind-spot attention at skip connections
The proposal recommends transformer blocks with dilated attention mask at skip
connections (following UBSN's Theorem 1). This is a good idea but needs
clarification:

UBSN uses a single blind-spot **convolution** at skips (one layer, not a full
transformer block). For a convolution-free design, using a full transformer
block is more expensive. Consider:
- A single M-WSA layer (no MLP) at each skip connection
- This provides the minimum J-invariance guarantee per Theorem 1 while keeping
  cost low

#### 3.2.5 Patch unshuffle and channel dimension
After `PixelUnshuffle(2)`, channels increase 4x. The proposal mentions a linear
projection to adjust channels. Be aware that this projection allows cross-channel
mixing, which could leak blind-spot information (this is exactly what G-CSA
addresses). The projection should either:
- Use grouped linear projection (matching G-CSA groups)
- Or be followed by G-CSA to control information flow

### 3.3 What I Recommend Adding

#### 3.3.1 Fix the invariance loss for Noise2Same mode
If pursuing unified BSN + Noise2Same (Section 5.3 of proposal), the current
`forward_masked` for SwinIA needs to actually produce different outputs. Options:
- Apply input masking AND architectural masking (belt and suspenders)
- Use dropout only during masked pass, not during raw pass
- Use different attention mask dilation for the two passes

#### 3.3.2 Progressive unblinding
Instead of binary mask removal at inference, consider:
1. Train with full dilated mask
2. Gradually reduce mask strength during late training epochs
3. This is smoother than the proposed learned-α approach and doesn't require
   a held-out set

#### 3.3.3 Explicit parameter budget
The proposal increases embed_dim from 144 to 192 and adds G-CSA + more decoder
blocks. Estimate the parameter count:

| Component | SwinIA v1 | SwinIA v2 (est.) |
|-----------|-----------|------------------|
| Embedding | ~0.03M | ~0.04M |
| Per block | ~0.12M | ~0.25M (with G-CSA) |
| Blocks | 5 groups × 4 = 20 | 3 enc + 3 dec × 4 = 24 |
| Skip proj | ~0.04M | ~0.15M (with blind-spot attn) |
| **Total** | **~2.5M** | **~6-8M** |

This 3x parameter increase should be justified by proportional quality gains.

---

## 4. Implementation Phases

### Phase 0: Pre-requisite Bug Fixes (1-2 days)

Fix existing bugs before starting v2 development. These affect both v1 and v2.

| Task | File | Priority |
|------|------|----------|
| Add config array length validation | `swinia.py:303` | P0 |
| Fix scheduler None crash in save_model | `trainer.py:277` | P0 |
| Add SwinIA to evaluator resizer | `evaluator.py:48` | P0 |
| Remove wasted K/V reverse operations | `swinia.py:131,224` | P1 |
| Clean up TODO comments | Various | P2 |

### Phase 1: Core Architecture — Sequential U-Net with Evolving Q/K/V (1-2 weeks)

**Goal:** Replace parallel encoder + frozen K/V with sequential U-Net encoder-
decoder where Q, K, V all evolve.

**New file:** `noise2same/backbone/swinia_v2.py`

**Step 1.1:** Implement the dilated attention mask (M-WSA)
```python
def build_dilated_mask(M: int) -> torch.Tensor:
    """TBSN-style mask for window size M.
    Each pixel attends to pixels at even coordinate offsets, excluding self."""
    coords = torch.stack(torch.meshgrid(
        torch.arange(M), torch.arange(M), indexing='ij'
    ), dim=-1).reshape(-1, 2)
    diff = coords[:, None] - coords[None, :]  # (M^2, M^2, 2)
    even_offset = (diff[..., 0] % 2 == 0) & (diff[..., 1] % 2 == 0)
    not_self = ~torch.eye(M * M, dtype=torch.bool)
    mask = torch.where(even_offset & not_self, 0.0, float('-inf'))
    return mask
```
- Verify: for M=8, each query attends to ~15 keys (M^2/4 - 1)
- Combine with Swin shift mask via addition

**Step 1.2:** Replace `DiagonalWindowAttention` with `DilatedWindowAttention`
- Q, K, V all projected from current features (single `nn.Linear(dim, 3*dim)`)
- Standard scaling: `head_dim ** -0.5` (remove shuffle-dependent scaling)
- Increase dims per head: 32 (from 9)
- Keep relative position bias

**Step 1.3:** Implement `SwinIA_v2` with sequential U-Net structure
- 3-level encoder-decoder
- Pixel unshuffle/shuffle for down/upsampling
- Linear channel projection after unshuffle (grouped to preserve blind-spot)
- Skip connections with single M-WSA layer for J-invariance

**Step 1.4:** Update `model.py` to handle SwinIA_v2
- Register new backbone type
- Handle loss computation (SwinIA_v2 uses structural masking like v1)

**Validation:** Train on BSD68 σ=25. Target: match or exceed SwinIA v1 (30.01 dB).
Expected: +0.5 to +1.0 dB from evolving K/V + sequential encoder.

### Phase 2: Grouped Channel Self-Attention (3-5 days)

**Goal:** Add G-CSA to prevent channel-dimension blind-spot leakage at multi-scale
levels.

**Step 2.1:** Implement `GroupedChannelAttention`
- Restormer-style transposed attention (Q·K^T in channel dim)
- Grouped: `G` groups of `C/G` channels each
- Constraint: `C/G < H*W` at each scale level

**Step 2.2:** Integrate into transformer blocks
- Block structure: LayerNorm → G-CSA → LayerNorm → M-WSA → LayerNorm → MLP
- Configure group sizes per encoder level based on spatial resolution

**Step 2.3:** Ablate G-CSA contribution
- Train with/without G-CSA
- Expected: +0.3 to +0.7 dB (based on TBSN ablation showing +0.65 dB)

**Validation:** BSD68 σ=25 ablation. Also test on SIDD if real-world pipeline ready.

### Phase 3: Unblinding Mechanism (3-5 days)

**Goal:** Implement learned unblinding for inference.

**Step 3.1:** Implement learned diagonal bias
```python
# Per-layer learned bias (initialized to 0)
self.unblind_alpha = nn.Parameter(torch.zeros(1))

# During inference:
if not self.training:
    attn = attn + self.unblind_alpha * torch.eye(N, device=attn.device)
```

**Step 3.2:** Two-stage training protocol
1. Stage 1: Train with full blind-spot mask (normal BSN training)
2. Stage 2: Short fine-tuning (10-20% of total epochs) with unblinding enabled
   and reduced learning rate

**Step 3.3:** Compare unblinding strategies
- Simple mask removal (baseline, as in v1)
- Learned scalar alpha (recommended)
- Per-head alpha (more flexible)

**Validation:** Measure gain from unblinding on BSD68 and SIDD.
Expected: +0.1 to +0.3 dB over mask removal.

### Phase 4: Real-World Noise — Randomized PD Wrapper (3-5 days)

**Goal:** Support spatially correlated noise (SIDD-style datasets).

**Step 4.1:** Implement Randomized Pixel-shuffle Downsampling
```python
class RandomizedPD:
    def __init__(self, strides=(2, 3, 4, 5)):
        self.strides = strides

    def __call__(self, x):
        stride = random.choice(self.strides)
        # Pixel-shuffle downsample
        # Intrapatch permutation
        # Patchwise rotation (0/90/180/270)
        return x_pd, reverse_fn
```

**Step 4.2:** Integrate as preprocessing wrapper in training pipeline
- Applied before network forward pass
- Reversed after network output
- At inference: fixed stride=2, average over 2 runs

**Step 4.3:** Add SIDD dataset configuration
- Config already exists (`config/experiment/sidd.yaml`)
- May need updates for PD wrapper integration

**Validation:** Train on SIDD. Compare with and without Randomized PD.
Expected: +0.5 to +1.5 dB on real-world noise.

### Phase 5: Research Extensions (2-4 weeks, lower priority)

#### 5.1 Unified BSN + Noise2Same Loss
- Fix `forward_masked` to produce genuinely different outputs for SwinIA
- Implement combined loss: `L = L_rec + lambda * sqrt(L_inv)`
- Requires careful design to avoid degenerate solutions

#### 5.2 Knowledge Distillation (TBSN2UNet approach)
- Train SwinIA v2 as teacher
- Distill into plain U-Net for inference speed
- Expected: ~4x speedup with minimal quality loss

#### 5.3 Mamba/SSM Variant
- Experimental: replace attention with state-space model
- Requires defining "blind-spot" for sequential scan
- High-risk, high-reward research direction

---

## 5. Risk Analysis

### 5.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dilated mask doesn't maintain J-invariance in sequential U-Net | Low | Critical | TBSN proves this works; verify with unit test |
| G-CSA group size constraint hard to satisfy at small spatial resolutions | Medium | Medium | Adaptive group sizing; fall back to no grouping at bottleneck |
| Learned unblinding overfits to validation set | Medium | Low | Use separate held-out set; early stopping on alpha |
| Randomized PD increases training time significantly | High | Medium | Implement efficient batched PD; parallelize stride variants |
| Parameter count too high for GPU memory | Medium | Medium | Gradient checkpointing; reduce embed_dim to 144 if needed |

### 5.2 Novelty Assessment for Publication

The proposal's Section 5 identifies four novelty claims. Current assessment:

1. **First fully convolution-free sequential U-Net BSN** — **Strong claim.**
   TBSN uses centrally-masked conv. If Option A works without any conv, this
   is genuinely novel. Risk: the first layer may need convolution for stability.

2. **Learnable unblinding** — **Moderate claim.** Novel but incremental.
   Likely +0.1-0.2 dB, which may not justify a standalone contribution.
   Better as part of a larger system.

3. **Unified BSN + Noise2Same** — **Strong claim if it works.** Previous
   attempt failed (SwinIA v1 appendix). Success with v2 would be significant.
   Risk: may still fail for fundamental reasons.

4. **SSM/Mamba variant** — **Very strong claim but very high risk.** Completely
   unexplored territory. Would be a top venue paper if it works but requires
   substantial theoretical groundwork.

**Recommended novelty strategy:** Combine claims 1 + 2 as the core contribution,
with 3 as the stretch goal. Defer 4 to a separate project.

### 5.3 Corrected Performance Expectations

The proposal's Section 6 estimates should be revised:

| Change | Proposal estimate | Revised estimate | Basis |
|--------|------------------|------------------|-------|
| Deep K/V | +0.5 to +1.0 dB | +0.3 to +0.6 dB | TBSN ablation shows 0.59 dB for full M-WSA vs SwinIA |
| Sequential encoder | +0.2 to +0.5 dB | +0.1 to +0.3 dB | Included in the 0.59 above |
| G-CSA | +0.3 to +0.7 dB | +0.3 to +0.7 dB | TBSN ablation shows +0.65 dB (validated) |
| Larger head dim | +0.1 to +0.3 dB | +0.1 to +0.2 dB | Minor contribution |
| Learned unblinding | +0.1 to +0.2 dB | +0.1 to +0.2 dB | Reasonable |
| Randomized PD | +0.5 to +1.5 dB | +0.5 to +1.5 dB | SIDD-specific, valid |

**Realistic total for synthetic noise (BSD68):** +0.8 to +1.5 dB over v1
**Realistic total for real-world noise (SIDD):** +1.5 to +3.0 dB including PD

---

## Appendix A: File Map for Implementation

```
noise2same/backbone/
    swinia.py          — Current v1 (keep as-is for comparison)
    swinia_v2.py       — NEW: v2 architecture (Phase 1-3)
    __init__.py         — Add SwinIA_v2 export

noise2same/
    model.py           — Update for v2 backbone registration
    trainer.py         — Bug fixes (Phase 0)
    evaluator.py       — Add SwinIA_v2 resizer support
    randomized_pd.py   — NEW: Randomized PD wrapper (Phase 4)

config/backbone/
    swinia_v2.yaml     — NEW: v2 configuration

tests/
    test_dilated_mask.py  — NEW: verify J-invariance of dilated mask
    test_swinia_v2.py     — NEW: forward pass, gradient flow, mask correctness
```

## Appendix B: Verified Reference Links

- TBSN: https://arxiv.org/abs/2404.07846 | https://github.com/nagejacob/TBSN
- UBSN: https://openaccess.thecvf.com/content/WACV2025/papers/Yu_Design_Principles_of_Multi-Scale_J-Invariant_Networks_for_Self-Supervised_Image_Denoising_WACV_2025_paper.pdf
- AT-BSN: https://arxiv.org/abs/2303.16783 | https://github.com/hnmizuho/AT-BSN
- AMSNet: https://arxiv.org/abs/2407.06514 | https://github.com/lll143653/amsnet
- SwinIA: https://openaccess.thecvf.com/content/WACV2025/papers/Papkov_SwinIA_Self-Supervised_Blind-Spot_Image_Denoising_without_Convolutions_WACV_2025_paper.pdf
