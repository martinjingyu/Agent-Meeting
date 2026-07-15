# Gallery Display Curator — Round 2 Revised Position

## What I changed and why

### 1. Model availability fix: MobileCLIP-S0 → MobileCLIP2-S0

**What changed**: My Round 1 design assumed MobileCLIP-S0 (~11M params) was available. I verified it is NOT available in the installed open_clip_torch 3.3.0. However, **MobileCLIP2-S0** (74.8M params, under 100M constraint ✓) **is available** with pretrained tag `dfndr2b`. Real benchmark: **0.51s per single forward pass** on this CPU machine (Intel UHD 770).

**Why it matters**: This changes the feasibility equation significantly. The aggregated plan's conclusion that "model inference is too expensive at ~3s/image" was based on MobileCLIP-S2 (99.2M params, 3.75s/forward) and didn't account for MobileCLIP2-S0's better speed. With 0.51s/image encoding, plus preprocessing (~0.1s) and text encoding (one-time ~0.2s), total is ~0.7-0.8s/image. For 15,000 candidates → ~3.5 hours, not 10-15 hours. This makes two-stage heuristic+model pipeline feasible.

### 2. Sharpness threshold calibration

**What changed**: My Round 1 design used a flat threshold of 5 (at 640px scale). My validation showed this is too aggressive — it would reject real photos in datasets like digital_domain (p5=2.3), thema-med (p5=2.3), and truro_school (p5=3.6).

**Revised approach**: Two-tier threshold system:
- **Hard reject floor**: sharpness < 2.0 (catches truly corrupted/blank images)
- **Quality tier**: use per-dataset percentile (p15 as adaptive threshold) for "low quality" filtering, but with a general "potential blur" flag for images between p15 and p5 that go into review pool rather than outright rejection

### 3. Accepting the aggregated plan's diversity algorithm

**What changed**: I now fully endorse the aggregated plan's **dhash-based feature hashing + greedy cluster selection** approach over my original "scene type quota" system. The dhash approach is more lightweight, doesn't need MobileCLIP for scene classification, and is more explainable.

**Why**: My original scene-type classification required 7 additional CLIP prompts per image, adding ~3x more model inference. The dhash feature hashing achieves similar diversity with zero extra inference cost.

### 4. Revised stance on human review for digital_domain

**What changed**: I now agree with the aggregated plan that pure heuristics cannot distinguish high-quality CGI from real photos in digital_domain. MobileCLIP2-S0's zero-shot classification helps, but for datasets mixing CGI/real, we should flag the final output for mandatory human review.

---

## My full current position

### Overall architecture: Two-stage heuristic + model pipeline

```
Stage A: Heuristic pre-filter (all 61,958 images, ~70 min CPU)
  → Fast rejection of obvious non-photos + duplicates
  → Candidate pool: ~12,000-18,000

Stage B: Model-based scoring (candidate pool only, ~3.5 hours CPU)
  → MobileCLIP2-S0 zero-shot real-scene classification
  → Quality score combining heuristic + model signals
  → dhash-based diversity clustering

Stage C: Final selection (~5 min)
  → Greedy cluster selection per dataset
  → Top 100 with diversity constraints
  → Human review pool output
```

### Stage A details (revised from Round 1)

1. **Read & decode**: PIL, max 640px longest side
2. **Hard reject** (immediate removal):
   - Sharpness < 2.0 → R_BLUR (640px scale, catches truly corrupted/dead images)
   - Aspect ratio < 0.3 or > 3.5 → R_EXTREME_AR
   - Mean brightness < 10 or > 248 → R_EXPOSURE
3. **Soft reject** (goes to review pool, not candidate pool):
   - Colorfulness < 5 AND Entropy < 4.0 → R_INFOGRAPHIC (flagged as probable non-photo)
   - Sharpness < per-dataset p5 → R_BLUR_SOFT (possibly blurry photo)
4. **dhash near-duplicate detection**: union-find (disjoint-set) on all images, Hamming ≤ 4 → cluster. Each cluster: keep highest-quality, rest → R_DUPLICATE
5. **Output**: candidate pool (~12k-18k images) + rejected pool + review pool

### Stage B details (revised)

1. **Model**: MobileCLIP2-S0 (74.8M params, `dfndr2b` pretrained) via open_clip
   - CPU inference: ~0.5s/forward (benchmarked)
2. **Zero-shot real-scene classification** (5 real prompts vs 5 non-real prompts):
   - P(real) via softmax on averaged cosine similarities
3. **Quality score** (composite):
   ```
   Q = 0.15 * norm_sharpness + 0.10 * norm_colorfulness 
     + 0.10 * norm_entropy + 0.40 * P(real) 
     + 0.15 * norm_brightness_quality + 0.10 * norm_edge_ratio
   ```
   Where norm_* uses per-dataset min-max normalization
4. **Scene diversity via dhash feature hashing**:
   - Split 64-bit dhash into 4×16-bit sub-hashes as coarse content signatures
   - Greedy selection: iterate candidates by Q descending; skip if the sub-hash bucket is already over-represented in the selected set

### Stage C details

1. Per-dataset: select up to 100 from the diversity-filtered, Q-sorted list
2. **Single-person anti-bias**: if face (Haar cascade) occupies >25% of image area AND only one face detected → apply 0.85 penalty to Q. Cap single-person images at 15% of final set.
3. If <100 high-quality candidates, output actual count with explanation
4. Output: human review pool for P(real) between 0.3-0.5, and Q within 10% of cutoff

### Key empirical data supporting these decisions

| Measure | Value | Source |
|---------|-------|--------|
| MobileCLIP2-S0 params | 74.8M (✓ under 100M) | Verified via open_clip |
| MobileCLIP2-S0 CPU forward | 0.51s | Benchmarked on Intel UHD 770 |
| Sharpness @640px (typical real photo) | 5-35 | 220 samples across 11 datasets |
| Sharpness p5 across datasets | 2.3-8.7 | Worst: thema-med (2.3), digital_domain (2.3) |
| Sharpness p15 (adaptive threshold) | 3.5-10 | For quality filtering |

### Where I agree with aggregated plan vs. diverge

**Agree**: 
- dhash-based diversity clustering (replacing my scene-type quota system)
- Per-dataset percentile thresholds (improving on my flat thresholds)
- Conservative bias: borderline → review pool, not final set
- Union-find for global duplicate detection (my sliding window was suboptimal)
- Cannot handle copyright/brand/legal review

**Diverge**:
- **Model feasibility**: Aggregated plan abandoned models entirely ("pure heuristic only"). My benchmarks show MobileCLIP2-S0 is viable at 74.8M params and ~0.5s/forward. The extra ~3.5 hours for Stage B is worth the improved real-scene discrimination, especially for mixed-content datasets (digital_domain, roland_berger, tuv_rheinland).
- **Sharpness floor**: Aggregated plan says threshold=5. My data shows threshold=2 is safer (many real photos are 2-5 at 640px). Use percentile-based thresholds for quality, not a hard floor of 5.
- **truro_school strategy**: Aggregated plan proposes stratified sampling. I agree Stage A runs on all 36k, but Stage B should still run on all candidates (~8k-10k expected after Stage A). At 0.5s/image that's ~1.5 hours — acceptable.

### Conclusion paragraph (for final plan)

This design implements a three-stage pipeline for selecting gallery-worthy real-scene images from 11 datasets (61,958 total images). Stage A uses efficient PIL-based heuristics (Sobel sharpness, Hasler-Susstrunk colorfulness, image entropy, dhash, aspect ratio) with **per-dataset adaptive percentile thresholds** to filter out ~70-80% of non-photo, low-quality, and duplicate content in approximately 70 minutes of CPU time. Stage B applies MobileCLIP2-S0 (74.8M parameters, verified under the 100M constraint, 0.51s/forward on Intel UHD 770) for zero-shot real-scene classification and composite quality scoring on the remaining ~12,000-18,000 candidates, completing in approximately 3.5 hours. Stage C uses dhash-based feature hashing for diversity-driven greedy selection, with a single-person anti-bias penalty (0.85× for dominant face regions) to ensure the final per-dataset Top 100 favors landscapes, architecture, group activities, and interior spaces over individual portraits. All outputs are structured CSV with per-image traceability (rejection code, P(real), quality score, diversity cluster, final rank) and include a human review pool for borderline cases. Total estimated runtime: 4.5-5 hours CPU-only. Key risks: CGI/AI-generated images may still pass MobileCLIP2-S0 detection (mitigated by mandatory human review flag for digital_domain); sharpness threshold calibration requires first-run statistics (mitigated by computing per-dataset percentiles adaptively). Legal compliance (copyright, brand, likeness) is not addressed by this pipeline.
