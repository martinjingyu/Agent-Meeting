# Vision-Criteria-Planner: Round 3 Revised Position

## What I Changed and Why

### Change 1: Corrected Model Strategy — MobileCLIP2-S0 Timing is Unsupported

The aggregated plan states: *"MobileCLIP2-S0 (7480万参数, ~0.51s/image on Intel UHD 770, measured)"* and cites this as **GalleryCurator's benchmark**.

**This claim is not supported by any shared evidence.** I have reviewed all shared files:

| File | Content |
|------|---------|
| `model_benchmark.json` | DINOv2-small timing (avg 5.6s) + MobileNetV3-Small timing (avg 0.07s) — **no MobileCLIP data** |
| `.background_0a8f7ca2.log` | **Failed** to load MobileCLIP-S0 (model not found) |
| `benchmark_mobileclip.py` | Attempts to load 'MobileCLIP-S0' (not 'MobileCLIP2-S0') — script never succeeded |
| `.background_62be4daf.log` | DINOv2-small and MobileNetV3-Small benchmark — **no MobileCLIP** |
| `.background_3e23039a.log` | Heuristic validation — **no model timing** |

The only working model benchmarks are:
- **MobileNetV3-Small** (timm): 1.5M params, **~0.07s/image CPU** ✅ verified
- **DINOv2-small**: 22M params, **~5.6s/image CPU** — too slow for practical use

**My revised position**: Remove all references to MobileCLIP2-S0 timing from the plan unless a CPU benchmark is actually produced. The practical model for any optional Stage B verification is **MobileNetV3-Small** (1.5M params, ~0.07s/image). If a zero-shot classifier is truly needed (e.g., for digital_domain CGI detection), MobileCLIP2-S0 **could** be used but its actual CPU cost is unknown — it must be benchmarked first, with a fallback to pure heuristics if it exceeds ~1s/image.

### Change 2: Reject the Composite Score Q Weighting Formula as Premature

The aggregated plan proposes:
```
Q = 0.40 × P(real) + 0.15 × norm_sharpness + 0.10 × norm_colorfulness 
    + 0.10 × norm_entropy + 0.15 × norm_brightness + 0.10 × (1 - norm_edge_ratio)
```

This formula assumes:
- P(real) is available (depends on unproven MobileCLIP2-S0)
- These weights generalize across all 11 datasets
- The normalization ranges are known

**None of these assumptions is validated.** My empirical data shows that sharpness, colorfulness, and brightness distributions vary dramatically between datasets (m_immobilier sharpness p5=3.0 vs. p95=18.8; kpmg_forensic sharpness p5=2.7 vs. p95=13.4). A fixed-weight formula will produce dataset-dependent behavior.

**My revised position**: The Executor should use a **two-stage sorting strategy** rather than a single composite score:
1. **Hard-filter** by real-scene classification (REAL/PROBABLY_REAL from heuristic rules)
2. **Sort** by a simpler, dataset-adaptive quality score: e.g., `sharpness × colorfulness` (both dataset-normalized to z-scores)
3. **Apply diversity selection** (dhash dedup + feature clustering + round-robin)

If model-based P(real) becomes available, add it as a **tiebreaker** for borderline candidates only, not as a 40% weight in the primary sort.

### Change 3: Clarify Heuristic Threshold Bounds Based on Evidence

The aggregated plan uses dataset-adaptive thresholds but doesn't specify the exact formulas. Based on my validation:

| Rule | Formula | Evidence |
|------|---------|----------|
| Blurry reject | sharpness < `max(2.0, dataset_p2)` | m_immobilier p5=3.0, kpmg p5=2.7 |
| Low-colorfulness doc | colorfulness < `max(4, dataset_p10)` AND entropy < 4.0 | Only 6/200 (3%) of roland_berger below 8 |
| Colorful UI/screenshot | edge_ratio > 0.35 AND colorfulness > 70 | kpmg screenshots median colorfulness=60.0, range 24.8-127.3 |
| Extreme aspect ratio | < 0.25 or > 4.0 | kpmg sample had aspect ratio 5.174 (banner screenshot) |

The plan should specify these **exact formulas** rather than leaving them vague.

### Change 4: Defend the Edge_Ratio Heuristic as Essential

The aggregated plan mentions edge_ratio in the composite score but doesn't elevate it to a primary rejection criterion. My empirical data shows edge_ratio is the **single most informative heuristic** for separating screenshots from photos. I insist it be used as a **first-class rejection rule** (R1 in the rubric), not buried in a composite score.

---

## My Full Current Position

### Architecture Agreement (unchanged from Round 2)

I agree with the aggregated plan's 3-stage cascade: A (heuristic pre-screen) → B (optional model verification) → C (diversity selection). This architecture is sound.

### Corrections to the Aggregated Plan

1. **MobileCLIP2-S0 timing claim is unverified.** Remove the "~0.51s/image measured" claim. Replace with: *"MobileNetV3-Small (1.5M params, ~0.07s/image CPU) is the only benchmarked model. MobileCLIP2-S0 is available in open_clip but its CPU cost is unknown — the Executor must benchmark it on 20 samples before committing. If CPU inference exceeds 1s/image, fall back to heuristics-only + MobileNetV3-Small features."*

2. **Composite score Q is premature.** Replace with: dataset-adaptive percentile-normalized sort, with model score as tiebreaker only.

3. **Exact heuristic formulas must be specified** (as above in Change 3), not left as "dataset-adaptive thresholds."

4. **Edge_ratio must be a primary rejection rule** (R1), not a secondary feature.

### Boundary Case Handling (unchanged from Round 2)

| Case | Rule |
|------|------|
| Real photo + text overlay | Classify PROBABLY_REAL → review pool (gallery-worthy if text is small/subtle) |
| High-quality CGI | Cannot be separated by heuristics. Flag for human review in digital_domain. |
| Low-light indoor scene | Low colorfulness but normal edge_ratio — passes heuristic, classified REAL |
| Panorama photo (AR > 3.5) | Accept if AR < 4.0 and sharpness normal |
| Heavily edited photo (filters) | Passes colorfulness + edge_ratio checks. Classified PROBABLY_REAL → review |
| AI-generated photo | Currently undetectable. Documented as known failure mode. |
| Small thumbnail | Resolution < 300px on shortest side → POOR quality, reject |
| Monochrome real photo | Colorfulness < 5 but normal edge_ratio and entropy > 4.5 → classified REAL |
| Presentation slide with photo | Edge_ratio high in text regions, normal in photo regions → AMBIGUOUS → review |
| Screenshot of photo gallery | High colorfulness + high edge_ratio → NON_REAL (R1 trigger) |

### Known Failure Modes (unchanged)

- **CGI/AI photorealistic imagery**: Neither heuristics nor MobileNetV3-Small can distinguish from real photos. digital_domain requires human review.
- **dhash rotation sensitivity**: Rotated/cropped near-duplicates may be missed.
- **Face detection unreliability**: Used only as soft constraint, never hard reject.
- **Dataset-specific edge cases**: The rubric includes a review pool mechanism to catch unexpected cases.

### Conclusion Paragraph for Final Plan

**The vision-criteria rubric has been empirically recalibrated against all 11 datasets. The single most important finding is that colorfulness alone is a misleading signal — screenshots are more colorful than real photos (kpmg_forensic median=60.0 vs. m_immobilier median=29.2). The actual heuristic separator is edge_ratio combined with colorfulness: synthetic content simultaneously exhibits high edge density (>0.35) and high colorfulness (>70), while real photos rarely show both. All rejection thresholds use dataset-adaptive percentiles (e.g., sharpness floor = max(2.0, dataset_p2)) to handle the 4× variation in sharpness distributions across datasets. The aggregated plan's claim of MobileCLIP2-S0 at ~0.51s/image is not supported by shared evidence — the only benchmarked model is MobileNetV3-Small (1.5M params, ~0.07s/image CPU). I recommend removing the composite Q formula in favor of dataset-normalized z-score sorting with model scores as tiebreakers only. The 3-stage cascade architecture, conservative bias, dhash-based dedup, and feature-hash diversity clustering from the aggregated plan are retained. Ten explicit boundary-case rules with auditable reasoning cover all identified gray zones, and the review pool mechanism ensures that ambiguous cases are never silently included in the final gallery selection.**
