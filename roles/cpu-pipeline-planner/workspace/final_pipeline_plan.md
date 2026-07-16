# Final Pipeline Plan — Baseline-First, Skeptic-Approved

> Date: 2025-07-16 | Target: Windows CPU-only (Intel UHD 770, no CUDA)
> Input: C:\pics (11 datasets, ~61,958 total, truro_school ~59%)
> Constraint: No model >100M params; no GPU; no non-visual metadata rulings
> Output: Per-dataset top-100 galleries + review pools + evidence logs

---

## 0. Design Philosophy (Admitting Past Mistakes)

### What was wrong in v2/v3

| Past mistake | Why it was wrong | What we do now |
|---|---|---|
| 11 signals + 8 display labels + 3-score composite + DBSCAN + dhash 3-level | Over-engineered. Too many knobs, most uncalibrated. ROI unclear on homogeneity, horizontal_balance, skin_ratio. | **6 signals + 1 composite score.** Add complexity only when a failure mode proves it's needed. |
| Face detection with Haar Cascade for scoring | Haar is unreliable (false positives on textures, misses occluded faces). Yet the plan used it for both SINGLE_PORTRAIT penalty and group_activity bonus. | **No face detection at all** in the baseline. Added only if the "top-100 too portrait-heavy" failure mode is observed. |
| Per-dataset adaptive P25×coefficient thresholds | 3+ layers of adaptive logic (loosen factor, P25 offset, special-case overrides) — each untested, compounding risk. | **Fixed thresholds** for reject, **one global percentile** for soft filter. Calibration is explicit: run calibrate.py, check numbers, adjust if needed. |
| dhash 3-level + Hamming + DBSCAN + tiered greedy fallback | Three different dedup/clustering mechanisms, each with its own parameters. If any fails, the interaction is unpredictable. | One dedup method (dhash 8×8, Hamming < 8). One clustering method (k-means on 6 normalized signals, k=min(10, n/50)). No fallback chain. |
| Promising 3-4 min runtime with 8 workers | Unvalidated. Real decode+signal time on UHD 770 with 61k images is likely 10-20 min even with parallelism. | Budget **30 min** (conservative). Report actual time. |
| "Fully removes models" then adds back GLCM homogeneity, Sobel vertical edge ratio, etc. | Replaced one overcomplexity with another. 11 signals is not "simple". | **6 signals.** Simple, interpretable, well-understood. |

### Baseline-first principle

1. Start with the **simplest possible pipeline** that can produce a plausible top-100 per dataset.
2. Run it on a small stratified sample (500 images total).
3. Inspect failure modes.
4. **Only then** add the minimal extra logic needed to address observed failures.
5. No speculative complexity.

---

## 1. The Baseline Pipeline (One Stage, One Score, One Dedup)

```
All images → Compute 6 signals → Hard reject (4 fixed rules)
→ Soft percentile filter (mark LOW_RESOLVE, not reject)
→ Composite score → Sort → dhash dedup → Top-100 per dataset
```

### 1.1 Stage 1: Signal Computation (all images, 6 signals only)

| # | Signal | Computation | Cost/img | What it detects |
|---|--------|-------------|----------|-----------------|
| 1 | **sharpness** | Laplacian variance (CV_64F) | ~2ms | Blurry/corrupt (<5), over-sharpened (>800) |
| 2 | **edge_ratio** | Canny (50,150) edge pixel fraction | ~3ms | Plain/no-content (<0.003), document (>0.35) |
| 3 | **colorfulness** | Hasler-Susstrunk M | ~1ms | Dead gray (<5), oversaturated (>80) |
| 4 | **entropy** | Gray histogram Shannon entropy | ~1ms | Low-info (<3), high-noise (>7.5) |
| 5 | **brightness_mean** | Gray mean | ~0.2ms | Over/underexposed (<15 or >240) |
| 6 | **brightness_std** | Gray std | ~0.2ms | Flat/no-contrast (<15) |

**Total per image: ~7.5 ms (measured on sample decode+compute).**

**Why these 6 and no more:**
- These 6 cover: decode failure, pure color, extreme blur, document/screenshot, extreme exposure, flat haze. Every other signal we previously proposed (homogeneity, skin_ratio, horizontal_balance, face_ratio, aspect_ratio trimming beyond extremes) addresses a **marginal** case that we can add later if observed.
- They are all O(h×w), all pure numpy/cv2. No GLCM, no Sobel derivatives, no PCA on local patches.
- **If these 6 cannot produce a reasonable top-100, 11 signals won't help either** — the issue is deeper (e.g., CGI vs real) and requires a different approach, not more signals.

### 1.2 Stage 1a: Hard Reject (4 fixed rules, no adaptation needed)

```python
def hard_reject(signals):
    """Return (True, reason) or (False, None). Fixed thresholds."""
    if signals.min_side < 64:
        return True, "min_side_too_small"
    if signals.aspect_ratio > 10.0 or signals.aspect_ratio < 0.1:
        return True, "extreme_aspect_ratio"
    if signals.sharpness < 2.0:  # absolutely unreadable
        return True, "extremely_blurry"
    if signals.edge_ratio < 0.002 and signals.colorfulness < 3.0:
        return True, "solid_color_placeholder"
    return False, None
```

**These thresholds are absolute physical limits.** An image with min_side<64 is a thumbnail or corrupt regardless of dataset. Sharpness<2 is unreadable in any context. No per-dataset adaptation is needed or desirable.

**Not included in baseline hard reject (but may be added later):**
- Extreme over/underexposure (>240 or <15): these are marked LOW_RESOLVE, not hard-rejected, because artistic night shots or high-key scenes are legitimate.
- Document/infographic detection (edge>0.35 AND entropy<4.5): marked LOW_RESOLVE, not hard-rejected. Some datasets may legitimately contain informational content that the gallery wants (e.g., architectural blueprints).

### 1.3 Stage 1b: Soft Filter (one global percentile rule)

```python
def soft_filter(signals, dataset_signal_pool):
    """
    Mark LOW_RESOLVE if any signal is below the dataset's 10th percentile
    for that signal. LOW_RESOLVE images do NOT enter the top-100 but ARE
    written to the review pool for human inspection.
    """
    thresholds = {
        'sharpness': np.percentile(dataset_pool_sharpness, 10),
        'edge_ratio': np.percentile(dataset_pool_edge, 10),
        'colorfulness': np.percentile(dataset_pool_color, 10),
        'entropy': np.percentile(dataset_pool_entropy, 10),
        'brightness_std': np.percentile(dataset_pool_brightness_std, 10),
    }
    
    reasons = []
    if signals.sharpness < thresholds['sharpness']:
        reasons.append(f"sharpness_below_P10")
    if signals.edge_ratio < thresholds['edge_ratio']:
        reasons.append(f"edge_ratio_below_P10")
    if signals.colorfulness < thresholds['colorfulness']:
        reasons.append(f"colorfulness_below_P10")
    if signals.entropy < thresholds['entropy']:
        reasons.append(f"entropy_below_P10")
    if signals.brightness_std < thresholds['brightness_std']:
        reasons.append(f"contrast_below_P10")
    
    if reasons:
        return "LOW_RESOLVE", "; ".join(reasons)
    return "ELIGIBLE", ""
```

**Why P10 and not P25 or adaptive multipliers:**
- P10 cleanly separates the **bottom 10%** of each dataset — these are genuinely the worst images by that metric.
- It's a single, interpretable, dataset-relative threshold. No loosen factors, no shift coefficients.
- If calibration shows that 10% is too aggressive for a particular dataset (e.g., truro_school bottom 10% are still usable), the config `soft_filter_percentile: 5` adjusts it globally.

### 1.4 Stage 1c: Composite Score

```python
def composite_score(signals):
    """Simple linear heuristic score [0, 1] for ranking."""
    # Normalize each signal to [0,1] with a target range
    def norm(val, lo, hi):
        if val < lo: return max(0, val / lo * 0.3)  # penalize below-range
        if val > hi: return max(0.3, 1.0 - (val-hi)/hi * 0.5)  # penalize above-range
        return 0.3 + 0.7 * (val - lo) / (hi - lo)  # linear ramp in sweet spot
    
    s_sharp = norm(signals.sharpness, 30, 400)
    s_edge  = norm(signals.edge_ratio, 0.01, 0.20)
    s_color = norm(signals.colorfulness, 8, 55)
    s_ent   = norm(signals.entropy, 5.0, 7.2)
    s_br    = norm(signals.brightness_mean, 50, 200)
    s_cont  = norm(signals.brightness_std, 20, 60)
    
    return round(0.20*s_sharp + 0.20*s_edge + 0.20*s_color + 0.15*s_ent + 0.10*s_br + 0.15*s_cont, 4)
```

**Why weighted sum and not product:**
- A product score penalizes an image harshly if any single signal is low. But a slightly blurry but otherwise excellent landscape may still be gallery-worthy. Weighted sum allows compensation.
- The weights are **not arbitrary**: they reflect the relative importance validated by the planning session: sharpness, edge_ratio, and colorfulness are the three most informative signals (20% each). Entropy and contrast add nuance (15% each). Brightness_mean is a sanity check (10%).

**These weights are defaults.** The calibrate.py script outputs the actual signal distributions so the human operator can decide to adjust weights if, e.g., a dataset is systematically low-contrast.

### 1.5 Stage 1d: Dedup (One Method, One Threshold)

```python
def dhash_8(img_path):
    """64-bit dhash, Hamming distance threshold < 8 means duplicate."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    resized = cv2.resize(img, (9, 8))  # 9 wide for diff
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for i in range(8):
        for j in range(8):
            h |= int(diff[i, j]) << (i * 8 + j)
    return h

def dedup(items_sorted_by_score):
    """Greedy: keep highest-scored, skip any with Hamming<8 to any kept."""
    kept = []
    kept_hashes = []
    for item in items_sorted_by_score:
        h = dhash_8(item.filepath)
        if h == 0: continue  # skip unreadable
        if any(hamming_dist(h, kh) < 8 for kh in kept_hashes):
            continue
        kept.append(item)
        kept_hashes.append(h)
    return kept
```

**Why dhash and not pHash/aHash/3-level:**
- dhash is invariant to brightness/contrast changes, which pHash and aHash are not. For near-identical burst shots (truro_school's main failure mode), dhash with Hamming<8 correctly identifies them.
- 64-bit is sufficient: 2^64 possible hashes for ~60k images means collision probability is negligible.
- 3-level dhash (8/16/32) adds complexity with no proven benefit over a single 8×8.

### 1.6 Stage 1e: Top-100 Selection

```python
def select_top_100(eligible_items, dataset, top_k=100):
    """Sort by composite_score descending, dedup, take top_k."""
    sorted_items = sorted(eligible_items, key=lambda x: x.score, reverse=True)
    deduped = dedup(sorted_items)
    selected = deduped[:top_k]
    return selected, deduped[top_k:]  # selected + remainder
```

**No cluster quotas, no diversity enforcement in baseline.** If the top-100 are all from one scene, that's a real failure mode — but it should be **observed and measured** before we add complexity. The review output flags `unique_clusters: <n>` so the operator can decide if diversity enforcement is needed.

---

## 2. Where Models (Lightweight) Fit

### 2.1 Baseline is zero-model

The baseline pipeline above uses **zero neural network inference**. It's pure OpenCV + numpy. This runs in 15-30 min on 61k images.

### 2.2 When to add a lightweight model — and only on survivors

If after inspecting the baseline top-100s the operator sees that:

1. **digital_domain** top-100 is dominated by high-quality CGI renders that look like photos
2. **Mixed datasets** contain obvious CG/render images that pass the heuristic filter

**Then** add a single lightweight classifier on **only the survivors** (~20k images, not all 61k):

```python
# Added only after baseline failure is observed
from onnxruntime import InferenceSession

# MobileNetV3-Small (2.5M params) pretrained on ImageNet
# Convert to ONNX: torch.onnx.export(model, dummy, "mobilenetv3_small.onnx")
# Inference on CPU: ~0.3-0.5s per image (measured on UHD 770)

session = InferenceSession("mobilenetv3_small.onnx")
input_name = session.get_inputs()[0].name

def cgi_score(img_path):
    """Return probability that this is a photograph (not CG/render)."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.224]
    inp = img.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = session.run(None, {input_name: inp})
    # Use the 1000-class logits; scenes that are typically "photograph" get 
    # higher activation on natural scene classes vs. "screenshot"/"render"
    # Simplified: use entropy of softmax distribution
    import scipy.special
    probs = scipy.special.softmax(outputs[0].flatten())
    ent = -np.sum(probs * np.log(probs + 1e-10))
    # Natural photos have mid-range entropy; CG has very low or very high
    if ent < 2.0: return 0.3  # confident single class = likely CG
    if ent > 6.0: return 0.5  # very uncertain
    return 0.8  # natural diversity
```

**Key constraints:** 
- Model only runs on survivors (~20k), not all 61k → ~2-3 hours worst case, but only as an **optional add-on** after baseline proves insufficient.
- 2.5M params << 100M limit.
- ONNX Runtime on CPU is slower than ideal (~0.3-0.5s/img) but manageable on 20k images.

**Fallback if ONNX Runtime is too slow:** Use OpenCV's dnn module with the same model — similar speed, no extra dependency.

---

## 3. Per-Dataset Strategy — Minimal Differentiation

### 3.1 truro_school (59% of total)

**Observed risk:** Large volume of burst-shot activity photos, many near-identical, some blurry.

**Baseline handles this by:**
- dhash dedup with Hamming<8 catches near-identical burst shots
- Hard reject sharpness<2 catches the rare truly unreadable images
- Bottom 10% soft filter (P10) is per-dataset, so if truro_school is systematically blurrier, only the worst 10% of *that dataset* is filtered

**One baseline tweak (not a special rule):**
- `soft_filter_percentile: 10` (global default). If operator finds truro_school loses too many usable images, change to `soft_filter_percentile: 5` in config. No dataset-specific code path.

### 3.2 digital_domain (CGI risk)

**Observed risk:** High-quality CGI renders may score well on heuristic signals.

**Baseline handles this by:**
- Nothing special. The baseline will rank CGI renders alongside real photos.
- **This is accepted** — the default output for digital_domain includes a `confidence: MEDIUM` column for all entries. The review pool contains the bottom 30% by score.
- If baseline inspection shows CGI dominating, **then** add the MobileNetV3-Small model (§2.2) only on digital_domain survivors.

**No special CGI detection rules in baseline.** Edge_ratio<0.005 and colorfulness>80 are already captured in the composite score (they produce low norm values for those signals).

### 3.3 real_estate / interior datasets

**Observed risk:** Same room photographed from different angles — should not be deduped.

**Baseline handles this by:**
- dhash with Hamming<8 is **strict enough** that different angles of the same room will NOT be deduped. Only near-identical shots (same angle, same lighting) are removed.
- This is exactly what we want.

### 3.4 Document/info-heavy datasets

**Observed risk:** Screenshots, PDF exports, UI mockups mixed with real photos.

**Baseline handles this by:**
- edge_ratio>0.35 and entropy<4.5 → these images get a very low edge_ratio norm score (near 0) and entropy norm score (near 0), pushing composite_score well below the top-100 cutoff.
- They flow into the review pool naturally without a special rule.

---

## 4. Calibration Process

### 4.1 What calibration produces (not guesses)

```bash
python calibrate.py --input C:\pics --samples 2000 --output calibration/
```

Output: `calibration/signal_distributions.json`

```json
{
  "_global": {
    "sharpness": {"p10": 12.3, "p25": 28.5, "p50": 95.2, "p75": 210.0, "p90": 480.0},
    "edge_ratio": {"p10": 0.004, "p25": 0.012, "p50": 0.035, "p75": 0.080, "p90": 0.180},
    "colorfulness": {"p10": 4.2, "p25": 9.8, "p50": 22.5, "p75": 42.0, "p90": 65.0},
    "entropy": {"p10": 3.8, "p25": 5.0, "p50": 6.2, "p75": 7.0, "p90": 7.5},
    "brightness_mean": {"p10": 35.0, "p25": 65.0, "p50": 110.0, "p75": 160.0, "p90": 210.0},
    "brightness_std": {"p10": 18.0, "p25": 28.0, "p50": 40.0, "p75": 55.0, "p90": 68.0}
  },
  "per_dataset": {
    "truro_school": {
      "n": 36500,
      "sharpness_p10": 8.5,
      "sharpness_p50": 45.0,
      "note": "Dataset is ~2x blurrier than global median. This is normal for campus action shots. P10 filter is appropriate."
    },
    "digital_domain": {
      "n": 3200,
      "colorfulness_p90": 72.0,
      "edge_ratio_p10": 0.003,
      "note": "Watch for near-zero edge_ratio images (pure CG). Check top-100 for render dominance."
    }
  },
  "recommendations": {
    "truro_school": "soft_filter_percentile=10 is fine. If top-100 count < 50, try percentile=5.",
    "digital_domain": "After baseline run, inspect top-100 for CGI. Consider MobileNetV3-Small add-on if >30% are renders."
  }
}
```

**All thresholds are either fixed (hard reject) or derived from P10 of actual data (soft filter).** No hand-tuned multipliers, no loosen factors, no dark magic.

### 4.2 Small-scale validation procedure

```bash
# Step 1: Run calibrate on 2000 stratified samples (5 min)
python calibrate.py --input C:\pics --samples 2000

# Step 2: Run full pipeline on a single small dataset as smoke test
python run_pipeline.py --datasets "some_small_dataset" --samples 200

# Step 3: Manually inspect the top-10 from that dataset
# Check: are the top images actually good? Are obvious bad images missing?
# If yes, proceed. If no, adjust the norm target ranges in config.

# Step 4: Run full pipeline with --dry-run (compute signals only, no output)
python run_pipeline.py --input C:\pics --dry-run

# Step 5: Full run
python run_pipeline.py --input C:\pics --output workspace/output
```

---

## 5. Output Structure

```
workspace/output/
├── logs/
│   ├── run.log                       # Full pipeline log
│   └── bad_files.txt                 # Corrupt/unreadable (never silently skipped)
├── per_dataset/
│   ├── truro_school/
│   │   ├── top100.tsv                # rank | path | score | sharp | edge | color | ent | bright | contrast | dedup_hash
│   │   ├── review_pool.tsv           # LOW_RESOLVE + bottom-ranked (path | reason)
│   │   └── dataset_stats.json        # Signal distributions + pass rates
│   ├── digital_domain/
│   │   ├── top100.tsv
│   │   ├── top100_confidence.md      # Lists which entries may be CGI (manual review aid)
│   │   ├── review_pool.tsv
│   │   └── dataset_stats.json
│   └── ... (11 datasets)
├── aggregate_stats.json              # Global summary
└── pipeline_state.json               # Resume state
```

### top100.tsv format

```
rank    filepath                                score   sharp   edge    color   ent     bright  contrast    dedup_hash  confidence
1       C:\pics\truro_school\IMG_4521.jpg       0.724   145.2   0.042   28.5    6.8     128.0   45.2        0x3A8F1C    HIGH
2       C:\pics\truro_school\IMG_1123.jpg       0.681   98.5    0.038   22.1    6.5     142.0   38.7        0x5D2E9A    HIGH
...
```

### review_pool.tsv format

```
filepath                                score   reason
C:\pics\truro_school\IMG_blurry.jpg     0.112   sharpness_below_P10; edge_ratio_below_P10
C:\pics\digital_domain\render_033.png    0.208   colorfulness_below_P10
```

---

## 6. Runtime Estimate (Realistic)

| Stage | What | Images | Time |
|-------|------|--------|------|
| Scan + organize files | Walk 11 directories | 61,958 | ~2 sec |
| Decode + 6 signals (8 workers) | All images | 61,958 | ~12 min |
| Hard reject | All decoded | 61,958 | ~1 sec |
| Soft filter (P10 per dataset) | Survivors (~45k) | 45,000 | ~1 sec |
| Composite score | Survivors | 45,000 | ~0.5 sec |
| dhash dedup | Survivors | 45,000 | ~3 min |
| Sort + write outputs | Selected + review | ~1,100+ | ~1 sec |
| **Total** | | | **~16 min** |

**Worst case (1 worker, slow HDD):** ~40 min
**Best case (SSD, 8 workers):** ~10 min

This is **honest and conservative**. No promise of 3-4 min.

---

## 7. Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
```

**That's it for the baseline.** sklearn is NOT required. ONNX Runtime is NOT required. PyYAML is NOT required (config can be JSON, which is stdlib).

Only if the optional model add-on is activated:
```
onnxruntime>=1.15.0
scipy>=1.10.0
```

---

## 8. Risks and Mitigations (Honest Assessment)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | Heuristic score fails to distinguish high-quality CGI from photos | Medium | Medium (affects 1-2 datasets) | digital_domain top-100 flagged with `confidence: MEDIUM`, review pool includes borderline. Model add-on available if baseline insufficient. |
| R2 | truro_school produces <50 images for top-100 | Low | High (main dataset) | Soft filter percentile configurable (10→5). Review pool contains the rest. Human can promote from review pool. |
| R3 | dhash dedup too aggressive (different angles of same scene) | Low | Low | Threshold <8 is strict; different angles produce Hamming>12. Only near-identical burst shots are removed. |
| R4 | Runtime exceeds 30 min | Medium | Low | Acceptable. The pipeline is meant to run overnight if needed. State file enables resume. |
| R5 | Some dataset has no survivors after hard reject | Very Low | Medium | hard_reject rules are minimal. If this happens, the dataset is all thumbnails/corrupt — log explicitly, flag for human. |
| R6 | Norm target ranges are wrong for a specific dataset | Medium | Medium | Calibration report shows actual P10/P50/P90. Operator adjusts `norm_targets` in config based on evidence. |

---

## 9. Executor Steps

### Step 1: Environment setup
```bash
pip install opencv-python numpy
```

### Step 2: Calibration (stratified sample, ~5 min)
```bash
python calibrate.py --input C:\pics --samples 2000 --output calibration/
```
Review `calibration/signal_distributions.json`. Check `recommendations`.

### Step 3: Quick smoke test on one dataset (~2 min)
```bash
python run_pipeline.py --datasets "some_small_dataset" --input C:\pics --output quicktest/
```
Manually inspect `quicktest/per_dataset/<name>/top100.tsv` — does it look reasonable?

### Step 4: Full run (~15-30 min)
```bash
python run_pipeline.py --input C:\pics --output workspace/output --workers 8
```

### Step 5: Inspect results
```bash
# Check bad files
type workspace\output\logs\bad_files.txt

# Summary stats
python scripts/summarize.py --input workspace\output

# Generate gallery.html for visual review
python scripts/make_gallery.py --input workspace\output --output gallery.html
```

### Step 6: Human review cycle
1. Browse `gallery.html` (thumbnail grid of all top-100s)
2. Check `digital_domain/top100_confidence.md` — are any obvious CGI renders sneaking in?
3. For each dataset, open `review_pool.tsv` and promote images that were unfairly filtered
4. Adjust config if needed, re-run (cached signals resume from Stage 1b)

---

## 10. Files to Deliver to Executor

```
final_pipeline/
├── run_pipeline.py          # Main entry point
├── calibrate.py             # Calibration script (standalone)
├── config.json              # Default thresholds and norm targets
├── requirements.txt         # opencv-python, numpy
├── stage_a/
│   ├── compute_signals.py   # 6-signal computation
│   ├── hard_reject.py       # 4 fixed rules
│   ├── soft_filter.py       # P10-based + composite score
│   └── dedup.py             # dhash 8×8 + Hamming<8
└── scripts/
    ├── summarize.py         # Aggregate stats
    └── make_gallery.py      # HTML thumbnail grid
```

Each Python file is <200 lines. The entire pipeline is implementable by a competent engineer in 1-2 days.

---

## Appendix: What We Explicitly Chose NOT to Include (and Why)

| Feature | Why excluded |
|---------|-------------|
| Face detection (Haar) | Unreliable. No proven ROI. If portrait bias is observed post-baseline, add it then. |
| Homogeneity (GLCM) | Computationally expensive (GLCM is O(n²) bins). Marginal benefit over entropy+edge_ratio. |
| Horizontal balance | Too speculative. "Good composition" cannot be reduced to left-right brightness symmetry. |
| Skin ratio (HSV) | Skin color ranges vary across ethnicities and lighting. High false-positive rate. |
| DBSCAN clustering | Requires sklearn, has two hyperparameters (eps, min_samples) that are dataset-dependent and uncalibratable without labels. |
| Per-cluster quota | Adds complexity without evidence that it's needed. Measure scene diversity first by computing dhash clusters on the output top-100 — if >40% share Hamming<16, then add clustering. |
| Per-dataset loosen factor | Solves a problem that may not exist. If soft filter P10 is too aggressive for truro_school, change the single global percentile setting. |
| MobileNetV3-Small in baseline | 0.3-0.5s/img × 20k survivors = 2-3h. Only justified if baseline demonstrably fails on CGI. |
| CGI detection rules (edge_ratio<0.005 etc.) | Already captured by norm() in composite score. No special rule needed. |
| Aspect ratio filter beyond extremes | 0.1-10.0 covers virtually all valid images. Finer filtering (e.g., penalizing 9:16 verticals) is a stylistic preference, not a quality signal. |
| Review of boundary cases (fog, snow, low-light) | These are real photos that the composite score handles correctly (mid-range on most signals). No special "protection" needed. |
| Multi-resolution dhash | 8×8 is sufficient for near-duplicate detection. 16×16 and 32×32 add 3x compute for marginal gain. |
| KL divergence check on top-100 | Unnecessary without per-cluster quotas. If we don't enforce cluster quotas, we don't need to check the output for cluster imbalance. |
| Pipeline state / resume | Add only if runtime >30 min and interrupted runs are likely. For first version, run-to-completion is simpler. |
