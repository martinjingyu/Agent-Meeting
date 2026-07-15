# Vision Criteria Rubric & Pipeline Plan — Gallery Image Selection

> **Purpose:** Concrete, falsifiable rubric + executable pipeline design for selecting the top 100 real-world photographed scenes per dataset from C:\pics for a corporate website gallery.

---

## 1. Decision Rubric: Three-Axis Framework

Every image is evaluated on **three independent axes**. Each axis produces a label + evidence string. The final decision is a composition rule, not a single score.

### Axis A: Real-Scene Authenticity (real vs. non-real)

| Label | Meaning | Visual Evidence Required | Threshold / Rule |
|-------|---------|--------------------------|------------------|
| **REAL** | Confident real-world photograph | Natural lighting gradients, lens-based depth-of-field (smooth bokeh or gradual falloff), realistic texture noise (sensor grain), irregular geometric composition (not perfectly aligned/centered), natural color distribution (no posterization, no flat palette), environmental context (sky, ground plane, walls meeting at non-orthogonal angles) | Heuristic pass + MobileCLIP photo-probability ≥ 0.65 |
| **PROBABLY_REAL** | Likely real but some ambiguity | Meets most REAL criteria but: could be a high-quality render with realistic lighting, or a heavily edited photo, or a photo with graphic overlays | Heuristic pass + 0.45 ≤ MobileCLIP photo-probability < 0.65 |
| **AMBIGUOUS** | Cannot decide with confidence | Mixed signals: e.g. real background + overlaid text/graphics; realistic textures but impossible geometry; photo-like but AI-generated artifacts present | Either heuristic borderline OR MobileCLIP 0.35–0.45 |
| **PROBABLY_NON_REAL** | Likely not a photograph | Strong evidence of: flat uniform regions (posterization), perfect orthogonal alignment, uniform vector-style edges, text-heavy layout, pure gradient backgrounds, repeating patterns consistent with UI/diagram | Heuristic fail OR 0.20 ≤ MobileCLIP photo-probability < 0.35 |
| **NON_REAL** | Confident non-photograph | Clearly: screenshot of UI/website, chart/graph, illustration/drawing, 3D render with obvious CGI artifacts (perfect reflections, no noise, plastic surfaces), text document, AI-generated with typical artifacts (asymmetric eyes/limbs, garbled text, impossible physics) | Heuristic fail AND MobileCLIP photo-probability < 0.20 |

**Heuristic pre-filter rules (Axis A):**
- **Low-colorfulness rejection:** If colorfulness (Hasler-Susstrunk) < 5.0 → mark NON_REAL (catches infographics, text docs, B&W diagrams). *Caveat: Industrial/architectural monochrome interiors may also score low — these get a second look via MobileCLIP.*
- **Extreme-sharpness rejection:** If sharpness (Sobel mean) > 400 AND colorfulness < 12 → mark NON_REAL (catches text screenshots with sharp edges but no color). *Caveat: High-res product photos with fine detail may also score high — checked against brightness/colorfulness jointly.*
- **Extreme-aspect-ratio rejection:** If aspect-ratio < 0.3 or > 3.5 → mark NON_REAL (catches banner ads, wide UI mockups, narrow vertical phone screenshots). *Caveat: Panoramas may have AR > 3.5 — these are checked by MobileCLIP before final rejection.*

### Axis B: Gallery Display Quality

| Label | Meaning | Visual Evidence Required | Threshold / Rule |
|-------|---------|--------------------------|------------------|
| **EXCELLENT** | Ready for gallery | Sharp (Sobel ≥ 40), good exposure (brightness 60–220), natural color (colorfulness ≥ 8), stable composition, recognizable subject, no obvious artifacts, resolution ≥ 800px on shortest side | All conditions met |
| **GOOD** | Acceptable with minor issues | Slightly soft (Sobel 20–40), mild exposure issue (brightness 40–60 or 220–240), slightly low color (5–8), or resolution 500–800px on shortest side | Meets most but not all EXCELLENT criteria |
| **FAIR** | Marginal — review pool | Soft/blurry (Sobel 10–20), noticeable exposure issue, very low color (3–5), resolution 300–500px, or mild compression artifacts | Heuristic borderline |
| **POOR** | Reject — not suitable | Very blurry (Sobel < 10), extreme exposure (brightness < 30 or > 245), severe compression, resolution < 300px, or corrupted/unreadable | Fails one or more hard thresholds |

**Additional quality signals considered:**
- **Laplacian variance** as secondary sharpness check (if Sobel is ambiguous)
- **Entropy** for information richness (low entropy → uniform/flat → poor display value)
- **Contrast ratio** (local std-dev of luminance): very low contrast → flat/flash-washout
- **Compression artifacts**: detect blockiness via discrete cosine transform approximation or simply use file-size-to-resolution ratio as proxy

### Axis C: Gallery Suitability (display appropriateness)

| Label | Meaning | Criteria |
|-------|---------|----------|
| **GALLERY_READY** | Suitable for public website showcase | REAL or PROBABLY_REAL + EXCELLENT or GOOD quality + subject is appropriate for corporate website (landscape, architecture, interior, people in context, events) |
| **REVIEW_NEEDED** | Borderline — human must decide | REAL/PROBABLY_REAL + FAIR quality; or AMBIGUOUS + EXCELLENT/GOOD quality; or REAL + EXCELLENT but subject is controversial/risky (cannot be automatically determined — flagged for review) |
| **NOT_SUITABLE** | Excluded from gallery | NON_REAL/PROBABLY_NON_REAL; or POOR quality; or duplicate/near-duplicate with a better version already selected |

---

## 2. Boundary Case Handling (Explicit Audit Rules)

These are the known grey zones. Each has an explicit rule rather than a magic number.

| Boundary Case | Detection Method | Handling Rule |
|---------------|-----------------|---------------|
| **High-quality 3D render** (e.g. architectural visualization) | Low sensor-noise proxy (check variance in uniform regions); perfect edge transitions; no chromatic aberration | If MobileCLIP photo-probability < 0.5 AND render-likely signals present → NON_REAL. If probability ≥ 0.5 → PROBABLY_REAL → REVIEW_NEEDED |
| **Heavily edited/tuned photo** (e.g. HDR, heavy Instagram filter) | High saturation, clipped highlights/shadows, unnatural color distribution | Check colorfulness and contrast. If within normal range → REAL. If extreme (colorfulness > 60 or contrast < 0.2) → PROBABLY_REAL → REVIEW_NEEDED |
| **AI-generated photorealistic image** | Asymmetric detail, garbled fine text, unnatural texture repetition, impossible reflections | No reliable automatic detection at this model budget. Flag as AMBIGUOUS if any AI artifact is detected via texture analysis. Otherwise treated as REAL but logged with "AI_check_inconclusive" warning for human review |
| **Photo + text/graphic overlay** (e.g. magazine cover, meme) | Detect high-contrast text regions via MSER or edge-density histogram | If overlay covers > 20% of image area → PROBABLY_REAL → REVIEW_NEEDED. Otherwise → REAL. Always log overlay-flagged candidates separately |
| **Screenshot of real photo** (e.g. photo displayed on monitor) | Detect screen-door pattern via frequency analysis; consistent grid artifacts | → NON_REAL (it's a screenshot, not a photographically captured scene) |
| **Museum/exhibition photo** (photo of a photo/painting) | Frame detection via edge rectangles; high-frequency detail only in center region | → PROBABLY_REAL → REVIEW_NEEDED (the scene itself is real but derivative) |
| **Microscope/medical image** | Extreme sharpness, unusual color profile, scale markers | → NON_REAL for gallery (not a natural photographed scene) |
| **Drone/aerial photo** | High angle, wide coverage, small ground features | → REAL (it is a real photograph). But check if resolution is sufficient for gallery display |
| **Night / low-light photo** | Very low brightness, high noise | If noise is high (detect via local variance analysis) AND brightness < 40 → POOR quality. Otherwise → quality adjusted accordingly |

---

## 3. Diversity & Duplicate Handling

### Near-Duplicate Detection
- **Method:** dhash (8×9 → 64-bit hash), Hamming distance ≤ 4 → near-duplicate cluster
- **Scope:** Full pairwise comparison within each dataset (O(n²) would be too expensive for truro_school's 36k images — use sliding-window with hash-table bucketing: first 8 bits of hash as bucket key, only compare within bucket)
- **Action:** Within each cluster, keep only the highest-quality image (Axis B score). Log others as "duplicate_suppressed"

### Scene Diversity
- **Goal:** Top 100 should cover at least 5–8 distinct scene categories per dataset
- **Method:** After Axis A+B filtering, cluster remaining candidates by:
  1. Color histogram (HSV 3D histogram, 8×4×4 bins → 128-dim vector, cosine distance)
  2. Dominant scene type inferred from average brightness + colorfulness + edge density
- **Diversity enforcement:** Greedy selection — pick top by composite score, then penalize (add distance-weighted penalty for similarity to already-selected images). Stop at 100 or when no good candidates remain.

### Per-Scene-Type Cap
- Single-person portraits: max 10% of final 100 (i.e., ≤ 10)
- Near-identical scenes (same location/angle): max 3
- If insufficient diverse candidates → output actual count, don't pad with low-quality fillers

---

## 4. Pipeline Architecture: Two-Phase Design

### Phase A — Heuristic Pre-filter (ALL images, CPU-only, ~70 min total)
Goal: Reduce 61,958 images to ~top 30% (~18,500 candidates) at low cost.

**Step A1: Load & Validate**
- Iterate dataset directories
- Attempt PIL open for each file
- Log read failures with file path + error type
- Track format, size, resolution

**Step A2: Compute Heuristics**
For each valid image (resized to max 640px on longest side for speed):
- Sharpness (Sobel mean gradient)
- Colorfulness (Hasler-Susstrunk)
- Brightness (mean pixel value)
- Aspect ratio
- Entropy (image information content)
- dhash (64-bit)

**Step A3: Apply Hard Rejection Rules**
```
IF sobel < 5 → POOR (extreme blur)
IF colorfulness < 3 → NON_REAL (likely text/doc/UI)
IF brightness < 20 or > 250 → POOR (extreme exposure)
IF aspect_ratio < 0.25 or > 4.0 → NON_REAL (extreme banner/UI)
IF resolution_min_side < 200 → POOR (too small)
```
These thresholds are intentionally conservative — they catch only the clearest non-candidates.

**Step A4: Soft Scoring**
Compute composite heuristic score:
```
heuristic_score = 0.4 * normalized_sharpness + 0.3 * normalized_colorfulness + 0.2 * normalized_brightness + 0.1 * (1 - normalized_aspect_ratio_extremity)
```
Where each normalized term maps to [0,1] via percentile-based scaling per dataset.

**Step A5: Candidate Selection**
Select top 30% by heuristic_score (or top 5000 if dataset is large) for Phase B.
For very small datasets (kpmg_forensic: 80 images), pass all valid images.

### Phase B — Model-Based Classification (~15–20 hours on CPU for ~18,500 candidates)
Goal: Confirm real-scene authenticity and gallery suitability.

**Step B1: MobileCLIP-S0 Zero-Shot Classification**
For each candidate image:
- Preprocess: resize to 256×256, center crop to 224×224 (MobileCLIP default)
- Compute image-text similarity for 8 photo prompts vs. 8 non-photo prompts
- Derive photo_probability = softmax(mean_photo_sim, mean_non_photo_sim)

**Estimated per-image time:** 2–5 seconds on CPU (Intel i5-12500)
- 18,500 candidates × 3.5s avg ≈ 18 hours
- Can be parallelized: process 4–6 images concurrently via Python multiprocessing (6 CPU cores) → ~3–4 hours wall-clock

**Step B2: Quality Validation on Real Candidates**
For images classified as REAL or PROBABLY_REAL:
- Compute full-resolution sharpness (original size, not thumbnail)
- Check for compression artifacts (blockiness score)
- Check for exposure clipping (percentage of pixels at 0 or 255)
- Compute final quality score

**Step B3: Final Gallery Ranking**
```
gallery_score = 0.4 * photo_probability + 0.3 * quality_score + 0.2 * diversity_bonus + 0.1 * scene_appeal_bonus
```

Where:
- quality_score = composite of sharpness (normalized), exposure, colorfulness
- diversity_bonus = higher for underrepresented scene types in current selection
- scene_appeal_bonus = higher for landscapes, architecture, group activities; lower for single portraits

**Step B4: Diversity-Aware Selection**
Apply greedy selection with similarity penalty (color-histogram distance). Select top 100.

### Fallback: Model-Free Variant (if MobileCLIP is too slow)
If Phase B is deemed too slow (~18h CPU), fallback to heuristic-only pipeline with:
- More aggressive heuristic thresholds
- Entropy-based content type discrimination
- Color-histogram-based scene clustering for diversity
- Manual review pool of ~500 candidates per large dataset

**Trade-off:** Without MobileCLIP, we cannot distinguish high-quality renders from real photos, or detect AI-generated images. The heuristic-only variant would pass more false positives and require more manual review.

---

## 5. Per-Dataset Strategy

| Dataset | Size | Content Profile | Strategy |
|---------|------|----------------|----------|
| **truro_school** | 36,266 | School photos (high real-photo density) | Phase A: pass top 5000 → Phase B: classify → diverse top 100. Expected yield: 100 excellent candidates easily. |
| **m_immobilier** | 5,000 | Real estate photos (all real) | Phase A: quality filter → Phase B: deduplicate (same property multiple angles) → diverse top 100. Key challenge: avoid selecting 20 photos of the same house. |
| **maior_capital** | 5,000 | Real estate + floor plans | Phase A: filter out floor plans (low colorfulness, orthogonal lines) → Phase B on remaining → top 100. |
| **tara_guerard** | 4,971 | Personal blog photos | Phase A: filter graphics/blog-design → Phase B → top 100. May have fewer suitable candidates due to 2000s-era aesthetic. |
| **boston_university** | 2,722 | Campus photos + UI screenshots | Phase A: filter UI (high sharpness + medium colorfulness = text) → Phase B → top 100. |
| **roland_berger** | 2,997 | ~50% infographics | Phase A: colorfulness < 7 → NON_REAL will catch most infographics → Phase B on remaining → top 100. Expected yield: low. |
| **digital_domain** | 1,669 | VFX/CGI + real photos | Phase B is critical here: MobileCLIP must distinguish real from CG. Even then, expect many AMBIGUOUS. |
| **tuv_rheinland** | 1,465 | ~60% diagrams/charts | Phase A will reject most → Phase B on remaining. Expected yield: very low (maybe 20–40). |
| **ul_solutions** | 1,515 | Product photos + graphics | Phase A: filter graphics → Phase B → top 100. Must check for product-only catalogs. |
| **kpmg_forensic** | 80 | ~90% screenshots | Phase A: reject all. Expected yield: 0–5. Document and move on. |
| **thema-med** | 273 | Mixed slides + photos | Phase A → Phase B. Expected yield: 10–30. |

---

## 6. Output Format

### Directory Structure
```
workspace/
├── run_pipeline.py                  # Main entry point
├── config.yaml                      # Per-dataset thresholds (adjustable)
├── phase_a_heuristics.py            # Phase A implementation
├── phase_b_classifier.py            # Phase B MobileCLIP implementation
├── diversity_utils.py               # Duplicate detection, diversity selection
├── quality_scoring.py               # Quality metrics (sharpness, exposure, etc.)
├── results/
│   ├── {dataset_name}/
│   │   ├── top100_gallery.csv       # Final gallery picks
│   │   ├── rejected_all.csv         # All rejected with reasons
│   │   ├── review_pool.csv          # Borderline cases for human review
│   │   └── dataset_stats.json       # Per-dataset statistics
│   ├── all_datasets_stats.json      # Aggregated statistics
│   └── sampling_report.json         # Quality control sampling results
└── logs/
    ├── pipeline_run_{timestamp}.log
    └── read_errors.log              # Corrupted/unsupported files
```

### Result CSV Schema (top100_gallery.csv)
```csv
relative_path,dataset,rank,gallery_score,photo_probability,quality_label,quality_score,
real_labels,aspect_ratio,width,height,sharpness,colorfulness,brightness,
rejection_reason,diversity_cluster_id,scene_type,is_duplicate_of,notes
```

### Rejection Reasons (controlled vocabulary)
- "non_real_photograph" — Axis A: NON_REAL or PROBABLY_NON_REAL
- "low_quality_blurry" — Axis B: sharpness below threshold  
- "low_quality_exposure" — Axis B: extreme brightness
- "low_quality_too_small" — Axis B: resolution < 300px
- "low_quality_compression" — Axis B: severe artifacts
- "duplicate_suppressed" — Better version already selected
- "diversity_limited" — Similar scene already in top 100
- "single_portrait_cap" — Portrait quota exceeded
- "gallery_not_suitable" — Axis C: NOT_SUITABLE
- "review_pool" — Borderline, needs human check
- "read_error" — Could not open/decode file

---

## 7. Known Failure Modes (Not Resolvable by This Rubric)

1. **AI-generated photorealistic images** — No reliable detection within <100M parameter budget. Must be flagged for human review.
2. **High-end 3D architectural renders** — Can look indistinguishable from real photos. MobileCLIP gives a probability, not certainty.
3. **Heavily filtered/processed photos** — May be misclassified as NON_REAL due to extreme colorfulness or sharpness.
4. **Genre ambiguity** — Museum photos (photo of a painting) are real photographs of non-real scenes. The rubric classifies these as PROBABLY_REAL → REVIEW_NEEDED.
5. **Copyright/brand/portrait rights** — This rubric explicitly does not address legal compliance. Output is an engineering pre-filter only.
6. **Cultural/context sensitivity** — A photo may be technically excellent but inappropriate for a corporate gallery (e.g., protest scenes, revealing clothing). This cannot be automatically detected and must be manually reviewed.
7. **Scene diversity across a single dataset** — If a dataset only contains one scene type (e.g., m_immobilier: only houses), the diversity cap may limit to well under 100. This is correct behavior — do not pad.

---

## 8. Running the Pipeline

```powershell
# From workspace directory:
cd C:\Users\LX034\Code\Agent-Meeting\roles\vision-criteria-planner\workspace

# Install any missing deps (should already be installed per Stage 1):
# pip install scipy scikit-image  (for sobel and additional image metrics)

# Phase A only (heuristic pre-filter, ~70 min):
python run_pipeline.py --phase A

# Full pipeline (Phase A + Phase B, ~4-20 hours depending on parallelization):
python run_pipeline.py --phase all --parallel 4

# Per-dataset mode (for testing/debugging):
python run_pipeline.py --datasets m_immobilier,kpmg_forensic --phase all

# Review results:
# See worksapce/results/{dataset_name}/ for CSVs
# See worksapce/results/sampling_report.json for quality check
```

---

## 9. How to Reuse on New Datasets

1. Add new dataset folder under C:\pics
2. Update config.yaml with dataset name (or let it auto-detect)
3. Run pipeline — no code changes needed
4. Review results CSVs
5. If thresholds seem off for new dataset, adjust per-dataset overrides in config.yaml:
   ```yaml
   my_new_dataset:
     sharpness_min: 15  # default is 5
     colorfulness_min: 4  # default is 3
     mobileclip_threshold: 0.55  # default is 0.5
   ```

---

## 10. Conclusion Paragraph (to include in final deliverable)

> This rubric and pipeline design provides an auditable, falsifiable framework for selecting gallery-ready real-world photographs from diverse, mixed-content image datasets entirely on CPU. The three-axis evaluation (real-scene authenticity → display quality → gallery suitability) with explicit boundary-case rules eliminates opaque scoring while maintaining conservative bias. The two-phase architecture — fast heuristic pre-filter (Phase A, ~70 min for all 61,958 images) followed by MobileCLIP-S0 model-based verification on candidates only (Phase B, ~15-20h CPU or ~3-4h with multiprocessing) — stays within the 100M parameter budget and CUDA-free constraint. For datasets where even Phase B is too slow, a model-free fallback with expanded manual review pool is provided. Key design decisions include: (1) dhash-based near-duplicate suppression within sliding-window hash buckets; (2) greedy diversity-aware selection using HSV histogram distance; (3) per-dataset configurable thresholds rather than global hardcodes; (4) explicit logging of all known failure modes the rubric cannot resolve. The output is a structured CSV trail per dataset (gallery picks, rejections with reasons, review pool, statistics) enabling full traceability for human review. Expected gallery yields range from 0–5 (kpmg_forensic) to 100 (truro_school, m_immobilier), with honest reporting when insufficient suitable candidates exist.
