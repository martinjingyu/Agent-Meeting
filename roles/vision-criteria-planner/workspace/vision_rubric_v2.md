# Vision Criteria Rubric — Revised Based on Empirical Validation

## Overview

This document defines the concrete, falsifiable rubric for:
1. **Real-scene vs. non-real-scene** classification
2. **Gallery-worthy display quality** assessment
3. **Boundary case handling** with auditable reasoning

It supersedes Round 1's rubric based on empirical validation results.

---

## Axis 1: Real-Scene Trustworthiness (REAL / PROBABLY_REAL / AMBIGUOUS / PROBABLY_NON_REAL / NON_REAL)

### Core Principle
Classify based on visual content evidence only — never filenames, EXIF, or source paths.

### Decision Criteria (ordered from cheapest to most expensive)

#### Criterion 1.1: Heuristic Pre-Screen (applied to ALL images, ~0.05-0.09s per image)

| Signal | What it measures | Real-photo typical range | Non-real typical range | Notes |
|--------|-----------------|------------------------|----------------------|-------|
| **Colorfulness** (Hasler-Susstrunk) | Perceptual color saturation/variety | 8-80 (real photos span wide range) | 5-140 (screenshots can be very colorful) | **NOT a separator alone** — screenshots often have HIGHER colorfulness than real photos |
| **Edge ratio** (Canny edge pixels / total pixels) | Proportion of sharp edges in image | 0.05-0.25 (natural edges) | 0.15-0.50 (text/UI edges) | Higher edge ratio + high colorfulness = strong screen/UI signal |
| **Image entropy** | Information density / complexity | 6.0-7.8 (natural scenes) | 3.0-7.5 (can be low for solid backgrounds or high for text-heavy) | Low entropy (< 4.0) + high brightness = probable document/blank |
| **Aspect ratio** | Width/height | 0.5-3.0 (most cameras) | < 0.3 or > 3.5 (extreme banners) | Extreme aspect ratios are strong non-real signal |
| **Brightness std** | Luminance variation | 40-80 (natural lighting variation) | 10-40 (flat UI/document lighting) or 60-120 (contrasty graphics) | Use intra-image std of pixel brightness |

#### Criterion 1.2: Combined Heuristic Rules (hard reject)

Apply these sequentially. Any single rule match → NON_REAL.

| Rule ID | Rule | Visual Evidence | Rationale |
|---------|------|----------------|-----------|
| R1 | `edge_ratio > 0.35 AND colorfulness > 70` | High edge density + highly saturated = synthetic graphic | Screenshots with colorful UI elements |
| R2 | `aspect_ratio < 0.25 OR aspect_ratio > 4.0` | Extremely wide/narrow = banner/ad | No camera produces 4:1+ or 1:4- naturally |
| R3 | `image_entropy < 3.5 AND brightness_std < 15` | Near-uniform color = blank/solid | Corrupted, placeholder, or solid-color image |
| R4 | `sharpness_640 < 2.0` | Extremely blurry | Too blurry for any meaningful classification |
| R5 | `colorfulness < 4 AND edge_ratio > 0.20` | Low color + high edges = document/infographic | Text on white background |

#### Criterion 1.3: Soft Scoring (for candidates passing R1-R5)

Composite score = weighted combination:
- `colorfulness_score`: normalized to [0,1] where colorfulness=20→0, 60→1, then clipped
- `edge_ratio_penalty`: 1 - min(edge_ratio / 0.3, 1.0) — penalizes high edge ratios
- `entropy_score`: normalized where entropy=7.0→1.0, entropy=4.0→0.0
- `brightness_std_score`: normalized where std=50→1.0, std=10→0.0

**Composite = 0.25 × colorfulness_score + 0.25 × entropy_score + 0.25 × brightness_std_score + 0.25 × edge_ratio_penalty**

- ≥ 0.55 → REAL
- 0.40–0.55 → PROBABLY_REAL (pass to review pool)
- 0.25–0.40 → AMBIGUOUS (pass to review pool)
- < 0.25 → PROBABLY_NON_REAL (reject unless MobileNetV3 confirms)

#### Criterion 1.4: MobileNetV3-Small Zero-Shot (optional, for AMBIGUOUS pool only, ~0.07s per image)

Use MobileNetV3-Small features + a simple linear probe (or nearest-centroid classifier) trained on:
- Positive class: images known to be real photos (from m_immobilier, truro_school samples)
- Negative class: images known to be non-real (from roland_berger infographics, kpmg_forensic screenshots)

**When to invoke:**
- Only for images classified as AMBIGUOUS or PROBABLY_NON_REAL by heuristics
- Not needed for datasets already dominated by real photos (m_immobilier, truro_school, tara_guerard)
- Most useful for: digital_domain (CGI vs real), roland_berger/thema-med/tuv_rheinland (infographics), boston_university (photos vs UI)

---

## Axis 2: Gallery-Worthy Display Quality (EXCELLENT / GOOD / FAIR / POOR)

### Decision Criteria

#### Criterion 2.1: Sharpness (Sobel gradient mean at 640px scale)

Based on empirical validation across 11 datasets:

| Rating | Sharpness Range | Visual Evidence | Notes |
|--------|----------------|-----------------|-------|
| EXCELLENT | > 30.0 | Crisp details, no visible blur | Professional photography |
| GOOD | 15.0 – 30.0 | Slightly soft but acceptably sharp | Typical consumer camera |
| FAIR | 8.0 – 15.0 | Noticeably soft but subject discernible | Usable if subject is interesting |
| POOR | < 8.0 | Blurry or motion-blurred | Reject for gallery |

**But**: Different datasets have different baselines. Use **dataset-adaptive percentile threshold**:
- Reject if sharpness < max(5.0, dataset_p5_sharpness)
- Where dataset_p5_sharpness is the 5th percentile of sharpness for that dataset

**Evidence from data:**
- m_immobilier (real estate, all real photos): median sharpness = 6.9 at 640px
- kpmg_forensic (screenshots): median sharpness = 8.1 at 640px
- Real indoor photos CAN have sharpness as low as 3.0 (m_immobilier p5 = 3.0)

**→ Fixed global threshold of 5.0 would be too aggressive for indoor real estate photos.**
**→ Use dataset-adaptive: reject if sharpness < max(5.0, p5_of_dataset)**

#### Criterion 2.2: Colorfulness / Visual Appeal

| Rating | Colorfulness Range | Visual Evidence | Notes |
|--------|-------------------|-----------------|-------|
| EXCELLENT | 40 – 80 | Rich natural colors | Landscapes, well-lit scenes |
| GOOD | 20 – 40 | Moderate color saturation | Indoor, overcast, urban |
| FAIR | 8 – 20 | Low saturation | Industrial, fog, night |
| POOR | < 8 | Near-grayscale | Probable document/infographic |

**But**: Some real scenes ARE low-colorfulness (industrial interiors, foggy landscapes, night scenes). Combine with edge_ratio to disambiguate:
- Low colorfulness (< 8) + low edge ratio (< 0.15) = real (fog/night)
- Low colorfulness (< 8) + high edge ratio (> 0.25) = document

#### Criterion 2.3: Brightness / Exposure

| Rating | Mean Brightness | Visual Evidence |
|--------|----------------|-----------------|
| GOOD | 60 – 220 | Well-exposed |
| FAIR | 30 – 60 or 220 – 240 | Underexposed or slightly overexposed |
| POOR | < 30 or > 240 | Severely underexposed/blown out |

#### Criterion 2.4: Resolution

- Reject if either dimension < 200px (too small for gallery display)
- Prefer images with at least 800px on the longer side
- But: don't reject purely on resolution if content is excellent (small well-composed photos exist)

### Composite Quality Score

**Quality = 0.40 × sharpness_norm + 0.25 × colorfulness_norm + 0.15 × brightness_norm + 0.10 × size_norm + 0.10 × entropy_norm**

Each component normalized to [0,1] based on dataset-specific distributions.

---

## Axis 3: Gallery Display Suitability (GALLERY_READY / REVIEW_NEEDED / NOT_SUITABLE)

### Decision Matrix

| Quality \ Realness | REAL | PROBABLY_REAL | AMBIGUOUS | PROBABLY_NON_REAL | NON_REAL |
|-------------------|------|---------------|-----------|-------------------|----------|
| EXCELLENT | GALLERY_READY | GALLERY_READY | REVIEW | REJECT | REJECT |
| GOOD | GALLERY_READY | GALLERY_READY | REVIEW | REJECT | REJECT |
| FAIR | REVIEW | REVIEW | REVIEW | REJECT | REJECT |
| POOR | REVIEW | REJECT | REJECT | REJECT | REJECT |

### Tiebreaker Rules

1. **Prefer scenes over people**: Among equally scored candidates, prefer images with landscape/building/indoor scenes over single-person portraits.
2. **Prefer group activities**: If people are present, prefer group photos over single-subject photos.
3. **Prefer natural lighting**: Prefer images with brightness_std > 30 (indicating natural lighting variation) over flatly lit images.

---

## Boundary Case Handling Rules

### B1: High-quality 3D Renderings
- **Evidence**: High sharpness, rich colors, perfect exposure — but unnaturally uniform textures, perfect geometry, no noise
- **Action**: These pass heuristic filters. Flag dataset `digital_domain` as HIGH RISK. Route GALLERY_READY candidates to human review with note: "CGI risk — verify realness"
- **Rationale**: Heuristics alone cannot distinguish photorealistic CGI from real photos

### B2: Heavily Edited / Filtered Photos
- **Evidence**: HDR artifacts, extreme saturation, surreal colors (colorfulness > 120), unnatural shadows
- **Action**: If colorfulness > 100 AND edge_ratio < 0.10 → downgrade one quality level (FAIR → REVIEW)
- **Rationale**: Over-processed images look unprofessional for a corporate gallery

### B3: AI-Generated Images (Midjourney / DALL-E style)
- **Evidence**: Perfect sharpness, surreal lighting, impossible geometries, overly smooth textures
- **Action**: Cannot reliably detect with heuristics. Flag as "potential AI" if all heuristics are extremely favorable (>90th percentile across all metrics). Route to human review.
- **Rationale**: This is an unsolved problem for lightweight heuristics

### B4: Screenshots of Real Photos
- **Evidence**: Image contains a real photograph embedded in a webpage/UI. Edge ratio is high (from UI chrome), but the inner photo region passes heuristics.
- **Action**: If edge_ratio > 0.30 AND aspect_ratio > 1.3 → classify as NON_REAL (the whole image is a screenshot, not a photo)
- **Rationale**: The task requires the IMAGE ITSELF to be a photograph, not an image containing a photo

### B5: Document / Infographic Scans
- **Evidence**: High edge ratio, low colorfulness, high brightness, low brightness_std
- **Action**: R5 catches most. For borderline cases, check if colorfulness < 12 AND edge_ratio > 0.20 → NON_REAL
- **Rationale**: Documents have characteristic text-edge patterns

### B6: Small Thumbnails / Avatars
- **Evidence**: Either dimension < 150px
- **Action**: Reject as NOT_SUITABLE regardless of content
- **Rationale**: Too small for gallery display

### B7: Near-duplicates (dhash cluster)
- **Evidence**: Hamming distance ≤ 4 between dhash values
- **Action**: Within each cluster, keep only the highest-quality image (by composite quality score)
- **Rationale**: Avoid gallery redundancy

### B8: Single-Person Portraits
- **Evidence**: Face occupies > 25% of image area (rough heuristic: aspect ratio near 1:1 or 3:4, high central sharpness)
- **Action**: Downgrade in ranking preference. Allow at most 10% of final top 100 to be single-person portraits.
- **Rationale**: Gallery should show spaces and activities, not headshots

### B9: Corrupted / Unreadable Files
- **Evidence**: PIL/Pillow raises exception on open
- **Action**: Log file path and error. Do NOT silently skip. Count in stats.
- **Rationale**: Task requirement

### B10: Low-Entropy Placeholder Images
- **Evidence**: entropy < 3.0, brightness_std < 10
- **Action**: NON_REAL (solid color placeholder, loading image, or corrupted)
- **Rationale**: No meaningful visual content

---

## Empirical Calibration (from validation runs)

### Heuristic Distributions (validation results)

**Real-photo-heavy dataset (m_immobilier, n=200):**
- Sharpness at 640px: median=6.9, mean=8.4, p5=3.0, p95=18.8
- Colorfulness: median=29.2, mean=32.2, p5=7.7, p95=68.2
- Brightness: median=159.1

**Screenshot-heavy dataset (kpmg_forensic, n=80):**
- Sharpness at 640px: median=8.1, mean=7.9, p5=2.7, p95=13.4
- Colorfulness: median=60.0, mean=62.6, p5=24.8, p95=127.3
- Brightness: median=142.3

### Key Insight: Separation Strategy
- Sharpness alone CANNOT separate screenshots from real photos (distributions overlap heavily)
- Colorfulness: screenshots are MORE colorful than real photos, not less
- **The separator is edge_ratio + colorfulness**: screenshots have BOTH high edge_ratio AND high colorfulness, while real photos rarely have both
- A scatter plot of colorfulness vs edge_ratio would show two clusters:
  - Real photos: wide colorfulness range, low-medium edge_ratio
  - Screenshots/graphics: high colorfulness, high edge_ratio

### MobileNetV3-Small Feasibility
- 1.5M parameters (well under 100M limit)
- ~0.07s/image on CPU (Intel UHD 770)
- Full dataset (61,958 images): ~1.2 hours
- DINOv2-small: too slow (5.6s/image, ~96 hours for full dataset)

### MobileCLIP Status
- 'MobileCLIP-S0' NOT available in open_clip 3.3.0
- Available alternatives: 'MobileCLIP2-S0', 'MobileCLIP-S1', 'MobileCLIP-S2'
- These were NOT benchmarked. If needed, test with MobileCLIP2-S0 first.
- For now, MobileNetV3-Small is the recommended model due to speed.

---

## Known Failure Modes This Rubric Cannot Resolve

| Failure Mode | Why it fails | Mitigation |
|-------------|-------------|------------|
| Photorealistic CGI | Heuristics see "perfect real photo" | Flag digital_domain + human review |
| AI-generated images | Same as above — no signal differentiator | Human review for suspiciously "perfect" images |
| Real photo of a screen/monitor | Looks like screenshot but IS a photo of a real scene | Edge_ratio catches some, but imperfect |
| Low-light real photos | Dark + low colorfulness = false positive for document | Brightness_std helps (natural scenes have variation) |
| Abstract art photos | Artistic photos of textures/patterns | Human judgment needed |
| Cross-dataset threshold drift | m_immobilier p5 sharpness = 3.0, but another dataset may differ | Dataset-adaptive percentiles, not global thresholds |
| Panorama/360 photos | Extreme aspect ratio (2:1 to 4:1) — falsely rejected | Relax aspect ratio rejection to < 0.25 or > 4.0 (was 3.5) |

---

## Revision Notes (Round 1 → Round 2)

**Changes from my Round 1 position:**

1. **Removed MobileCLIP-S0 dependency.** The model is not available in the installed open_clip 3.3.0. Replaced with MobileNetV3-Small (1.5M params, ~0.07s/image) as optional model-based verification.

2. **Fixed colorfulness heuristic.** Stage 1 assumed "low colorfulness = infographic." Validation showed screenshots have HIGHER colorfulness than real photos. The actual separator is edge_ratio + colorfulness combined.

3. **Adopted dataset-adaptive percentiles.** Global thresholds fail because real estate photos (m_immobilier p5=3.0) differ from school photos (truro_school p5=14.0).

4. **Softened sharpness threshold for indoor real estate.** Real indoor photos can have sharpness as low as 3.0 at 640px. A global threshold of 5.0 would reject valid indoor photos.

5. **Added edge_ratio as primary separator.** This is the most informative single heuristic for distinguishing photos from screenshots/documents.

6. **Dropped DINOv2-small from pipeline.** At 5.6s/image it's too slow for any practical use even on candidate pools.

7. **Confirmed MobileNetV3-Small as viable fast model.** At 1.5M params and ~0.07s/image, it can run on the full dataset in ~1.2 hours if needed.
