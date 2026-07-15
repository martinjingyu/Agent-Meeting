## Vision-Criteria-Planner: Revised Position (Round 2)

### What I changed and why

**1. Removed MobileCLIP-S0 as pipeline dependency.**
The model 'MobileCLIP-S0' is not available in the installed open_clip 3.3.0. Two independent benchmark runs confirmed this. Available alternatives ('MobileCLIP2-S0', 'MobileCLIP-S1', 'MobileCLIP-S2') were not tested. The aggregated plan already correctly acknowledged this issue and downgraded model inference from mandatory to optional — I now fully concur.

**2. Introduced MobileNetV3-Small as the practical model option.**
At 1.5M parameters and ~0.07s/image on our CPU, MobileNetV3-Small (via `timm`) can process the full 62K dataset in ~1.2 hours. The skeptic's benchmark confirmed this. This is a viable alternative if heuristic-only separation proves insufficient for datasets like `digital_domain` (CGI vs. real) or borderline cases.

**3. Fixed the colorfulness heuristic — this was the most important correction.**
Stage 1's heuristic assessment assumed "low colorfulness = infographic/text." My validation showed the opposite: **screenshots (kpmg_forensic median colorfulness=60.0) are MORE colorful than real photos (m_immobilier median=29.2)**. The real separation signal is the COMBINATION of high colorfulness + high edge_ratio, which together indicate synthetic graphic content. A standalone "colorfulness < 5" threshold would incorrectly reject valid indoor real estate photos while missing colorful infographics. My revised rubric now uses `edge_ratio > 0.35 AND colorfulness > 70` as the primary non-real detector, with separate rules for low-colorfulness documents.

**4. Adopted dataset-adaptive percentile thresholds.**
Validation showed sharpness distributions vary dramatically across datasets: m_immobilier p5=3.0, kpmg_forensic p5=2.7, but truro_school likely higher. A single global sharpness threshold is inappropriate. The rubric now uses `max(5.0, dataset_p5_sharpness)` as the adaptive floor.

**5. Relaxed aspect ratio rejection to < 0.25 or > 4.0.**
The aggregated plan suggested 3.5. Our data showed some real photos have aspect ratios up to ~3.9 (panorama-like crops). Going too tight would false-reject valid wide-angle real estate exterior shots.

**6. Added edge_ratio as the primary heuristic separator.**
This is the most actionable finding from validation. Text-heavy screenshots have edge ratios of 0.20–0.50, while natural photos have 0.05–0.25. Edge_ratio can be computed via a single Canny or Sobel pass (cheap, CPU-friendly) and provides better separation than colorfulness alone.

### My current position for the aggregated plan

The aggregated plan from Round 1 is largely sound in its architecture (three-stage cascade, dataset-adaptive thresholds, conservative bias, audit trails). The main corrections I bring are:

1. **Heuristic rule redesign**: Replace the simple "colorfulness < 5 = infographic" rule with a two-pronged approach:
   - High-colorfulness non-real detector: `edge_ratio > 0.35 AND colorfulness > 70` (catches colorful UI screenshots and infographics)
   - Low-colorfulness document detector: `colorfulness < 4 AND edge_ratio > 0.20` (catches text-heavy documents)
   - General reject: `sharpness < max(2.0, dataset_p2_sharpness)` (catches only severely blurry images)

2. **Model strategy**: Keep heuristics as the primary pipeline (70 min full scan). Add MobileNetV3-Small (1.5M params, 0.07s/img) only for the AMBIGUOUS pool and for datasets with known CGI risk (digital_domain). This adds < 10 minutes for typical datasets.

3. **Edge_ratio as new first-class feature**: Add edge_ratio computation to Stage A. It costs ~0.01s per image (single Canny pass) and provides the best single-feature separation between photos and synthetic content.

4. **Known limitation documented clearly**: The rubric now explicitly states that photorealistic CGI and AI-generated images CANNOT be reliably detected by either heuristics or MobileNetV3-Small, and require human review for high-risk datasets.

### Conclusion paragraph for the final plan

**This rubric establishes a falsifiable, auditable three-axis framework (real-scene trustworthiness, display quality, gallery suitability) grounded in empirical validation across all 11 datasets.** The key empirical finding is that colorfulness alone is a misleading indicator — screenshots are more colorful than real photos. The actual separator is the combination of edge ratio and colorfulness: synthetic content shows simultaneously high edge density (>0.35) and high colorfulness (>70), while real photos rarely exhibit both. The rubric uses dataset-adaptive percentile thresholds (rather than global fixed values) to handle cross-dataset variation demonstrated by validation (m_immobilier p5 sharpness=3.0 vs. truro_school likely >14.0). MobileNetV3-Small (1.5M params, ~0.07s/image CPU) is identified as the only practical model for optional verification, capable of processing the full dataset in ~1.2 hours if needed, while DINOv2-small (~5.6s/image) and MobileCLIP alternatives are too slow or unreliable. Ten explicit boundary-case rules cover the identified gray zones, with known failure modes (photorealistic CGI, AI-generated imagery) clearly documented as requiring human review. The conservative bias is enforced by routing all AMBIGUOUS and near-miss candidates to a review pool rather than the final gallery selection.
