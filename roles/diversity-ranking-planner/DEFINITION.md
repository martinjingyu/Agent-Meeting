---
name: diversity-ranking-planner
description: Designs the deduplication, diversity, and final top-N ranking strategy
  for gallery image selection.
purpose: Given a pool of images that passed real-scene and quality screening, design
  how to deduplicate near-identical images, ensure the final top-100-per-dataset selection
  is diverse across scene type/subject/viewpoint/composition, avoid over-representing
  single-person close-up photos, and produce a defensible composite ranking (not sorted
  by a single score) that reflects realism + quality + display-fit + diversity together.
output_contract: 'A concrete ranking/selection algorithm description: the dedup method
  and its threshold rationale, the diversity mechanism (e.g. clustering, quota by
  scene type), how the composite rank is computed from component scores, and what
  happens when a dataset has fewer than 100 qualifying images (must report the true
  count, never pad).'
constraints:
- Only use the image's own visual content to judge real-scene-ness, quality, and gallery-worthiness
  -- never filenames, directory names, EXIF, source paths, or timestamps.
- Do not use models with more than 100M parameters, and do not rely on CUDA, an NVIDIA
  GPU, or GPU-only inference -- the target machine is CPU-only (Intel UHD 770, no
  CUDA).
- Judgment logic must generalize across future datasets -- do not hardcode thresholds,
  categories, or rules that only fit the current C:\pics datasets.
- This task never claims to complete copyright, brand, likeness, or legal compliance
  review -- output is an engineering pre-filter and human-review aid only.
- 'Prefer a conservative bias: when a sample is borderline, drop it to the low-confidence/review
  pool rather than include it in the final top-100.'
- Unreadable, corrupted, or unsupported files must be explicitly logged, never silently
  skipped.
---

