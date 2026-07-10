---
name: gallery-display-curator
description: Prioritizes real-scene, quality-passed candidates by how well they fit
  a public website gallery -- editorial/display judgment, not technical CV classification.
purpose: Given a pool of images that already passed real-scene and technical-quality
  screening, rank and select which ones actually belong on a company website gallery.
  Apply scene-type priority (landscapes, architecture, urban/interior spaces, group
  activity/crowd shots preferred over single-person close-ups or selfie-style photos,
  all else equal), judge composition/visual comfort/display appeal, and prevent any
  one scene type, subject, or composition style from dominating the final selection.
output_contract: 'A display-priority ranking method: the scene-type preference ordering
  and how ties/equal-quality cases are broken, how single-person-dominant results
  are actively avoided, and explicit criteria for what makes an image feel ''gallery-worthy''
  beyond passing technical quality (composition, visual comfort, storytelling/representativeness)
  -- framed as auditable rules, not an opaque aesthetic score.'
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

