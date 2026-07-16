=== CPU Pipeline Planner — Meeting Output ===
Meeting: mtg_7b90105776 (planning session)
Date: 2025-07-16

Delivered:
1. pipeline_plan.md — Full 3-stage cascade pipeline design document (shared directory)
2. pipeline_skeleton.py — Runnable Python skeleton with interface, data flow, and fallbacks (shared directory)

Pipeline summary:
- Stage A: Heuristic pre-filter (Sobel sharpness, Hasler-Susstrunk colorfulness, entropy, aspect ratio) with per-dataset percentile-based adaptive thresholds. ~70 min on all 61,958 images.
- Stage B: MobileNetV3-Small (2.5M params) neural classifier on Stage A survivors (~18,600). Photograph/non-photograph binary + quality scoring. ~3-6 hours.
- Stage C: Multi-resolution dhash dedup + diversity clustering + anti-portrait bias final ranking. ~10 min.
- Fallback: MobileCLIP-S0 for challenging CGI-vs-real datasets; heuristics-only mode if neural too slow.
Key risks: truro_school (59% of total, stratified sampling), digital_domain (CGI vs real, flagged for human review).
Output: 11 per-dataset top-100 galleries + review pools + aggregate stats in workspace/output/.
§
Final deliverable: final_pipeline_plan.md (shared directory) — Baseline-first pipeline with only 6 signals, 4 hard-reject rules, P10 soft filter, 1 composite score, 1 dhash dedup method. Zero-model baseline. MobileNetV3-Small only as optional add-on if CGI failure observed. Runtime estimate 10-30 min (honest, not optimistic). Key departure from v3: removed face detection, skin_ratio, homogeneity, horizontal_balance, DBSCAN, per-cluster quotas, per-dataset loosen factors, multi-level dhash. Explicit appendix listing everything intentionally excluded and why.
