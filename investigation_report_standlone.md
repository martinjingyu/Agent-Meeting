# Investigation of the real image datasets

**Date:** 2026-08-20  
**Scope:** bounded pre-implementation investigation  
**Source:** `/corpus`, inspected locally without web research about any entity  
**Result:** investigation complete; no production gallery system and no final galleries were built

## Executive summary

The corpus contains 20 organizations, 33,780 candidate files, and about 14.84 GB
of candidate data. Safe raster decoding succeeded for 33,768 files and failed
for 12. Every organization received a summary review and a visible mixed-source
inspection. Selected cases then received complete ranked-band, opening-sequence,
duplicate-boundary, signal, or 100-image comparison reviews.

The central finding is that this is not one ranking problem. The data contains
several materially different gallery regimes:

- Rich sources such as M Immobilier, IMG Academy, Truro School, Digital Domain,
  TÜV Rheinland, UL Solutions, Maior Capital, and Boston University visibly
  contain enough attractive, relevant material for a full 100-image gallery,
  often with worthwhile candidates well beyond rank 100 in a rough diagnostic
  ordering. A small fixed cap would discard useful breadth.
- Pali Adventures and CMAS show a gradual marginal-value transition: about the
  first 100 diagnostic candidates remain broadly useful, then weaker, repeated,
  portrait-, poster-, or logo-heavy material increasingly appears. This is the
  clearest evidence that stopping should be based on remaining contribution, not
  source count or a universal score threshold.
- KPMG Forensic is genuinely source-limited: all 80 decoded images were viewed,
  and the source contains only a few repeated stock-photo families before dense
  charts and report graphics take over. THEMA is also genuinely difficult: much
  of its apparent breadth is flags, landmarks, maps, or generic regulatory and
  medical stock. Roland Berger has thousands of files but is dominated by
  report pages and charts; it has more photography than KPMG, but its exact
  defensible gallery ceiling remains uncertain.
- Scalar semantic relevance does not equal public-gallery suitability. Summary
  matching repeatedly promoted logos, charts, montages, and documents whose
  information is unreadable at carousel scale. A generic “professional photo”
  direction promoted arbitrary headshots and travel-like scenes. A contextual
  activity direction was usually more useful, especially for schools and
  technical organizations, but still could not prove entity attribution.
- Diversity is useful only with safeguards. Gentle same-set reordering visibly
  reduced adjacent runs without changing selection. Aggressive embedding
  novelty sometimes rescued campus, academic, facility, or activity views, but
  also promoted portraits, posters, diagrams, QR/certification marks, and
  miscellaneous editorial material. Difference alone is not useful variety.
- Exact hashes found no byte-identical within-entity groups, despite obvious
  resized, recompressed, cropped, and overlaid derivatives. Perceptual hashing
  is therefore useful, but no global distance is safe: Butler derivatives remain
  visually identical at comparatively large distances, while Boston floor-plan
  templates can represent different floors even at very small distances.
- A 256M-parameter vision-language model often described the broad visible form
  of 70 difficult cases, but confused fine activities, missed decisive branding,
  hallucinated details, and collapsed under a more prescriptive prompt. On this
  host restricted to eight logical CPUs, the tested global-caption path took
  8.34 seconds per image—about 23.2 hours for 10,000 by straight-line arithmetic.
  It is a useful diagnostic and a possible bounded-review research direction,
  not evidence for a production choice.
- The lighter MobileCLIP2-S0 single-view embedding direction processed 1,000
  thumbnails at 8.22 images/second on the same eight-logical-CPU allocation. A
  linear 10,000-image inference-only projection is about 20.3 minutes. The
  larger SigLIP2 comparison was about 4.80 images/second on 250 images and was
  not consistently better visually. These are promising navigation costs on
  this unusually strong desktop CPU, not target-worker or gallery-quality proof.

The evidence supports several directions worth testing later, but it does not
select a production architecture. In particular, it does not establish a final
stopping rule, a semantic model, a shortlist size, an MMR weight, a pHash
threshold, or a required VLM stage.


## Environment and practical constraints

### Observed development environment

- Ubuntu 26.04 LTS in a Docker container on WSL2, kernel
  `6.6.87.2-microsoft-standard-WSL2`.
- AMD Ryzen 9 9950X: 16 physical cores, 32 logical CPUs; process affinity
  initially covered CPUs 0–31.
- 93.1 GiB RAM and 24 GiB swap exposed to the container. Available RAM varied
  during the run; approximately 82–85 GiB was available during inventory/final
  collection.
- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported VRAM,
  driver 596.72. It was used to accelerate investigation-wide embeddings and
  VLM comparisons.
- `/workspace` is an ext filesystem with roughly 454 GiB free at final
  collection. `/corpus` is a 9p/DrvFS mount and remained explicitly `ro`.
- The corpus is about 14 GB, and the private investigation workspace grew to
  include a 7.3 GB virtual environment, 2.2 GB model cache, 789 MB thumbnail
  cache, and additional embeddings/sheets. None of those temporary assets is in
  `output/`.

This is a high-end development machine, not a proxy for the later approximate
8-vCPU/16-GiB CPU-only worker. GPU timings are useful only for investigation
throughput. Eight-logical-CPU affinity tests on this processor are stronger
evidence than an unrestricted run but still do not reproduce a cloud CPU,
physical-core allocation, or memory constraint.

The core software used for investigation included Python 3.12.12, Pillow 12.3,
NumPy 2.5, OpenCV 5.0, pandas 3.0, PyTorch 2.11, Transformers 5.15, and
OpenCLIP 3.3. These versions describe the experiments, not production
dependencies.

### Corpus safety

`/corpus` was mounted read-only for the full run. A write probe was rejected by
the filesystem before investigation, and final inspection still reported the
`ro` mount option. The final metadata fingerprint (sorted relative path, byte
size, and nanosecond mtime—not a byte-content hash) is recorded in
`measurements.json`. No source files were renamed, rewritten, or reorganized.

## What was inspected and how

All 20 `summary.md` files were read first. Candidate discovery was recursive and
excluded each summary. The current corpus happens to be flat: there were no
nested candidate files, symlinks, hidden directories, or missing summaries.
Those absences describe only this dataset and do not test future nested or
adversarial discovery behavior.

Every candidate received a safe raster decode attempt. Successful decodes
produced an investigation-only thumbnail (maximum side 384 px), dimensions,
format identification, simple brightness/entropy/edge/sharpness diagnostics,
and perceptual hashes. This was broad inspection infrastructure, not a gallery
implementation.

For every entity, a deterministic light sheet showed up to 144 decoded images:
104 pseudo-uniform source samples plus targeted small, unusual-aspect,
text/graphic-like, flat/blank, and soft-image probes, with overlap filled by the
uniform sample. All 80 KPMG images were shown. Every one of these 20 sheets was
visually inspected. This ensures broad coverage but does not mean every pixel of
every large entity received equal scrutiny.

Deeper cases were chosen only after that pass. They covered rich photographic
sources, source-limited cases, derivative-heavy scrapes, text-heavy corpora,
schools, property sets, technical laboratories, special content, and visible
relevance boundaries. Full 100-image temporary comparison sheets were inspected
for IMG Academy, M Immobilier, Tara Guérard, and TÜV Rheinland. All four 50-image
bands through rank 200 were inspected for 16 entities. These rankings were
deliberately rough navigation instruments and were never treated as final
galleries.

## Visible trace for every organization

| Entity | Decoded / failed | Visible collection and important issue | Depth and rationale |
|---|---:|---|---|
| `boston_university` | 2,720 / 5 | Extremely heterogeneous: campus, teaching, research, residences, events, headshots, floor plans, charts, UI, food, and editorial art. Strong entity context exists but is sparse in an unfiltered view. | **Deep:** signal, rank-band, opening, duplicate, and VLM cases; useful for visible relevance, rescue recall, and broad-opening questions. |
| `butler_school` | 1,676 / 0 | Mostly tiny 150×150 or 66×66 news derivatives. Useful service training, table setting, food, graduation, and hotel scenes coexist with logos and many resized copies. | **Focused:** light view plus pHash boundary audit. Further semantic work would mostly repeat richer training/school cases. |
| `cmas_org` | 398 / 0 | Distinctive underwater sport and diving, mixed with flags, logos, posters, headshots, sponsors, and extreme banners. | **Deep:** complete length transition, signal disagreement, opening, novelty, duplicate, and VLM cases; central special-content counterexample. |
| `digital_domain` | 1,669 / 0 | Strong VFX, film, advertising, virtual-human, and behind-the-scenes imagery; also headshots, dark plates, shot derivatives, and scenery with weak visible attribution. | **Deep:** ranked bands, signals, and pHash. Important because synthetic/fantastical imagery is authentic output rather than an automatic defect. |
| `heliservices` | 541 / 0 | Coherent helicopters, aerial work, base, passengers, and Hong Kong views, plus icons, diagrams, awards, food/social images, and repeated aircraft views. | **Light only:** useful ordinary positive case; main patterns were examined more deeply in technical and motif-heavy comparators. |
| `img_academy` | 925 / 0 | Polished multi-sport training, coaching, campus, academics, and student life, mixed with scorecards, logos, UI, and blanks. | **Deep:** rich-length bands, all opening methods, and three full 100-image comparisons; clearest archetype-collapse and forced-variety case. |
| `kpmg_forensic` | 80 / 0 | A few repeated generic office/people stock families, then charts, diagrams, awards, and report graphics. | **Deep/full source:** all 80 viewed; duplicate, signal, length, and VLM evidence. Establishes real source limitation rather than selector failure. |
| `lakefield_college_school` | 518 / 0 | Good outdoor, classroom, arts, athletics, and campus imagery plus history, arbitrary user thumbs/headshots, icons, social marks, black frames, and low-resolution derivatives. | **Light only:** boundary patterns were strongly represented by Northwest, Truro, IMG, and Boston. |
| `m_immobilier` | 5,000 / 0 | Consistently high-resolution interiors and exteriors, with plans, maps, neighbourhood views, and many similar empty rooms. | **Deep:** ranks through 200, signals, duplicate boundaries, opening, and two complete 100-image sets; tests long-gallery breadth and pool bias. |
| `maior_capital` | 4,994 / 6 | Rich interiors, exteriors, pools, aerial/coastal views, and plans. Nearly every useful photo carries a large baked-in watermark. | **Deep:** ranks through 200 and duplicate review; key counterexample to categorical watermark rejection. |
| `northwest_school` | 516 / 0 | Strong classes, arts, athletics, outdoors, and community, but many isolated student/staff portraits, teams, banners, text graphics, controls, and blanks. | **Deep:** ranks through 200; useful portrait/context and opening-motif evidence. |
| `pali_adventures` | 708 / 0 | Camp activities and facilities with conspicuous resized/recompressed repeats of food, ropes, groups, cabins, controls, and small web assets. | **Deep:** all bands through 200 and pHash audit; strongest gradual stopping and derivative case. |
| `peddie_org` | 543 / 0 | Many small headshots and external college logos/icons; fewer but useful campus, arts, sports, classroom, and group scenes. Count overstates usable breadth. | **Light only:** retained as a counterexample in the all-entity evidence; deeper school cases covered the same uncertainty with richer alternatives. |
| `roland_berger` | 2,997 / 0 | Dominated by report pages, charts, and branded quote cards; a minority of events, offices, portraits, and topical/editorial photos. Most text is unreadable at sheet scale. | **Deep:** signals, ranks, and VLM boundary cases. Tests semantic relevance versus carousel suitability; exact usable ceiling remains uncertain. |
| `rosey_summer_camps` | 1,260 / 0 | Rich sport, outdoor, arts, academic, cooking, and social scenes, mixed with repeated category banners, UI cards, flags/logos, and derivative families. | **Deep:** ranks through 200 and pHash audit; supports a long gallery with gradual tail decline. |
| `tara_guerard` | 4,970 / 1 | Beautiful weddings, venues, florals, tables, and stationery mixed with collages, jewelry/products, publications, screenshots, fashion, personal/editorial material, and old scans. | **Deep:** signals, ranks, opening, novelty, two complete 100-image sets, duplicate and VLM cases; exposes relevance and montage-heavy ranking failures. |
| `thema_med` | 273 / 0 | Generic flags, landmarks, maps, regulatory/medical stock, and report covers; few visibly organizational people or events. | **Deep:** broad source/rank inspection, signals, and VLM cases; strongest geographic-proxy and genuinely difficult-source case. |
| `truro_school` | 1,000 / 0 | Broad, high-resolution school life across ages, academics, arts, sports, outdoors, boarding, and community, with only occasional documents and weak frames. | **Deep ordinary comparator:** ranks through 200 and all opening methods; strong evidence that 100 need not be padding. |
| `tuv_rheinland` | 1,465 / 0 | Many diagrams, certification marks, and dense service graphics, plus a smaller but still abundant pool of real labs, inspections, equipment, mobility, and staff work. | **Deep:** ranks, signals, pHash, opening, three full 100-image sets, and VLM cases; central relevance/suitability and ordering case. |
| `ul_solutions` | 1,515 / 0 | Polished testing, industrial, product, energy, lab, and people imagery plus stock-like heroes, documents, screenshots, portraits, and panels. | **Deep comparator:** ranks and signals; tests whether contextual technical work generalizes beyond TÜV. |

The less deeply examined entities were Heliservices, Lakefield, and Peddie;
Butler received only a focused duplicate study beyond its light pass. This is an explicit
coverage limit, not a claim that they need no special handling later.

## Corpus facts and scraping problems

### Observation

- Candidate count ranged from 80 (KPMG) to 5,000 (M Immobilier and Maior
  Capital). Candidate count did not track usable visual breadth.
- Median decoded size was 1.08 megapixels, but 2,583 images were under 0.1 MP.
  Butler alone had 1,218 such images; Peddie had 244 and Rosey 332.
- 374 images had an aspect ratio outside 1:3 to 3:1. The largest decoded asset
  was a 18,898×18,898 TÜV graphic (357 MP), showing that maximum-size guards
  matter even though most files are ordinary.
- Pixel formats were 29,194 JPEG, 4,206 PNG, 340 WebP, 17 MPO, 8 GIF, and 3 BMP.
  Filename suffixes did not reliably describe pixels: at least 44 `.png` files
  decoded as another format and 36 `.jpg`/`.jpeg` files decoded as non-JPEG.
- The 12 failures included three unsupported SVGs, two `.png` files that Pillow
  could not identify, six tiny invalid `.jpeg` files in Maior, and one invalid
  Tara `.jpg`. A failing entity/file did not prevent unrelated inspection.
- A full development inventory—discovery, decode attempt, thumbnail, basic
  metrics, perceptual hashes, and scratch output—took 222.47 seconds with 12
  worker processes on this host. This is not a constrained production result.

### Interpretation and scoped conclusion

Decode-based discovery and inexpensive structural diagnostics appear practical
and necessary. Extensions, resolution, edge density, and sharpness help find
problems, but they are not standalone quality judgments. Tiny Butler derivatives
may still reveal a unique activity; a huge report page may remain unsuitable;
and a low-sharpness historical image may add real context. Later work should use
such signals as triage evidence and keep explicit decode/skipping provenance.

### Limitation

The current source has no nested files, symlinks, hidden entity directories, or
missing summaries. This investigation therefore did not establish safe future
behavior for those cases; that belongs in implementation fixtures and tests.

## Duplicate and repetition findings

### Question

Can an inexpensive perceptual hash suppress human-visible derivatives without
collapsing genuinely different views?

### Experiment

All decoded thumbnails received 256-bit pHash and SHA-256 grouping within each
entity. Nearest pairs were sampled across pHash distance bands 0–4, 5–8, 9–12,
13–16, 17–24, and 25–32. Eight entity sheets were visually audited, including
derivative-heavy, templated, property, creative, and technical sources.

### Observation

- SHA-256 found zero byte-identical within-entity groups. Yet the sheets contain
  abundant obvious human duplicates, proving that exact hashing alone would
  miss the important scraped-derivative problem.
- A transitive pHash-distance-12 probe placed 1,638 of 1,676 Butler images,
  1,235 of 1,260 Rosey images, and 552 of 708 Pali images into clusters. Visual
  checks confirmed extensive resized/recompressed families in these sources.
- On Butler, audited pairs even at distances 20–32 were usually the same source
  image in another size or crop. Distance 12 is conservative there and leaves
  human-obvious duplicates beyond the threshold.
- Boston provides the opposite failure. Different residence floor plans with a
  shared template can fall at distance 2–4. They may be motif-redundant in a
  public gallery, but they are not the same image or necessarily interchangeable.
- M Immobilier distances 26–32 included both same-scene/listing derivatives and
  genuinely different views worth retaining. Digital Domain showed the same
  boundary between redundant frames and meaningful production/shot variation.
- TÜV's pHash-near material often consisted of different templated diagrams or
  certification marks. That is useful evidence for motif control, not exact
  duplicate identity.

### Interpretation

Pixel hashes solve a narrow provenance problem; pHash is useful for derivative
navigation; and motif-level repetition is a separate gallery-level judgment.
The three should not be collapsed into one threshold.

### Scoped conclusion

Perceptual comparison appears production-plausible, but one global hard distance
is unreliable without additional evidence such as dimensions, crop/overlay
relationships, stronger local features, or conservative ambiguity handling.
Related subjects must remain eligible when their composition, action, viewpoint,
or informative detail contributes something new. The audit did not compare all
modern duplicate methods, so it does not select a final technique.

## Visible relevance and embedding-signal findings

### Question

Do single-view image/text embeddings distinguish attractive organizational context
from arbitrary portraits, geographic proxies, dense graphics, and generic stock?
Does a substantially larger embedding model materially improve these cases?

### Experiment

Two OpenCLIP-compatible models were applied to all 33,768 thumbnails on the
development GPU using each model's standard single-view preprocessing (no
multi-crop; the standard transform can resize/center-crop):

- MobileCLIP2-S0 (`dfndr2b`), 74.8M parameters, approximately 286 MiB cached;
- ViT-B-16-SigLIP2 (`webli`), 375.2M parameters, approximately 1.5 GiB cached.

For 12 contrasting entities, the top 24 results were compared for raw summary
sentence similarity and generic prompt contrasts covering professional photos,
real activity/context, facilities/products, contextual people, isolated
portraits, stock, flags/maps/landmarks, documents/charts, UI, panels, technical
work, property views, education, events, and synthetic artwork. Prompts and
linear blends were probes, not trained or calibrated decision rules.

### Direct observation

- Raw summary similarity often found visible identity—Boston signs/buildings,
  KPMG/TÜV marks, entity graphics—but also strongly preferred dense text pages,
  logos, and montages. This is semantic relevance without carousel suitability.
- “Professional photo” was actively misleading on several sources. It promoted
  isolated headshots at Boston, CMAS, and Roland Berger and travel/landmark
  scenes for THEMA. Attractive pixels did not make these representative.
- Context-over-proxy scoring was repeatedly useful as navigation: underwater
  action for CMAS, classroom/lab work for schools, production process for
  Digital Domain, and labs/inspections for TÜV and UL. It still admitted generic
  industry stock and could not establish that a pictured person, lab, or office
  belonged to the named entity.
- Model choice changed the failure mode. For CMAS, MobileCLIP's photo direction
  favored headshots while SigLIP2 found underwater action. For other sources the
  larger SigLIP2 was not consistently better and sometimes ranked more travel,
  graphic, or narrow-archetype material.
- IMG's blended results remained too sport/action-heavy despite a source rich in
  campus, academic, boarding, coaching, and facility images. M Immobilier's
  photo/context directions remained interior-heavy despite many exteriors.
- Tara's summary direction favored montages and branding while its context
  direction found more useful event scenes. Roland Berger's summary direction
  was nearly all charts, while photo directions included arbitrary editorial
  portraits and topical scenes.

### Interpretation

Global embedding signals appear valuable for candidate navigation and for
constructing a two-sided review pool, especially when several complementary
prompts are retained. They do not provide visible attribution, human-level
format suitability, or reliable breadth by themselves. Larger parameter count
did not translate into a generally superior curation signal on these cases.

### Feasibility observation

GPU inference over all 33,768 thumbnails took 62.15 seconds for MobileCLIP2-S0
and 58.19 seconds for SigLIP2, excluding first-time asset acquisition. Those are
development conveniences only.

With CUDA hidden and affinity restricted to logical CPUs 0–7 on this host:

| Direction | Sample | Inference | Rate | Peak process RSS | Linear 10k inference-only arithmetic |
|---|---:|---:|---:|---:|---:|
| MobileCLIP2-S0 | 1,000 | 121.69 s | 8.22 images/s | 2.75 GB | 20.3 min |
| SigLIP2-B/16 | 250 | 52.11 s | 4.80 images/s | 4.07 GB | 34.7 min |

The samples were deterministic cross-entity strides, standard single model
views, and
unoptimized PyTorch/OpenCLIP runs. Model loading was 2.74 and 4.47 seconds,
respectively. These measurements suggest corpus-wide embeddings deserve later
production testing, but the projection excludes source decoding, scoring,
deduplication, deeper review, selection, ordering, materialization, and report
generation. It also does not compensate for a slower target CPU. The 250-image
SigLIP sample makes its 10k projection especially uncertain.

### Scoped conclusion and limitation

MobileCLIP2-S0 is a plausible cost/quality baseline for later tests, not a chosen
model. SigLIP2's larger footprint and lower measured CPU rate were not justified
by consistent visible gains in this investigation. Different embedding families,
fine-tuning, crops, resolutions, or learned combinations were not exhaustively
compared.

## Gallery length and stopping behavior

### Question

How much worthwhile material do different entities visibly contain, and can a
single scalar decline identify when marginal additions stop helping?

### Experiment

A deliberately simple diagnostic ranking combined within-entity percentiles of
MobileCLIP summary similarity, context-over-proxy, photo-over-graphic, and a
minor beauty prompt (weights 0.25/0.40/0.30/0.05). A relaxed 180×160 minimum,
aspect range 0.25–4.0, and greedy pHash-distance-12 suppression were applied.
Four complete 50-image bands through rank 200 were visually inspected for 16
entities. This construction intentionally exposed behavior; it is not a proposed
selector or stopping formula.

### Direct observation

**Rich through and beyond 100.** M Immobilier, Truro, IMG Academy, Digital
Domain, Boston, TÜV, Maior, and UL remained predominantly attractive and useful
through rank 200. Rosey remained strong through 150 and still useful at 151–200.
Northwest's later band added more portraits/groups but retained useful scenes.
These sources visibly refute any assumption that a careful gallery should
naturally stop near 20–40.

**Gradual transition.** Pali's 1–100 formed a broad camp story; 101–150 became
more candid and repetitive; 151–200 mixed a few valuable activities/facilities
with weak portraits, low-resolution frames, and repeated motifs. CMAS remained
strong through about 100 and retained useful material into roughly the 140s,
after which logos, portraits, and posters increasingly dominated. Neither
source presents a clean score cliff.

**Limited or misleading abundance.** KPMG exhausted its few stock-photo families
quickly. THEMA's first few dozen were at least visually plausible medical or
regulatory stock, but geography and topical proxy material soon dominated.
Roland Berger's photography was concentrated near the top; charts and report
pages filled much of 51–200. Tara maintained a high diagnostic score through
200 even while montages and old blog panels made the set less suitable than
clean single event photographs elsewhere in the source.

The diagnostic median scores illustrate why one number is insufficient:

| Entity | 1–50 | 51–100 | 101–150 | 151–200 | Visual reading |
|---|---:|---:|---:|---:|---|
| M Immobilier | .900 | .867 | .838 | .821 | All bands remain useful; motif repetition is the main weakness. |
| IMG Academy | .803 | .738 | .702 | .672 | Strong throughout, but perspective breadth is badly ordered. |
| Pali Adventures | .706 | .581 | .480 | .374 | Score decline broadly tracks a real gradual quality/repetition decline. |
| CMAS | .786 | .702 | .575 | .465 | Useful to near 100; posters/logos/heads increasingly invade later. |
| Roland Berger | .883 | .833 | .813 | .788 | Scores remain high while unreadable report material dominates. |
| Tara Guérard | .881 | .843 | .828 | .812 | High plateau hides montage/panel suitability problems. |
| THEMA | .731 | .601 | .525 | .423 | Decline accompanies increasing geography/topical proxies. |
| KPMG | .493 | .187 | — | — | Only 62 survived relaxed gates/dedup; later material is mostly graphics. |

Scores are only included to show their disagreement with the sheets; visual
inspection supports the conclusions.

### Interpretation

Several kinds of evidence matter at the margin: continued public suitability,
visible relationship to the entity or its work, nonredundant information, and
the availability of stronger complementary perspectives. An entity-relative
score decline can sometimes help, but it cannot distinguish a rich montage
plateau from a rich single-photo tail, and its numerical scale is not portable
between entities.

### Scoped conclusion

The evidence supports a true marginal-value decision and explicitly rejects
both fixed small caps and automatic filling. It does not yield a tested automatic
stopping rule. Later work should evaluate the tail against competitive
unselected material and visually verify unusually short, motif-dominated, or
cap-hitting results. A long gallery can be correct; a short gallery can also be
correct; source count alone establishes neither.

## Opening sequences, useful variety, and ordering

### Question

Can embedding novelty improve a gallery's explanatory breadth, and can ordering
improve an already good set without changing selection?

### Experiment

For Boston, CMAS, IMG, M Immobilier, Tara, Truro, and TÜV, a 500-item diagnostic
pool was used to compare four automatic 100-image sequences:

1. strength-sorted top 100;
2. the exact same 100, locally reordered within a 12-candidate moving window to
   reduce similarity to the previous three items;
3. mild MMR (0.82 strength / 0.18 embedding novelty); and
4. an intentionally aggressive novelty stress test (0.45 / 0.55).

The first 12 were compared for all seven. Complete sheets were inspected for all
four methods on IMG, M Immobilier, Tara, and TÜV.

### Observation: openings

- A sequence of individually strong images can still explain too little. IMG's
  score opening was dominated by girls' wrestling and related sports; TÜV's by
  laboratories and apparatus; M Immobilier's by empty interiors; Truro's by
  younger-pupil/classroom scenes. Complementary campus, academic, exterior,
  field, client, and performance views existed later.
- Boston's first dozen were plausible but less visibly identifiable and less
  collectively explanatory than combinations of branded campus, teaching,
  research, activity, and student-life candidates available elsewhere.
- Reordering improved rhythm but did not repair a biased opening set. The first
  item also remained fixed in every probe, so this experiment did not solve the
  anchor/hero choice.
- Aggressive novelty sometimes improved the opening: IMG gained campus,
  academics, facilities, and boys' sports; Truro gained older pupils, campus,
  and performance. It also produced clear regressions: CMAS gained an arbitrary
  portrait, old meeting, poster, and equipment close-up; TÜV placed an
  infographic second; IMG admitted promotional/blurred material.

### Observation: complete sets and ordering

The same-set local spread reduced mean adjacent embedding cosine similarity for
every entity: IMG .499→.476, TÜV .510→.467, Boston .381→.326, and Tara
.524→.498, with comparable changes in the other cases. The effect was visibly
useful in several immediate runs while preserving exactly the same selected set.

It could not recover missing perspectives. IMG remained sport-archetype-heavy,
M Immobilier remained interior-heavy, and Tara remained montage-heavy because
those problems already existed in selection or pool construction.

Mild MMR changed only 1–6 selections per 100 and usually made little visible
difference. Forced novelty changed 8–39 and reached as deep as diagnostic pool
rank 372. Some additions genuinely increased explanatory breadth; others were
weak precisely because they were different. On M Immobilier, even forced
novelty mostly found different interiors, demonstrating that a narrow candidate
pool cannot be repaired by sequence optimization.

### Interpretation

Selection and ordering are separable. A narrow same-set ordering mechanism has
credible evidence for reducing awkward adjacency after a good set exists.
Embedding diversity is a fallible clue to missing perspectives, not a universal
reward. Its success depends on having strong complementary candidates in the
pool and on protecting relevance, suitability, and opening quality.

### Scoped conclusion and limitation

The later system should evaluate the first several images collectively and
compare its order against an informative baseline. This evidence does not choose
MMR, a novelty weight, a window size, or a fixed opening category pattern. The
temporary pool inherited biases from the experimental score and pHash rule, and
there was no formal human preference panel or reference gallery.

## Visible relevance and special content in context

The repeated boundary cases support contextual rules, not broad bans:

- **People and portraits.** Isolated Boston, CMAS, Northwest, and Roland Berger
  portraits often looked polished but communicated no organization in their
  pixels. People teaching, competing, testing, presenting, creating, or working
  in a visible facility were much more explanatory. A staff portrait could
  still be useful in another context or later position; “contains a face” is not
  a sufficient rejection rule.
- **Flags, landmarks, and maps.** THEMA's isolated national and city imagery
  increasingly substituted geography for the organization. The same content
  can help when embedded naturally in a real ceremony, competition, facility,
  or operational scene. The evidence argues against proxy dominance, not flags.
- **Text and branding.** Boston's large name wall and branded campus entrance
  add visible identity to real scenes. TÜV signage inside a test facility is
  useful. KPMG/Roland report pages and TÜV infographics may be exactly on-topic
  yet are unreadable or unattractive at normal sheet scale. OCR or text match
  cannot stand in for carousel value.
- **Graphics and collages.** Tara's collages visibly relate to events but were
  over-promoted relative to cleaner single photographs. Clear visual artwork may
  still represent a creative entity; an unreadable slide usually does not.
- **Synthetic-looking imagery.** Digital Domain's fantastical renders, virtual
  humans, and film frames are authentic examples of its work. A generic
  synthetic-image ban would destroy representativeness.
- **Watermarks.** Maior's large watermark is baked into almost every useful
  property photograph. A universal watermark ban would empty an otherwise rich
  source; the watermark remains a visual cost that should affect comparisons
  where cleaner alternatives exist.
- **Generic contextual work.** UL and TÜV technical scenes explain testing and
  engineering well, even when the pixels do not independently prove which
  company produced them. These can be useful, but a gallery dominated by
  interchangeable stock-like scenes would still under-explain the entity.

The recurring two-question distinction is supported by the data: does the image
help explain this organization or its visible world, and is it pleasant/useful
as a carousel image? Neither semantic association nor aesthetics alone answers
both.

## Small-VLM boundary experiment

### Question

Can a very small global-view VLM add useful evidence on ambiguous inclusions and
competitive exclusions, and is it practical on the later CPU envelope?

### Experiment

The 70-case set was the union of top results from competing embedding signals
for nine entities plus early candidates introduced by forced novelty for four
entities. It was deliberately boundary-heavy, not a random accuracy set. Each
used the same maximum-384-pixel global thumbnail, no crops or separate OCR,
deterministic generation, batch size four, and a 48-token limit.

Model: `HuggingFaceTB/SmolVLM-256M-Instruct`, 256.5M parameters, pinned snapshot
`7e3e67edbbed1bf9888184d9df282b700a323964`, approximately 494 MiB cached.

Two prompts were compared:

- short: “Briefly describe the image.”
- prescriptive: a longer request to mention readable names/logos and distinguish
  acting, posed, or isolated people without inferring provenance.

All 70 images and both outputs were visually inspected.

### Observation

The short prompt usually identified the broad form that mattered—portrait,
logo, chart, collage, lab, underwater activity, property, event, or technical
work. This was useful for exposing why different embedding signals selected an
image. It also failed in consequential ways: underwater hockey became a vague
pool trick; IMG track steps became a water slide; golf/hockey and
football/lacrosse contexts were confused; TÜV testing became bicycle repair;
and visible entity branding was sometimes omitted. Separate prompting also
hallucinated a city identity.

The more explicit prompt was dramatically worse. Many responses collapsed to
“A.”, “Yes.”, single unrelated words, numbered lists, or mismatched text. This
is a direct counterexample to the assumption that more detailed instructions
necessarily produce safer semantic structure.

CPU and CUDA runs used different dtypes and matched exactly on only 20 of 70
caption strings. Most differences were paraphrases, but high-impact behavior
would need testing on the exact intended production path.

### Cost

| Run | Inference for 70 | Per image | Peak process RSS | Accelerator memory |
|---|---:|---:|---:|---:|
| Development GPU, short prompt | 31.92 s | .456 s | 3.35 GB | 2.34 GB allocated |
| CPUs 0–7, 8 threads, short prompt | 583.91 s | 8.34 s | 5.30 GB | none |
| Development GPU, prescriptive prompt | 29.04 s | .415 s | 3.35 GB | 2.34 GB allocated |

At the observed CPU rate, straight-line arithmetic is about 41.7 minutes for
300 cases, 69.5 minutes for 500, and 23.2 hours for 10,000—before all other
gallery work. The CPU path was unoptimized and the host differs from the target,
so these are not deployment predictions. They do establish an obvious obstacle
for this exact configuration.

### Scoped conclusion

Short global captions are promising as a fallible diagnostic or perhaps as
bounded two-sided review evidence after substantial cost testing. This
experiment does not establish that any VLM belongs in production, that captions
should become hard labels, or that this model/prompt is appropriate. The GPU
result is a diagnostic tool, not CPU-production evidence. Schema/range checks,
prompt regression cases, conservative handling, and visible provenance would be
necessary before a generated judgment could influence an opening.

## Production-candidate directions versus investigation tools

| Direction tested | What the evidence supports | Status for later work |
|---|---|---|
| Safe decode, dimensions, format sniffing, simple quality diagnostics | Necessary, inexpensive corpus hygiene; catches real invalid and misleading-extension cases. | **Production-candidate primitive**, but metrics must not become unreviewed quality gates. |
| Exact hashing plus perceptual comparison | Exact hash alone is insufficient; perceptual similarity finds many derivative families. | **Production-candidate subproblem** requiring conservative boundary handling and motif separation. |
| Global MobileCLIP2/SigLIP2 embeddings and prompt contrasts | Useful for navigation, broad context, and two-sided candidate pools; visible false positives and archetype collapse remain. | **Production-candidate research direction**, not a selected model or scoring formula. |
| Same-set local sequence spreading | Visibly improves several adjacent runs without pretending to fix selection. | **Promising narrow ordering direction**; needs stronger baselines and broader regression review. |
| Embedding MMR / novelty | Can rescue useful perspectives but also rewards weak difference. | **Diagnostic/limited evidence**; unsafe as a universal objective without suitability safeguards. |
| SmolVLM global captions | Useful broad-content descriptions on many boundary cases; prompt-sensitive and fallible. | **Diagnostic/reference tool for this run**; exact CPU configuration is too slow corpus-wide and expensive even for a large pool. |
| Development GPU inference | Enabled full-corpus model comparisons and rapid visual investigation. | **Diagnostic acceleration only**; establishes no CPU compliance. |
| Manual contact-sheet inspection and notes | Grounds every visual conclusion and reveals metric/model failures. | **Investigation and validation tool**, not an unattended production selection mechanism. |

This table intentionally does not compose the directions into an architecture.
Complementary methods need not be forced into one score, and later work remains
free to reject all tested model choices.

## Implications for later design and validation

These are scoped implications, not a recipe:

1. **Protect recall in rich sources.** IMG, M Immobilier, Truro, TÜV, UL,
   Digital Domain, Boston, and Maior all contain valuable complementary material
   beyond the first diagnostic 100. Any bounded review pool should include both
   tentative selections and competitive exclusions and be audited for missing
   perspectives.
2. **Make marginal contribution visible.** Pali and CMAS are useful future
   stopping-test cases; KPMG and THEMA are useful limited-source cases; Roland
   and Tara are cases where semantic score can stay high while public suitability
   declines. Do not calibrate length from one family alone.
3. **Evaluate the opening as a set.** A strong scalar first item does not repair
   a narrow first dozen. Opening comparisons should include collective entity
   understanding, motif balance, and visible context, while allowing several
   defensible anchors.
4. **Separate selection, motif control, and ordering.** Same-set reordering can
   improve rhythm; it cannot recover omitted perspectives. Near duplicates and
   motif repetition also require different evidence.
5. **Audit model labels on counterexamples.** Portraits, groups, flags, panels,
   collages, text, synthetic work, and watermarks all have both acceptable and
   unacceptable corpus examples. Hard content bans are not supported.
6. **Treat visible entity evidence as valuable but not mandatory for every
   image.** Branded real scenes and characteristic activities are strong; generic
   contextual work can still contribute. Hidden source association should not
   rescue arbitrary portraits or travel images.
7. **Benchmark the exact CPU path early.** The embedding and VLM measurements
   differ by orders of magnitude. Pool size, view count, token count, runtime,
   model revision, quantization, and target CPU can all change the feasibility
   conclusion.
8. **Use complete visual outputs for acceptance.** Proxy scores and artifact
   validation cannot demonstrate a good gallery. Rich tails, weak tails, missed
   alternatives, and motif runs only became clear on full sheets.

## Limitations and unresolved questions

- This investigation used only the supplied 20 entities. It did not establish
  generalization to an external corpus or reserve an untouched organization-level
  holdout; all current entities informed the observations.
- It did not create human reference galleries, formal pairwise preferences, or
  inter-rater annotations. Visual judgments are direct and documented but remain
  subjective.
- No production system, final selection manifests, final contact sheets, or
  production validator were built. Temporary ranked and 100-image sheets are
  experiments only.
- No actual 10,000-image entity was run end to end, and no run occurred on the
  approximate 8-vCPU/16-GiB target. Linear projections are explicitly labeled.
- Cached model execution used offline-library settings for the CPU timing runs,
  but outbound networking was not kernel-denied. This investigation does not
  establish offline production operation.
- The VLM experiment used one global thumbnail, one model, two prompts, and no
  separate OCR or high-resolution escalation. Its negative cost result is scoped
  to that path; its positive caption examples do not establish general accuracy.
- The embedding comparison covered two model families and hand-written generic
  prompts. No learned reranker, trained aesthetic model, face/group detector,
  OCR-specific model, local-feature duplicate system, or broad model sweep was
  evaluated. Such methods should be tested only against a concrete remaining
  uncertainty.
- The ranked-band pool inherited an experimental score and fixed pHash rule.
  It can demonstrate that good material exists or that a method collapses, but
  it cannot prove the objectively best count or order.
- Roland Berger's exact usable-photo ceiling remains unresolved. The source is
  visibly report-heavy, yet some competitive event/workplace exclusions deserve
  closer comparison before declaring an exact source-limited count.
- Authenticity is often not provable from pixels alone. UL/TÜV technical work,
  Digital Domain final frames, and generic event/office scenes need summary and
  corpus context without allowing filenames or provenance to overrule the image.
- Heliservices, Lakefield, and Peddie received only light visual inspection;
  Butler received focused duplicate work. Their main patterns are documented,
  but less common failure modes may remain undiscovered.
- The current source had no nested files, symlinks, hidden directories, or
  missing summaries, so those operational requirements remain implementation
  test obligations.

The most important unresolved technical questions are therefore not “which
single score wins,” but whether a later automatic method can preserve high
recall in rich sources, make defensible marginal stopping decisions, distinguish
visible context from hidden association, and form a collectively strong opening
while meeting the exact CPU/offline envelope.

## Stopping assessment for this investigation

The evidence-based investigation stopping conditions were met:

- every current organization received a summary and visual light inspection;
- materially different rich, ordinary, limited, derivative-heavy, text-heavy,
  and special-content cases received deeper examination;
- duplicate, semantic-signal, visible-relevance, length, opening, variety,
  ordering, and approximate feasibility questions received direct comparisons;
- ordinary successes, false positives, false negatives, and counterexamples were
  visually reviewed, including complete 100-image temporary sets;
- production-candidate directions were separated from diagnostic tools;
- less-examined areas and unresolved uncertainty are explicit; and
- further experiments with the same uncalibrated rankings would mostly repeat
  understood behavior rather than materially improve the pre-implementation
  handoff.

The run stops here by design. Choosing, building, validating, and demonstrating
the reusable gallery system remains the next task.
