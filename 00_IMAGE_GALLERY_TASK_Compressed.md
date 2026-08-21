# Task: Build High-Quality Representative Image Galleries

## Objective

Build a reusable implementation that automatically selects and orders up to 100 source images per business, school, nonprofit, or other organization for a polished public-facing gallery.

The reusable implementation is the primary deliverable. The supplied galleries demonstrate its quality; they must not be manually curated results with code added afterward. Apply the same unattended implementation to every supplied entity and make it generalize to unseen organizations and visual distributions. Do not embed entity names, hand-picked files, per-entity thresholds, industry exceptions, or rules tailored to the supplied corpus.

A visitor who knows nothing about an entity should be able to understand what it is, what it does, and what it feels like primarily from its gallery. Results should feel deliberately curated by a person with strong visual taste: attractive, relevant, representative, coherent, and suitable for a public carousel. Structural correctness or high individual scores alone do not establish success.

## Input and boundaries

The source root contains one directory per entity, normally with a local `summary.md` and candidate images in arbitrary subdirectories. Use the summary and source images as context. The summary is not a mandatory category checklist, and the source may reveal useful perspectives not stated in it. Do not research entities on the web or use external entity knowledge for selection.

Treat the source corpus as immutable. Never alter, rename, reorganize, or overwrite it. Gallery work is limited to selection and ordering; do not crop, enhance, regenerate, recolor, or retouch selected images. Copies must remain byte-for-byte faithful to their sources, and their original paths must remain recoverable from clear filenames or another simple, inspectable convention.

Expect messy scraped input: nested files, misleading extensions, corrupt or non-image files, tiny assets, derivative sizes, opaque or repetitive paths, missing metadata, hidden content, and unsafe links. Discover valid images by safe decoding rather than extension alone, contain paths within the source, record material failures in the final report, skip bad files, and continue unrelated entities. Filenames and metadata may offer clues but must not overrule what the pixels show.

If a required summary is missing or unreadable, fail that entity clearly without blocking other entities. Production selection must work locally and offline after one-time setup.

## Gallery length

Select at most 100 images per entity. One hundred is a cap, not an automatic quota, but a fuller gallery close to 100 is preferred whenever worthwhile material exists. Do not shrink a useful 80- or 100-image gallery merely to raise its average score. Later images may be somewhat weaker than the opening while still adding attractive, relevant, nonredundant breadth.

Stop according to the marginal value of the best remaining images. An addition should still be publicly presentable, visibly related to the organization or its world, aesthetically or informationally useful, and sufficiently distinct. Do not pad with unrelated stock, flags or landmarks used only as geographic proxies, repetitive motifs, arbitrary portraits, dense text, or poor assets.

A short or empty gallery can be correct when the source truly lacks suitable material. It is not justified by the selected set itself, a low eligible count, or an automated label. Before declaring a source limited, inspect enough of the source and competitive rejected material to distinguish genuine scarcity from weak recall, excessive rejection, duplicate or motif overcontrol, a stopping error, or poor ordering.

Avoid the opposite overcorrection: removing weak geographic, promotional, or topical material must not collapse a rich source into a tiny one-note gallery. A relevant photograph need not explain the entire organization by itself. Doctors discussing, people working in an appropriate setting, characteristic machinery, training, facilities, events, products, or other source-supported context can each contribute one part of the overall story.

## Visual curation

Judge the complete gallery rather than independent images.

### Relevance and representative breadth

Judge visible content, not hidden page association. Prefer authentic organizational evidence—characteristic activities, work, products, services, creative output, people in context, facilities, events, ceremonies, or readable identity within a real scene—over material connected only through filenames, article topics, or broad industry similarity.

Generic-looking imagery is not automatically bad. It can be useful when it is attractive, clearly appropriate to the entity's work or world, and contributes to a balanced gallery. No single image must summarize the entire organization. At the same time, generic or topical imagery must not dominate when stronger, more distinctive source material exists.

Represent the important perspectives that the source can show without imposing a universal category checklist. Avoid collapsing onto one easy archetype—only buildings, exteriors, group portraits, laboratory scenes, products, or finished creative work—when attractive complementary views exist. A slightly weaker but still pleasant image can be valuable when it adds a meaningful perspective. Novelty does not rescue an ugly or irrelevant image.

### Photographs, text, and unusual formats

Prefer strong photographs for a public carousel. Text-heavy screenshots, report pages, ordinary charts, slide-like graphics, QR codes, document covers, and promotional panels should normally be avoided unless an item is exceptionally attractive and genuinely worth showing. A small amount of tasteful non-photographic work can be appropriate when it represents the entity's real creative output and remains visually compelling.

Very wide or tall images are less pleasant in a carousel and should be disfavored, but aspect ratio is a preference rather than a hard ban. A highly relevant and attractive rectangular image may still deserve inclusion.

Watermarks, overlays, frames, and minor imperfections are graded disadvantages, not automatic rejection rules. Reject corrupt, severely blurred, badly cropped, misleading, unsafe, or plainly unpleasant images. A modest unavoidable watermark can be acceptable when the source is consistently watermarked and the photograph remains strong.

### People and portraits

People often communicate work, learning, service, sport, culture, and community. Prefer people shown in meaningful activity or organizational context. Arbitrary isolated portraits and selfies are usually weak, especially near the beginning. Relevant, pleasant portraits can be included, but too many portraits or overly prominent portrait ranks can make the gallery less attractive and less explanatory.

Posed groups may legitimately be common in a source such as a school. Some repetition is acceptable when the photographs remain relevant and the source does not offer equally good alternatives. Control visible monotony without treating every image of the same group or subject as interchangeable.

### Duplicates and motif repetition

Remove exact duplicates and suppress human-visible near duplicates such as resizes, recompressions, small crops, overlays, or nearly identical moments, keeping the better version. Do not remove genuinely different compositions, actions, or viewpoints merely because they share a subject.

Also control motif-level repetition across technically distinct images. Flags, landmarks, portraits, group poses, panels, logos, buildings, products, listings, or scene templates can overwhelm a gallery even when no pair is a near duplicate. Automated labels are not proof; inspect the contact sheet and ask whether each additional image adds enough new visual or aesthetic value.

### Opening and order

Treat the first several images and roughly the first dozen as a collective introduction. The first image should be strong, but do not sacrifice the gallery in search of one perfect hero. The opening should quickly establish identity and atmosphere through attractive, varied, representative images. It should not be dominated by one facility type, portrait, group, panel, flag, product angle, or dark scene when better alternatives exist.

Selection and ordering are different responsibilities even if the implementation handles them together. Selection must provide the right content and breadth; ordering cannot recover missing images. Arrange the selected set into a coherent carousel with strong images early, sensible changes of perspective, few near-identical neighbors, and a useful middle and tail. Do not merely sort one score or optimize smooth transitions at the expense of relevance and beauty.

## Implementation freedom and production requirements

The executor owns the implementation choices. Do not assume or require a particular model, model size, architecture, runtime, captioning approach, scoring scheme, shortlist size, or pipeline layout. Evaluate methods by the galleries and production behavior they actually produce.

Investigate promising approaches at the level of the complete pipeline. Do not reject a method merely because a naive version applied independently to every candidate would be too slow; also do not retain an expensive component without demonstrating that it improves complete galleries. Use measurements and visual comparisons to choose an approach while preserving freedom to find a better one.

Use safely available acceleration productively during investigation, experimentation, tuning, and broad regression review. It is wasteful to force all development work onto CPU when a GPU can accelerate iteration. Development acceleration does not prove production feasibility.

After one-time setup, the intended-quality production implementation must run on CPU alone on approximately 8 vCPUs and 16 GiB RAM. It must not silently disable quality-critical work, switch to a materially weaker curator, or produce a different low-resource quality tier. Optional acceleration may improve speed, but CPU must remain the authoritative deployable path.

A clean offline run for one entity containing up to about 10,000 candidates should finish in roughly one hour on that production class. A small overrun can be acceptable for an unusually difficult source; multi-hour execution is not. Measure the complete work needed to discover and decode images, perform all intended analysis, select and order, materialize the gallery, render the contact sheet, and write the final report. Distinguish direct measurements from projections and state the actual CPU, affinity, memory limit or available memory, peak memory, GPU visibility/use, cache state, and wall time concisely in the final report.

Development and setup may download ordinary dependencies or implementation assets. Provide a reproducible setup and one documented unattended run command. After setup, selection must require no internet, API, credentials, uploads, or external entity lookup. Verify a complete nontrivial offline run with outbound access blocked as strongly as the environment permits.

Reruns must be predictable and safe. Resolve existing output before expensive work, do not leave partial results looking complete, do not destroy a prior successful gallery after failure or interruption, and do not reuse stale analysis after source, summary, implementation, or material configuration changes. The implementation details are up to the executor; the behavior is what matters.

## Required deliverables

Keep the user-facing output simple and practical:

1. the reusable pipeline code with concise setup and run instructions;
2. one complete ordered final gallery for every successful entity;
3. one legible complete contact sheet for every gallery, with all contact sheets collected in one clearly named subfolder; and
4. one final report.

The final report should concisely state the method at a high level, how to run it, selected counts, failures or skipped inputs that matter, visual strengths and weaknesses, source-limited cases, performance/resource/offline evidence, and remaining uncertainty.

Contact sheets must show every selected image exactly once in gallery order and be large enough to judge the first image, opening dozen, middle, tail, relevance, attractiveness, breadth, repetition, and sequence. Labels should make order and source correspondence clear without obscuring the images.

Keep temporary comparisons, investigation sheets, caches, virtual environments, downloaded development residue, alternate galleries, and ad hoc scripts outside the final deliverable.

## Evaluation and acceptance

Inspect every final complete contact sheet honestly. Review every entity's count, opening, first dozen, middle, tail, visible relevance, attractiveness, breadth, near duplicates, motif repetition, ordering, and stopping decision. Review strong cases as seriously as weak ones so corrections do not damage behavior that already works.

For short, weak, narrow, or motif-dominated galleries, compare selected images with competitive exclusions and source material. For full galleries, inspect the tail and confirm that reaching 100 remains worthwhile. After material changes, rerun a broad mix of easy, difficult, previously strong, and previously weak sources with the same general implementation.

Completion requires:

- the unchanged reusable implementation produces all galleries without per-entity intervention;
- strong-source galleries are attractive, visibly relevant, representative, and generally full when worthwhile material exists;
- weak sources are handled honestly rather than padded or overclaimed;
- openings work as collective introductions and do not bury the most representative photographs;
- useful breadth is preserved without geographic, portrait, group, panel, building, product, or other motif domination;
- exact and near duplicates are suppressed without deleting genuinely different views;
- text-heavy and awkward formats are rare and justified by exceptional value;
- source bytes remain unchanged and final galleries/contact sheets are complete and consistent;
- messy inputs and entity failures are isolated;
- setup and reruns are reproducible and safe;
- the same intended-quality method meets the CPU-only, offline, memory, and runtime objectives without silent weakening; and
- the final deliverable contains only the pipeline, final galleries, grouped contact sheets, and final report needed by the user.
