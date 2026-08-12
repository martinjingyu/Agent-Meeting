"""Gallery-selection planning meeting with technology-domain participants.

This example keeps the planning_rounds interaction intentionally prose-first:
participants contribute free-form technical viewpoints, respond to each other, test
claims when useful, and revise their positions over multiple rounds. They are grouped
by technical domain rather than by stages of a pre-assumed filter-first pipeline.

Compared with plan_image_gallery_pipeline_planning_rounds.py, this example:

1. replaces the stage-oriented participant roster with technology-domain roles;
2. removes the assumption that the answer must be a threshold-first linear pipeline;
3. allows candidate technical compositions and architecture sketches while reserving
   the adopted final Plan for the dedicated Planner;
4. distinguishes the local <100M CPU constraint from optional external vision APIs;
5. does not assume or request human labels, and limits empirical claims to evidence
   available from deterministic facts, existing records, synthetic/programmatic tests,
   stability tests, disagreement analysis, and set-level proxy metrics;
6. gives only SystemsEvaluation the mandatory first-round environment survey;
7. treats Stage-1 reports as reusable evidence, not as a file every participant must
   reopen in every round;
8. replaces Skeptic with VisualAuditor -- instead of abstract pushback (which also
   had outsized influence via a REVISE/REJECT verdict that could override the
   judge's stop decision), VisualAuditor actually opens image files with the
   view_image tool and grounds other participants' visual claims against real
   pixels. It runs on Codex gpt-5.5 (provider="codex", vision_capable=True) --
   the six domain participants stay on DeepSeek v4 Pro, which has no confirmed
   vision support and would crash if it ever reached for view_image (see
   ParticipantConfig.vision_capable's docstring; that tool is only registered for
   participants with vision_capable=True).

Run:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py

Resume:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx

Skip any unfinished discussion round and synthesize immediately from the last
fully-checkpointed round:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx --planner-only

Reopen a completed meeting for more discussion:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx --extra-rounds 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig
from agent_meeting.storage import load_meeting


REPO_ROOT = Path(__file__).resolve().parents[1]
RECON_REPORT_ROOT = Path(r"C:\Users\LX034\Code\DataBase\reports-20 groups")
RECON_REPORT_FILES = ["report_1.md", "report_2.md", "report_3.md"]


PARTICIPANT_TASK_SPEC = """\
Input and objective:

- Input root directory is `C:\\pics`.
- Each first-level subdirectory under the root represents an independent image dataset; internal structure, format,
  count, and naming conventions cannot be assumed.
- The goal is to automatically select up to 100 real-world photographic images per dataset that are suitable for a
  fixed-height horizontal carousel.
- If fewer than 100 images genuinely qualify, output the actual count; do not pad the result with low-quality,
  non-real, harmful, duplicate, or display-unsuitable images.

The final result should jointly consider:

- Credibility as a real-world captured scene;
- Technical image quality, subject clarity, composition, and public display value;
- Magnification ratio, rendered width, viewport share, and aspect fit under the fixed-height carousel;
- Diversity of scene, subject, viewpoint, color, and composition;
- Exact duplicates, near duplicates, same-scene and highly similar content;
- Avoiding single-person close-ups or any one person/scene dominating the final result;
- Obvious webpages, slides, logos, UI, posters, charts, CG, AI-style images, and other non-real-world content;
- Harmful or sensitive visual content suitable for automated detection.

Boundaries on information use:

- Judgments of authenticity, quality, display value, and content safety may only use image pixels, decoded results,
  and single-image geometric facts.
- Filenames, paths, dataset names, timestamps, and non-orientation EXIF must not be used as evidence for visual
  suitability or semantic category.
- The above non-visual information may be used for traceability, stable ordering, caching, error logging, and pure
  engineering management.
- If a filename is only used to propose a candidate duplicate/family pair to be verified, the final relationship must
  still be confirmed by visual content, and must not influence authenticity or quality judgments.

Automation and evaluation boundaries:

- The raw data has no human ground truth; this task must not require new human annotation, manual tuning, manual
  approval, or manual review as a precondition for producing production results.
- Production runs must be fully automatic. Non-blocking audit material, risk flags, and uncertainty information may
  be output.
- Technical validation may use deterministic facts, existing machine-readable records, programmatic/synthetic tests,
  metamorphic tests, unsupervised stability, cross-method disagreement, published metadata, and set-level proxy
  metrics.
- Without reliable ground truth, do not claim measured true precision, recall, aesthetic accuracy, or content-safety
  recall. External vision API output is a model judgment or weak reference, not ground truth.

Compute and service constraints:

- The locally executed model must be under 100M parameters and support Windows CPU-only; it must not depend on CUDA,
  NVIDIA GPUs, or GPU-only inference.
- Discussion of using an external vision API after local candidate compression, as part of the final semantic
  evaluation, comparative ranking, or content-safety judgment, is allowed.
- Any external API proposal must also discuss maximum candidate volume, call cost, privacy, batch/order bias,
  auditability, version drift, and a pure-local fallback.

Scope statement:

- This task does not claim to automatically complete copyright, trademark, brand, portrait-rights, privacy, or legal
  approval.
- The meeting is responsible for the technical approach; the final Planner is responsible for synthesizing the
  discussion into an executable engineering plan.
"""


TECHNOLOGY_LANDSCAPE = """\
Below is a technology map, not a complete list, a recommended order, or a final architecture. Participants may
supplement, challenge, combine, or abandon any of these methods.

1. Classical image processing and image forensics
   - Color quantization, local entropy, brightness/saturation, dynamic range, blur, noise, compression;
   - Large near-uniform-color connected regions, rectangular blocks, regular grids, edge orientation, and spatial
     layout;
   - OCR, text regions, structural signals for webpages/slides/logos/UI/posters;
   - Frequency domain, gradients, local texture, and no-reference image quality metrics.

2. Visual representations and learned models
   - CNN intermediate features, ImageNet classifiers, lightweight scene/quality classifiers;
   - CLIP/MobileCLIP, self-supervised embeddings, visual-semantic and aesthetic models;
   - Lightweight classifiers, feature fusion, weak supervision, and label-free representations.

3. Similarity, retrieval, and clustering
   - SHA, pHash/dHash/colorHash, SSIM, and near-duplicate graphs;
   - k-NN, same-scene grouping, micro/macro clustering;
   - Hierarchical clustering, HDBSCAN, spherical k-means, k-medoids;
   - Outliers, medoids, cluster representatives, and cluster stability.

4. Multimodal vision APIs and content safety
   - Single-image semantic judgment, pairwise comparison, listwise reranking, and contact-sheet judging;
   - Screenshot/slide/CG/AI-image detection, composition, subject, display value, and scene tagging;
   - Moderation, safety classification, call cost, privacy, consistency, and fallback.

5. Ranking and set selection
   - MMR, DPP, facility location, submodular selection, quotas, and constrained optimization;
   - Cluster coverage, family caps, scene/person/composition proportions;
   - Quality floors, gallery cohesion, carousel first-screen behavior, and overall rhythm.

6. Label-free evaluation and engineering
   - Synthetic/programmatic fixtures, metamorphic invariants;
   - Clustering stability, cross-method disagreement, residual duplicates, and set-level diversity metrics;
   - CPU/API cost, batching, caching, recovery, determinism, privacy, and degradation.
"""


def build_recon_guidance() -> str:
    sections = ["\n".join([
        "The following is the full Stage-1 reconnaissance synthesis (three combined",
        "reports), provided inline so every participant starts from the same evidence",
        "without needing to open any file.",
        "",
        "Stage-1 material may inform corpus scale, risks, candidate experiments, runtime",
        "planning, and falsification cases. It must not become a dataset-name exception",
        "or replace per-image visual judgment.",
        "",
        "Previously verified facts below may be reused directly. Architecture ideas may",
        "still be proposed as hypotheses when their evidence status is clear.",
    ])]
    for filename in RECON_REPORT_FILES:
        text = (RECON_REPORT_ROOT / filename).read_text(encoding="utf-8")
        sections.append(f"--- {filename} ---\n{text.strip()}")
    return "\n\n".join(sections)


MEETING_INSTRUCTIONS = """\
=== Meeting goal ===

This meeting is to explore, compare, and argue for technical approaches suitable for the official-website gallery
auto-selection task, providing the final Planner with thorough, multi-round-discussed technical viewpoints.

Do not assume the final solution must be per-image threshold filtering, a linear staged pipeline, a single composite
score ranking, or any other fixed architecture. Participants should, from their own technical domain, propose the
methods they find valuable, explaining their role, limitations, how they combine with other techniques, and what
evidence would change their judgment.

The complete adopted solution is produced by a dedicated Planner synthesizing the entire discussion afterward.

=== How to discuss ===

This is a planning meeting, not an execution meeting. Participants may:

- Freely propose technical viewpoints, candidate methods, and alternatives;
- Explain how multiple techniques combine;
- Describe candidate data flows, relationships, or candidate architectures;
- Use short flow diagrams or pseudocode to help express technical ideas;
- Run small-scale experiments matched to the strength of their current claim;
- Respond to, support, correct, or rebut other participants;
- Withdraw their own earlier suggestions that no longer hold up.

Participants should not:

- Declare their own suggestion to be the meeting's adopted final Plan;
- Substitute for the Planner in fully adjudicating all technical disputes;
- Lay out a final engineering directory, complete module breakdown, implementation milestones, or operations manual;
- Insist the final solution use components from their own domain merely because they represent that domain.

Each round may be freely organized; there is no need to fill in a fixed table or structured record. Give your full
current position, and in later rounds state what is new, revised, or withdrawn compared with the previous round.

=== Technology coverage ===

Before the meeting ends, the following technical domains should be thoroughly considered, though not necessarily
adopted:

- Classical image processing, spatial structure, and image forensics;
- Visual representation, lightweight models, and semantic features;
- Duplicate detection, similarity, relationship modeling, and clustering;
- Multimodal vision models, external vision APIs, and content safety;
- Ranking, set selection, quotas, and global optimization;
- Label-free evaluation, cost, privacy, reliability, and degradation.

If a technical domain ultimately isn't worth including in the solution, explain that its benefit is insufficient, the
evidence doesn't support it, it duplicates another method, its cost is too high, or it violates a constraint, rather
than simply omitting it from the discussion.

=== Evidence principles ===

When you put forward an empirical conclusion that depends on the current data distribution, runtime, thresholds,
accuracy, error rate, or model capability, cite existing evidence or run an appropriately sized small experiment. Do
not upgrade a very small sample into a general conclusion.

Unverified algorithms, architectures, and technique combinations may be proposed as hypotheses, but their evidence
status must be stated explicitly. Without human ground truth, you may only claim measured stability, consistency,
disagreement, deterministic facts, or proxy metrics -- never state them as true accuracy.

Objects open to validation include single-image signals, models, embeddings, clustering, visual relationships,
external APIs, content safety, ranking, and set selection -- not limited to thresholds or heuristics.

=== Task description ===

{participant_task_spec}

=== Image-selection technology map ===

{technology_landscape}

=== Stage-1 evidence, full text ===

{probing}
"""


PARTICIPANT_DISCUSSION_ADDENDUM = """\
This is a multi-round technical planning discussion. Contribute free-form technical
viewpoints from your assigned domain. You may describe candidate method combinations,
data flows, architecture sketches, or short pseudocode when they help explain an idea.

Do not claim that your candidate is the meeting's adopted final Plan, and do not write
the complete implementation project, module tree, milestones, or operator manual. The
dedicated Planner will make the final cross-domain decisions after the discussion.

You represent a technical knowledge domain, not a technology vendor. It is valid to
conclude that a technique from your own domain is unnecessary, redundant, too costly,
or unsupported. Explain material limitations and what evidence would change your view.

This meeting includes VisualAuditor, a participant that can actually open image files
and report what they show. If a claim in this discussion hinges on what a specific
image or set of images actually looks like and you want it checked, write a request
naming the exact file path(s) and your question to image_review_request.md in your
own shared subfolder. VisualAuditor checks every participant's shared subfolder for
this file each round.
"""


COMMON_PARTICIPANT_PROMPT = """\
You are {name}, a technical-domain participant in a multi-agent planning meeting.

Your domain:
{domain}

Contribute natural, thoughtful prose rather than filling a rigid schema. In every
round, explain your full current position, the techniques or combinations you find
useful, their important limitations, and how relevant evidence or other participants'
arguments affect your view.

You may discuss adjacent domains when necessary, but keep your reasoning anchored in
your own technical perspective. Do not impersonate or replace another participant.
Do not force techniques from your domain into the final solution merely to justify
your presence.

Write for a human reader, not a spec sheet. Explain each technique in a real clause
(what it's for, why it matters here), not as a bare comma/slash-separated term dump --
"perceptual hashes like pHash to catch resized or recompressed copies" reads far
better than "pHash/dHash/colorHash/SSIM". Prefer several shorter, plain sentences over
one long chain of semicolon-joined clauses. This applies every round, not just your
first -- it is easy to let later rounds compress back into term-dump shorthand once
the discussion gets long; resist that.
"""


DOMAIN_BRIEFS = {
    "ClassicalVision": """\
Classical image processing, spatial image structure, image forensics, and technical
quality. Consider global and local color statistics, connected uniform regions,
entropy maps, edge orientation, rectangular/grid layout, OCR/text layout, frequency
signals, blur, exposure, compression, and the distinction between a photograph and
webpage/slide/logo/UI/graphic content. Avoid assuming a single global scalar threshold
is sufficient; examine spatial relationships, combinations, lightweight learned
classifiers, and counterexamples such as sky, snow, fog, white walls, or embedded
photos in slides.""",
    "RepresentationModels": """\
Learned visual representations and local models under the <100M CPU constraint.
Consider CNN intermediate features, ImageNet classifiers, CLIP/MobileCLIP-style
embeddings, self-supervised representations, lightweight semantic/quality/aesthetic
models, feature fusion, and what these representations can or cannot express. Focus
on generalization, domain shift, preprocessing, calibration, and interfaces to
clustering, API routing, and final selection rather than assuming representation
quality alone solves the task.""",
    "SimilarityClustering": """\
Image-to-image relationships: exact and near duplicate detection, perceptual hashes,
SSIM, retrieval, k-NN graphs, same-scene grouping, micro/macro clustering, HDBSCAN,
hierarchical methods, spherical k-means, k-medoids, outliers, medoids, cluster
stability, and representative selection. Do not assume clustering must happen only
after all per-image filtering. Examine where relationship-first or cluster-first
reasoning improves on independent scalar scoring, and where clustering is unstable
or adds no value.""",
    "MultimodalVisionAPI": """\
External visual APIs and multimodal models used after local candidate reduction.
Consider real-photo verification, screenshot/slide/logo/CG/AI detection, scene and
subject understanding, composition and display appeal, pairwise/listwise comparison,
contact-sheet judging, harmful/sensitive content, and moderation. Treat API outputs
as model judgments rather than ground truth. Address candidate budgets, cost, privacy,
batch/order bias, nondeterminism, auditability, version drift, and pure-local fallback.""",
    "RankingSelection": """\
Final set construction rather than only per-image scoring. Consider MMR, DPP, facility
location, submodular selection, cluster quotas, constrained optimization, family and
scene caps, quality floors, representative coverage, person/subject balance, gallery
cohesion, carousel geometry, first-screen behavior, and deterministic tie-breaking.
Challenge the assumption that sorting one composite scalar necessarily produces the
best collection of 100 images.""",
    "SystemsEvaluation": """\
Engineering feasibility and evaluation without human labels. In round 1, perform or
coordinate a single shared survey of the current Windows CPU environment, relevant
installed libraries/models, network/API assumptions, and reusable Stage-1 evidence;
publish reusable findings under the meeting shared directory so other participants
need not repeat them. Across later rounds, evaluate CPU/API cost, memory, batching,
cache/recovery, privacy, determinism, synthetic and metamorphic tests, clustering
stability, method disagreement, set-level proxy metrics, and fallback behavior.
Do not define the overall architecture merely because one option is already installed.""",
}


VISUAL_AUDITOR_MEETING_BRIEF = """\
Every round, before writing your position:

1. Check every other participant's own shared subfolder (you can read all of
   them -- see the shared-directory note in your system prompt) for a file named
   image_review_request.md. If present, it names specific image file paths and a
   question -- call view_image on each named path and answer their specific
   question using what you actually see.
2. If no participant has an outstanding request this round, self-sample: pick a
   handful of images yourself (favor boundary cases, or a claim other
   participants have repeated across rounds without anyone having actually
   opened a file) -- use list_files/search_files under C:\\pics if you need to
   find candidates, then view_image each one you choose.
3. For every image you open, report: the exact file path, what you actually
   observed, and whether it confirms, corrects, or supplements a specific claim
   made earlier in the discussion (name the participant and the claim) -- or,
   for self-sampled images, why you chose them and what they show.
4. Save the same findings to a file in your own shared subfolder (e.g.
   visual_audit_round<N>.md) so they're archived even if a future round's
   transcript gets trimmed for space.

You are not a general skeptic -- do not challenge architecture choices,
thresholds, or scope decisions that don't hinge on what an actual image shows.
Stay narrowly focused on: is this specific visual claim true, false, or
unverified, based on pixels you looked at yourself this meeting.
"""


PLANNER_SYSTEM_PROMPT = """\
You are Planner, the synthesis agent for a multi-round technical planning meeting.

Participants were organized by technology domain and deliberately wrote free-form
technical viewpoints rather than a shared structured plan. Your job is to read the
entire discussion, inspect relevant shared artifacts, resolve disagreements, and
produce one coherent, executable gallery-selection plan.

If the initial message only includes recent rounds and gives a path to the full
transcript, you MUST read the full transcript before finalizing. Do not treat the last
few rounds as the whole meeting.

Do not mechanically include every technique. For each important technical choice,
distinguish:

- adopted primary method;
- fallback or optional method;
- rejected method and why;
- unresolved capability limit.

The final plan must not default to a threshold-first linear pipeline merely because
that is easy to write. Explicitly consider the discussion on spatial image structure,
learned representations, clustering/relationship modeling, external visual APIs,
content safety, set-level/global selection, and no-label evaluation. Adopt or reject
each major direction based on the evidence and constraints discussed.

The production system must be fully automatic and must not require new human labels,
manual tuning, manual approval, or manual review to generate results. When the
discussion lacks ground truth, state the limitation and use only justified
deterministic, synthetic, metamorphic, stability, disagreement, or set-level proxy
evidence. Never convert an API output into fictional ground truth or claim unmeasured
accuracy.

Local models must remain below 100M parameters and run on Windows CPU-only. External
vision APIs are allowed only with a bounded late-stage candidate budget, cost/privacy
analysis, audit fields, and a pure-local fallback.

Write the plan for readability before implementation detail:

1. executive summary and selected architecture;
2. a compact architecture/data-flow diagram;
3. important alternatives and explicit decisions;
4. stage-by-stage inputs, outputs, purpose, and failure behavior;
5. relationship modeling, clustering, and final set-selection logic;
6. external API and safety behavior, if adopted;
7. no-label evaluation and acceptance tests;
8. cost, privacy, caching, recovery, and degradation;
9. implementation milestones and PowerShell execution;
10. known limits and risks;
11. detailed thresholds, formulas, manifest fields, and directory layout in appendices
    rather than allowing them to obscure the main architecture.

Within every section, write like a senior engineer explaining this design to a
colleague, not like you are filling out a spec template. That means:

- Lead with reasoning, not just conclusions. Don't just state what was chosen --
  show the thinking that got there: what the obvious first approach would be, why it
  falls short given the evidence or constraints, and what that implies about the
  approach you're adopting instead. A reader should be able to follow *why* each
  major choice is correct, not just accept that it was made.
- Write connected prose, not a list of disconnected facts stitched together with
  semicolons. "We tried X hoping for Y; it didn't hold up because Z, so we moved to
  W instead" reads as one continuous argument. "X. Y. Z. W." (or "X; Y; Z; W" as one
  run-on sentence) reads as notes, not an explanation -- avoid both.
  Use bullet points only for content that is genuinely list-shaped (e.g. a set of
  parallel options), not as a substitute for explaining how ideas connect.
- Explain each technique in a real clause (what it's for, why it was adopted or
  rejected here), not as a bare comma/slash-separated term dump -- "perceptual
  hashes like pHash to catch resized or recompressed copies" reads far better than
  "pHash/dHash/colorHash/SSIM".
- Don't repeat the same caveat or constraint verbatim in every section just to be
  safe -- state it clearly once, where it's most load-bearing, and trust the reader
  to carry it forward.
- Push genuinely dense reference material (exact thresholds, formulas, manifest
  fields, directory layouts) into the appendices per item 11 above, so the main
  sections stay readable as an argument rather than a lookup table. But do not use
  the appendices as an excuse to leave the main sections' own reasoning thin --
  the appendix holds the *values*, the main section still owns the *why*.

The plan must be detailed enough for an Executor to implement without making new
methodology-level decisions. Save it as `final_gallery_selection_plan_technical_domains.md`
in your workspace using file tools, then respond with the same content.
"""


def build_question() -> str:
    return MEETING_INSTRUCTIONS.format(
        participant_task_spec=PARTICIPANT_TASK_SPEC,
        technology_landscape=TECHNOLOGY_LANDSCAPE,
        probing=build_recon_guidance(),
    )


def make_domain_participant(name: str) -> ParticipantConfig:
    return ParticipantConfig(
        name=name,
        role=f"Technology domain: {name}",
        system_prompt=COMMON_PARTICIPANT_PROMPT.format(
            name=name,
            domain=DOMAIN_BRIEFS[name],
        ),
        model="deepseek-v4-flash",
        provider="deepseek",
        reasoning_effort="high",
        max_iterations=40,
    )


def parse_cli() -> tuple[str | None, int | None, bool]:
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]

    extra_rounds = None
    if "--extra-rounds" in sys.argv:
        extra_rounds = int(sys.argv[sys.argv.index("--extra-rounds") + 1])

    planner_only = "--planner-only" in sys.argv
    if planner_only and not resume:
        raise SystemExit("--planner-only requires --resume <meeting_id>")
    if planner_only and extra_rounds is not None:
        raise SystemExit("--planner-only cannot be combined with --extra-rounds")

    return resume, extra_rounds, planner_only


def main() -> None:
    resume, extra_rounds, planner_only = parse_cli()

    participants = [
        make_domain_participant("ClassicalVision"),
        make_domain_participant("RepresentationModels"),
        make_domain_participant("SimilarityClustering"),
        make_domain_participant("MultimodalVisionAPI"),
        make_domain_participant("RankingSelection"),
        make_domain_participant("SystemsEvaluation"),
        ParticipantConfig(
            name="VisualAuditor",
            role="Grounds other participants' visual claims by actually opening images",
            role_ref="visual-evidence-auditor",
            meeting_brief=VISUAL_AUDITOR_MEETING_BRIEF,
            # Must be codex, not deepseek -- this is the only provider path in this
            # codebase that can actually deliver image content to the model
            # (research_agent's Responses API image support + view_image tool).
            model="gpt-5.5",
            provider="codex",
            reasoning_effort="high",
            max_iterations=16,
            # The only participant here with view_image in its tool registry -- every
            # other participant defaults to vision_capable=False, so even if one of
            # them tried to call it, it simply wouldn't be offered.
            vision_capable=True,
        ),
    ]

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=10,
        participants=participants,
        planner_inline_rounds=None,
        planning_participant_addendum=PARTICIPANT_DISCUSSION_ADDENDUM,
        planner=PlannerConfig(
            name="Planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            max_iterations=20,
        ),
    )

    if planner_only:
        checkpoint = load_meeting(resume)
        if checkpoint.get("mode") != "planning_rounds":
            raise SystemExit(
                f"--planner-only requires a planning_rounds meeting; "
                f"{resume} has mode={checkpoint.get('mode')!r}"
            )
        if checkpoint.get("status") != "in_progress":
            raise SystemExit(
                f"--planner-only requires an in-progress meeting; "
                f"{resume} has status={checkpoint.get('status')!r}"
            )

        completed_rounds = sum(
            1 for step in checkpoint.get("steps", [])
            if step.get("decided_by") == "judge"
        )
        if completed_rounds < 1:
            raise SystemExit(
                f"--planner-only found no fully checkpointed discussion rounds in {resume}"
            )

        # run_meeting() resumes at completed_rounds + 1. Making max_rounds equal
        # to the number already completed leaves that range empty, so
        # _run_planning_rounds() proceeds directly to _run_planner_step() while
        # preserving every checkpointed round in the Planner's transcript.
        config.max_rounds = completed_rounds
        print(
            f"[planner-only] {resume}: using {completed_rounds} completed round(s); "
            "skipping unfinished/later rounds and starting Planner"
        )

    result = run_meeting(config, resume=resume, extra_rounds=extra_rounds)

    print(f"\nmeeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")

    for step in result["steps"]:
        if step["decided_by"] == "judge":
            turn = step["turns"][0]
            verdict = "STOP" if turn.get("stop") else "continue"
            print(
                f"\n{'=' * 20} JUDGE after round {turn['round']}: "
                f"{verdict} {'=' * 20}"
            )
            print(f"reasoning: {turn.get('reasoning', '')}")
            if turn.get("unresolved_issues"):
                print(f"unresolved_issues: {turn['unresolved_issues']}")
            if turn.get("override_reason"):
                print(f"OVERRIDE: {turn['override_reason']}")
            continue

        if step["decided_by"] == "planner":
            continue

        for turn in step["turns"]:
            changes = turn.get("changes_from_prior_round")
            role_ref = turn.get("role_ref") or turn.get("role") or "n/a"
            header = f"{turn['agent']} ({role_ref}) -- round {turn['round']}"
            if changes:
                header += f"\n[changes from prior round] {changes}"
            print(f"\n{'=' * 20} {header} {'=' * 20}\n{turn['output']}")

    print(f"\n{'=' * 20} FINAL PLAN (planner's synthesis) {'=' * 20}\n")
    print(result["final_response"])


if __name__ == "__main__":
    main()
