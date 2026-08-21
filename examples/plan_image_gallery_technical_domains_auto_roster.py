"""Same meeting as plan_image_gallery_technical_domains_rounds.py, with exactly one
difference: the technology-domain roster (which domains exist, how many, and what
each one's brief says) is not hand-picked -- it is designed at runtime by
agent_meeting.role_architect.design_domain_roster() from the task spec and the same
three Stage-1 evidence reports, before the meeting itself starts. That's the piece
that makes this pipeline end-to-end: task + evidence in, a ready-to-run participant
roster out, then straight into the same planning_rounds meeting.

Everything else -- MEETING_INSTRUCTIONS, TECHNOLOGY_LANDSCAPE (now also handed to
the role architect as a reference menu, not a required list), the VisualAuditor
participant, the Planner, the CLI flags -- is identical to
plan_image_gallery_technical_domains_rounds.py. See that file's docstring for the
full list of differences from plan_image_gallery_pipeline_planning_rounds.py.

Run:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_auto_roster.py

Resume:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_auto_roster.py --resume mtg_xxxxxxxxxx

Skip any unfinished discussion round and synthesize immediately from the last
fully-checkpointed round:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_auto_roster.py --resume mtg_xxxxxxxxxx --planner-only

Reopen a completed meeting for more discussion:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_auto_roster.py --resume mtg_xxxxxxxxxx --extra-rounds 2

Note on --resume: run_meeting(resume=...) does NOT restore config.participants from
the checkpoint -- every round is built straight from whatever ParticipantConfig list
the caller passes in, resumed or not (see runner.py's per-round ThreadPoolExecutor
calls). So a dynamically-designed roster has to be made resumable by hand: the roster
this script designs is written to runs/<meeting_id>_domain_roster.json *before* the
first round starts (using run_meeting(meeting_id=...) to learn the id up front rather
than after a successful return, so a mid-run crash still leaves the roster file next
to the in-progress checkpoint). --resume/--extra-rounds/--planner-only load that same
file and rebuild the identical participant list from it instead of calling the role
architect again -- calling it again could return a different roster (it's an LLM
call) whose names wouldn't match the transcript/shared-subfolders already on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig
from agent_meeting.role_architect import DomainRole, design_domain_roster
from agent_meeting.storage import load_meeting, meeting_path, new_meeting_id


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "00_IMAGE_GALLERY_TASK_Compressed.md"
INVESTIGATION_REPORT_PATH = REPO_ROOT / "investigation_report_standlone.md"

# Names already spoken for by fixed (non-domain) participants -- the role architect
# doesn't know about them, so its output is checked against this set rather than
# trusting it can't collide.
_RESERVED_NAMES = {"VisualAuditor", "Planner"}


def _roster_path(meeting_id: str) -> Path:
    return meeting_path(meeting_id).with_name(f"{meeting_id}_domain_roster.json")


def _save_roster(meeting_id: str, roster: list[DomainRole]) -> None:
    payload = [{"name": r.name, "domain_brief": r.domain_brief} for r in roster]
    _roster_path(meeting_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_roster(meeting_id: str) -> list[DomainRole]:
    path = _roster_path(meeting_id)
    if not path.exists():
        raise SystemExit(
            f"no saved domain roster at {path} -- this meeting wasn't started by this "
            "script, or the roster file was removed; without it, --resume/--extra-rounds/"
            "--planner-only cannot reconstruct the same participants the transcript "
            "already references"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DomainRole(name=e["name"], domain_brief=e["domain_brief"]) for e in payload]


# Loaded from disk (rather than inlined as a string literal, as this used to be) so
# the task spec fed to the role architect and every participant matches whatever the
# repo's own task-definition file currently says, with no risk of the two drifting
# apart the way a copy-pasted duplicate eventually would.
PARTICIPANT_TASK_SPEC = TASK_SPEC_PATH.read_text(encoding="utf-8")


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
    intro = "\n".join([
        "The following is the full Stage-1 investigation report, provided inline so",
        "every participant starts from the same evidence without needing to open any",
        "file.",
        "",
        "Stage-1 material may inform corpus scale, risks, candidate experiments, runtime",
        "planning, and falsification cases. It must not become a dataset-name exception",
        "or replace per-image visual judgment.",
        "",
        "Previously verified facts below may be reused directly. Architecture ideas may",
        "still be proposed as hypotheses when their evidence status is clear.",
    ])
    text = INVESTIGATION_REPORT_PATH.read_text(encoding="utf-8")
    return f"{intro}\n\n--- {INVESTIGATION_REPORT_PATH.name} ---\n{text.strip()}"


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
You run as a single streamed response, not a tool-calling agent -- no file access,
no shared/ directory, one pass. The full discussion (every round, every
participant) is included directly in the message you receive; there is no
transcript file to go read.

Participants were organized by technology domain and deliberately wrote free-form
technical viewpoints rather than a shared structured plan. Your job is to read the
entire discussion below, resolve disagreements, and produce one coherent,
executable gallery-selection plan in a single pass.

=== The single most important constraint on this document ===

The Executor who implements this plan will read ONLY the document you produce right
now. They will never see the meeting transcript below, never see any script or file
a participant wrote during the meeting, never know this meeting happened at all.
That has concrete consequences for how you must write:

- Resolve every meeting-internal reference before it reaches the output. Do not
  write "the relation types discussed", "as argued in an earlier round", "per
  ClassicalVision's proposal", "see shared/<participant>/report.md", or any other
  phrasing that only makes sense to someone who read the meeting. State the actual
  content directly and completely, as if you derived it yourself.
- Translate ad hoc terminology into standard engineering language. Participants
  sometimes invent a shorthand name for a technique, a data structure, or a
  category that made sense in the flow of discussion but is not a term the
  Executor will recognize cold. Either replace it with the standard name for what
  it actually is, or, if no standard term fits, define it plainly on first use and
  then use that definition consistently.
- Completeness over brevity. Never defer a detail to "see the discussion" or "as
  established earlier" -- if a value, threshold, formula, or decision is needed to
  implement the plan, it must be written in this document (in the appendix if it's
  dense reference material, but written). An Executor who has only this file must
  be able to build the entire system without asking a follow-up question about
  what a term means or where a number came from.
- Do not name participants, round numbers, or the meeting process itself anywhere
  in the output. The document should read as if a single engineer designed the
  whole thing, not as a summary of who said what.

Do not mechanically include every technique. For each important technical choice,
distinguish:

- adopted primary method;
- fallback or optional method;
- rejected method and why;
- unresolved capability limit.

This "rejected method and why" duty is not optional prose coloring -- it is a
required, separately checkable deliverable. Include a dedicated section (see item
12 in the structure list below) titled "Overridden participant recommendations"
that enumerates every case where two or more participants independently converged
on a specific, evidence-backed recommendation (a measured number, a tested
threshold, a design they both arrived at from separate experiments) and your final
Plan does something different. For each entry, name the recommendation, cite the
specific measured evidence behind it (the number, not just "participants found
X"), and state in one sentence why the Plan overrides it and what that costs
(e.g. a known residual miss rate the discussion measured but the adopted approach
does not close). If you introduce an architectural principle that no participant
raised, to reject a recommendation they converged on, that principle and the
tradeoff it costs belong in this section too -- silently replacing a
measured, converged recommendation with an unstated design preference is exactly
the failure this section exists to make visible. An empty section is a valid and
expected outcome when nothing was overridden; do not pad it.

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
    rather than allowing them to obscure the main architecture;
12. overridden participant recommendations, as required above -- may be empty.

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
methodology-level decisions.

Respond with ONLY the final plan itself, formatted as Markdown, starting with a
top-level heading. No preamble, no meta-commentary about the meeting or your own
process.
"""


def build_question(probing: str) -> str:
    return MEETING_INSTRUCTIONS.format(
        participant_task_spec=PARTICIPANT_TASK_SPEC,
        technology_landscape=TECHNOLOGY_LANDSCAPE,
        probing=probing,
    )


def make_domain_participant(role: DomainRole) -> ParticipantConfig:
    return ParticipantConfig(
        name=role.name,
        role=f"Technology domain: {role.name}",
        system_prompt=COMMON_PARTICIPANT_PROMPT.format(
            name=role.name,
            domain=role.domain_brief,
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


def _build_participants(roster: list[DomainRole]) -> list[ParticipantConfig]:
    participants = [make_domain_participant(role) for role in roster]
    participants.append(
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
            # Was 16 -- far below every other participant's 40, despite VisualAuditor
            # having the heaviest per-round workload (it's the sink for every other
            # participant's image_review_request.md). Every meeting we've run this on
            # showed VisualAuditor's request backlog never clearing across 6 rounds;
            # raised to parity so it isn't structurally unable to keep up.
            max_iterations=40,
            # The only participant here with view_image in its tool registry -- every
            # other participant defaults to vision_capable=False, so even if one of
            # them tried to call it, it simply wouldn't be offered.
            vision_capable=True,
        )
    )
    return participants


def main() -> None:
    resume, extra_rounds, planner_only = parse_cli()
    probing = build_recon_guidance()

    if resume:
        meeting_id = resume
        roster = _load_roster(meeting_id)
        print(f"[role-architect] reusing the {len(roster)}-domain roster saved for {meeting_id}:")
        for role in roster:
            print(f"  - {role.name}")
    else:
        meeting_id = new_meeting_id()
        print("[role-architect] designing the technology-domain roster from the task spec + evidence...")
        roster = design_domain_roster(
            task_spec=PARTICIPANT_TASK_SPEC,
            meeting_id=meeting_id,
            technology_reference=TECHNOLOGY_LANDSCAPE,
            evidence_paths=[INVESTIGATION_REPORT_PATH],
        )
        for role in roster:
            if role.name in _RESERVED_NAMES:
                raise SystemExit(
                    f"role architect chose {role.name!r}, which collides with a fixed "
                    "participant name (VisualAuditor/Planner) -- rerun to get a fresh roster"
                )
        print(f"[role-architect] designed {len(roster)} domain(s):")
        for role in roster:
            print(f"  - {role.name}: {role.domain_brief.splitlines()[0][:100]}...")
        # Written before run_meeting() is even called -- see the module docstring's
        # note on --resume for why this can't wait until run_meeting() returns.
        _save_roster(meeting_id, roster)

    config = MeetingConfig(
        question=build_question(probing),
        mode="planning_rounds",
        max_rounds=6,
        participants=_build_participants(roster),
        planner_inline_rounds=None,
        planning_participant_addendum=PARTICIPANT_DISCUSSION_ADDENDUM,
        planner=PlannerConfig(
            name="Planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            # Single streamed LLM call instead of an agentic tool loop -- see
            # agent_meeting.single_call_synthesis.stream_synthesize and
            # PlannerConfig.synthesis's docstring. max_iterations is ignored in this
            # mode (no tool loop to bound).
            synthesis="single_call",
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

    result = run_meeting(
        config,
        resume=resume,
        extra_rounds=extra_rounds,
        meeting_id=None if resume else meeting_id,
    )

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
