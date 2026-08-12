"""Planning meeting (planning_rounds mode): same gallery-curation task as
plan_image_gallery_pipeline.py / plan_image_gallery_pipeline_moderator.py, but with a
third orchestration style -- participants never draft a Plan themselves, they only
contribute points/suggestions/ideas each round; after every round a separate judge
model decides whether the discussion has said enough (stop) or needs another round
(continue), hard-capped at max_rounds regardless of what the judge says; once
stopped, a dedicated Planner agent synthesizes the entire multi-round discussion into
the final Plan and writes it to disk. See agent_meeting/runner.py's
_run_planning_rounds/_run_judge_step/_run_planner_step and agent_meeting/judge.py.

Model split: the four domain participants (VisionCriteria/CPUPipeline/
DiversityRanking/GalleryCurator) run on DeepSeek v4 Pro; VisualAuditor -- the
participant that actually opens image files with the view_image tool and grounds
other participants' visual claims against real pixels -- runs on Codex gpt-5.5,
since DeepSeek v4 Pro has no confirmed vision support and only Codex's Responses
API path in this codebase can actually deliver image content to the model; the
judge runs on Codex gpt-5.5 (agent_meeting/judge.py, fixed by the framework, not
configured here); the planner runs on Codex gpt-5.6-sol at high thinking via
research_agent's codex credentials, since only the planner needs to reason over the
whole transcript and produce something implementation-ready.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_planning_rounds.py

Resume a partial (in_progress -- crashed mid-run) run the same way as the other
examples (already-completed rounds are skipped; the planner step never runs
mid-round, so resume never has to worry about a partially-run planner):
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_planning_rounds.py --resume mtg_14bf5d2086

Reopen a meeting that already reached status="completed" (e.g. the judge stopped
after round 1 and you want more discussion) for N more rounds -- keeps every already-
completed round's turns, drops the old planner step, runs N more rounds, then
re-synthesizes the plan over the full old+new discussion:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_planning_rounds.py --resume mtg_14bf5d2086 --extra-rounds 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig
# Private, but reused rather than duplicated -- this is the framework's default
# planning_rounds extra_system_prompt; we only want to ADD the VisualAuditor note
# below, not replace it, and copying its text here would drift the moment runner.py's
# copy is edited.
from agent_meeting.runner import _IDEAS_ONLY_ADDENDUM

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_en.txt"
RECON_REPORT_ROOT = Path(r"C:\Users\LX034\Code\DataBase\reports-20 groups")
RECON_REPORT_FILES = ["report_1.md", "report_2.md", "report_3.md"]

MEETING_INSTRUCTIONS = """\
=== Meeting goal ===
The goal is to understand the actual content of each dataset, and to calibrate an executable plan for the official-
website gallery selection algorithm described below in "Original task requirements" (including its fixed-height
horizontal carousel display-format constraint).

This is a planning meeting, not an execution meeting. The goal is to produce a concrete plan that can be handed
directly to a later Executor. You do not need to (and should not) run the full pipeline over the entire image corpus.

This meeting's process differs from before: each round you only need to give key points, suggestions, and ideas --
do not write the final solution, pipeline design, or a numbered step-by-step implementation sequence yourself -- that
is done after the meeting by a dedicated Planner role. If you notice yourself writing numbered implementation steps
like "Step 1/Step 2", that means you've overstepped -- rewrite it as a suggestion for the Planner to consider.

Small-scale verification is allowed: for example, using the files/terminal tools to read a small number of sample
images from a dataset under C:\\pics and run a short piece of code to verify whether some threshold, heuristic
metric, or model call is actually feasible, to support your judgment -- but do not attempt to process an entire
dataset or run the full pipeline.

=== Meeting hard constraint: boundaries on using Stage 1 priors ===
The Stage 1 exploration results below may only be used for:
* Estimating scale, cost, runtime, candidate-pool size, and QA sampling volume.
* Identifying risk areas that need focused review.
* Choosing configurable initial thresholds or dry-run calibration points.
* Designing output statistics, logs, and a manual-review package.

The Stage 1 exploration results below may NOT be used for:
* Directly judging whether a specific image, dataset, or file type is suitable for the gallery.
* Hardcoding dataset-name special cases, e.g. "skip the model for this dataset / it must all be real photos."
* Using filenames, paths, directory names, source descriptions, timestamps, EXIF, or content_note as evidence of
  visual suitability.
* Upgrading a small-sample observation into a general hard rule.

If you propose any strategy based on the current data distribution, it must be labeled as a "configurable
default/risk note/cost estimate," and must not be written as an inviolable suitability rule.

Boundary examples (for judging whether a Stage 1 finding "can be used directly," rather than going by feel):
* Can be used directly: a specific image's native size/orientation/aspect ratio, and whether it needs to be enlarged
  at the given target display height and by how much -- this is a measurement of the image's own geometric facts,
  and can be used directly as evidence for whether it suits the current display format ("this image needs an 8.57x
  enlargement, unsuitable" is a valid conclusion).
* Cannot be used directly: because a dataset has a higher proportion of enlargement issues, concluding that "this
  whole dataset is unsuitable for the gallery" or "this dataset should skip the display-fit check" -- this upgrades a
  dataset-level statistic into a judgment about the dataset itself, which is still the hard-constraint-forbidden
  "hardcoded dataset special case."
* Can be used directly: a list of specific duplicate file paths from a Stage 1 census/exhaustive pass (e.g. several
  images with identical SHA) -- this is a reviewable deterministic fact and can be used directly for deduplication.
* Cannot be used directly: skipping the real-world-credibility or display-fit judgment for individual images in a
  dataset merely because a Stage 1 report says "this dataset overall skews engineering/reference material" -- a
  categorical description cannot replace a per-image judgment.

Every requirement below about "citing Stage 1 evidence" applies to every round of this meeting (not just round 1):
whenever your statement in any round involves a substantive judgment based on the data distribution, you must be
able to cite a specific file path and evidence type -- it must not degrade in later rounds into speaking "from the
impression of the previous round."

=== Round 1 mandatory task: environment capability probe ===
Before proposing any architecture, in round 1 everyone must first use the terminal tool to actually probe whether the
candidate libraries/models your role will use are really available on this machine (not guessing from documentation,
but actually importing/calling them once), and write the result into a shared file under the shared/ directory (e.g.
shared/env_capability_probe.md) so others can read it directly without re-probing:
* Whether loading/calling actually succeeds (not "should be supported") -- if it errors, paste the full error
  message.
* Clearly unavailable options should be marked "excluded" immediately -- do not repeatedly re-propose the same
  already-tested-and-failed option in later rounds.
* If someone else has already probed the same library/model you need in shared/env_capability_probe.md, cite their
  result directly rather than re-probing.
This step exists to avoid a one-time fact like "can this model actually run on this machine" being dragged out and
exposed piecemeal by trial and error through rounds 5-7 -- that would waste several rounds that could otherwise be
spent discussing architectural disagreements.

=== Original task requirements ===
{task_spec}

=== Stage 1 exploration priors (completed dataset scan results, no need to re-explore) ===
{probing}
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


def build_question() -> str:
    task_spec = TASK_SPEC_PATH.read_text(encoding="utf-8")
    probing = build_recon_guidance()
    return MEETING_INSTRUCTIONS.format(task_spec=task_spec, probing=probing)


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

# Adds the VisualAuditor request protocol on top of (not instead of) the
# framework's default ideas-only framing, so every participant -- not just
# VisualAuditor -- knows requests are possible.
PARTICIPANT_DISCUSSION_ADDENDUM = (
    _IDEAS_ONLY_ADDENDUM
    + "\n\n"
    + (
        "This meeting includes VisualAuditor, a participant that can actually open "
        "image files and report what they show. If a claim in this discussion hinges "
        "on what a specific image or set of images actually looks like and you want "
        "it checked, write a request naming the exact file path(s) and your question "
        "to image_review_request.md in your own shared subfolder. VisualAuditor "
        "checks every participant's shared subfolder for this file each round."
    )
)


def main() -> None:
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]
    extra_rounds = None
    if "--extra-rounds" in sys.argv:
        extra_rounds = int(sys.argv[sys.argv.index("--extra-rounds") + 1])

    # Participants and judge default to DeepSeek v4 Pro at medium thinking (judge is
    # hardcoded this way in agent_meeting/judge.py); set explicitly here too so the
    # roster doesn't silently fall back to whatever model/provider each role's own
    # DEFINITION.md frontmatter happens to specify.
    participant_defaults = dict(model="deepseek-v4-flash", provider="deepseek", reasoning_effort="high")

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=3,
        participants=[
            ParticipantConfig(
                name="VisionCriteria",
                role="Visual-only criteria and boundary-case rubric",
                role_ref="vision-criteria-planner",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="CPUPipeline",
                role="Windows CPU execution, dependency, and runtime planner",
                role_ref="cpu-pipeline-planner",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="DiversityRanking",
                role="Deduplication, similarity, ranking, and diversity planner",
                role_ref="diversity-ranking-planner",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="GalleryCurator",
                role="Gallery display quality, review package, and human QA planner",
                role_ref="gallery-display-curator",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="VisualAuditor",
                role="Grounds other participants' visual claims by actually opening images",
                role_ref="visual-evidence-auditor",
                meeting_brief=VISUAL_AUDITOR_MEETING_BRIEF,
                # Must be codex, not deepseek -- this is the only provider path in
                # this codebase that can actually deliver image content to the model
                # (research_agent's Responses API image support + view_image tool).
                model="gpt-5.5",
                provider="codex",
                reasoning_effort="high",
                max_iterations=16,
                # The only participant here with view_image in its tool registry --
                # every other participant defaults to vision_capable=False, so even
                # if one of them tried to call it, it simply wouldn't be offered.
                vision_capable=True,
            ),
        ],
        planning_participant_addendum=PARTICIPANT_DISCUSSION_ADDENDUM,
        planner=PlannerConfig(
            name="Planner",
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            max_iterations=25,
        ),
    )
    result = run_meeting(config, resume=resume, extra_rounds=extra_rounds)

    print(f"\nmeeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")

    for step in result["steps"]:
        if step["decided_by"] == "judge":
            turn = step["turns"][0]
            verdict = "STOP" if turn.get("stop") else "continue"
            print(f"\n{'=' * 20} JUDGE after round {turn['round']}: {verdict} {'=' * 20}")
            print(f"reasoning: {turn.get('reasoning', '')}")
            if turn.get("unresolved_issues"):
                print(f"unresolved_issues: {turn['unresolved_issues']}")
            if turn.get("override_reason"):
                print(f"OVERRIDE: {turn['override_reason']}")
            continue
        if step["decided_by"] == "planner":
            continue  # printed separately as FINAL PLAN below
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
