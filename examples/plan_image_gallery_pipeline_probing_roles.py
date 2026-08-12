"""Planning meeting (planning_rounds mode): same gallery-curation task and Stage-1
probing evidence pack as plan_image_gallery_pipeline_planning_rounds.py, but with a
different participant roster -- instead of four generic pipeline-planner roles, the
participants ARE the six Stage-1 probing roles that produced
RECON_REPORT_ROOT/<role_name>/report.md (corpus_cartographer, visual_taxonomist,
carousel_quality_analyst, duplicate_diversity_analyst, graphic_text_risk_analyst,
automation_probe), plus a Skeptic. The idea: the people who know their own probing
evidence best argue directly for how it should shape the pipeline, instead of going
through an intermediary planner role that only reads their reports secondhand.

Everything else -- task spec, Stage-1 usage boundaries, recon guidance, round/judge/planner
mechanics -- is identical to plan_image_gallery_pipeline_planning_rounds.py. See that
file's docstring and agent_meeting/runner.py's _run_planning_rounds/_run_judge_step/_run_planner_step
and agent_meeting/judge.py for the mechanics.

Model split (fixed by the framework, not configured here): participants and the judge
run on DeepSeek v4 Pro at medium thinking; the planner runs on Codex gpt-5.6-sol at
high thinking via research_agent's codex credentials, since only the planner needs to
reason over the whole transcript and produce something implementation-ready.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_probing_roles.py

Resume a partial run the same way as the other examples (already-completed rounds are
skipped; the planner step never runs mid-round, so resume never has to worry about a
partially-run planner):
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_probing_roles.py --resume mtg_14bf5d2086
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_en.txt"
RECON_REPORT_ROOT = Path(r"C:\Users\LX034\Code\DataBase\两次report\7_20_exp_report_multi_role_2")

RECON_ENTRYPOINTS = [
    "synthesis/report.md",
    "synthesis/records/dataset_matrix.csv",
    "synthesis/records/agreements_disagreements.csv",
    "automation_probe/report.md",
    "carousel_quality_analyst/report.md",
    "corpus_cartographer/report.md",
    "duplicate_diversity_analyst/report.md",
    "graphic_text_risk_analyst/report.md",
    "visual_taxonomist/report.md",
]

MEETING_INSTRUCTIONS = """\
=== Meeting goal ===
Task clarification: The goal is to understand the actual contents and calibrate a future selection algorithm for an official website gallery presented as a horizontal carousel with consistent image height and proportional scaling.

This is a planning meeting, not an execution meeting. The goal is to produce a concrete plan that can be handed
directly to a later Executor. You do not need to (and should not) run the full pipeline over all 61,958 images.

This meeting's process differs from before: each round you only need to give key points, suggestions, and ideas --
do not write the final solution, pipeline design, or a numbered step-by-step implementation sequence yourself -- that
is done after the meeting by a dedicated Planner role. If you notice yourself writing numbered implementation steps
like "Step 1/Step 2", that means you've overstepped -- rewrite it as a suggestion for the Planner to consider.

Small-scale verification is allowed: for example, using the files/terminal tools to read a small number of sample
images from a dataset under C:\\pics and run a short piece of code to verify whether some threshold, heuristic
metric, or model call is actually feasible, to support your judgment -- but do not attempt to process an entire
dataset or run the full pipeline.

=== This meeting's participant roster ===
This meeting's participants are not generic pipeline-planning roles -- they are the six probing roles that actually
performed the Stage 1 exploration (corpus_cartographer / visual_taxonomist / carousel_quality_analyst /
duplicate_diversity_analyst / graphic_text_risk_analyst / automation_probe), plus a Skeptic role. Each probing role
should speak, argue, and suggest primarily based on the report and records they themselves originally produced; when
another participant's suggestion conflicts with your own Stage 1 evidence, point out the conflict and cite the
specific file. When you feel a suggestion falls outside your own probing scope (e.g. corpus_cartographer being asked
to judge aesthetic quality), say clearly that it's outside your area of expertise and point out which role should
answer it.

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

=== Original task requirements ===
{task_spec}

=== Stage 1 exploration priors (completed dataset scan results, no need to re-explore) ===
{probing}
"""


def build_recon_guidance() -> str:
    lines = [
        "=== Recon Evidence Pack: selected multi-role probing report ===",
        f"Selected report root: {RECON_REPORT_ROOT}",
        "",
        "Only use this one selected probing run for this meeting. Do not mix in the other probing report unless the meeting explicitly asks for cross-run comparison.",
        "You (a probing-role participant) authored one of these role reports yourself. Your primary evidence should be your own role_name/report.md, records/, examples/, and review_materials/ -- but you may also read another role's report to check for conflicts or to answer a question outside your own scope by pointing at whose scope it is.",
        "",
        "Round-1 requirement for every participant:",
        "- State which role report is your own, and ground your opening position in it with concrete file citations.",
        "- If you reference another role's findings, cite the exact file path(s) and state whether the evidence is a census, deterministic full pass, sample, feasibility test, or untested hypothesis.",
        "- If a probing report recommends manual review, do not copy that into the final pipeline. Convert it into an automated policy such as AUTO_EXCLUDE, AUTO_QUARANTINE, ABSTAIN_FROM_GALLERY, audit_log_only, or a falsification/metric requirement.",
        "- If you disagree with another role's probing finding, run or propose a small reproducible counter-test against the artifact, not a subjective objection.",
        "",
        "Suggested entrypoints, not assignments:",
    ]
    for rel_path in RECON_ENTRYPOINTS:
        lines.append(f"- {RECON_REPORT_ROOT / rel_path}")
    lines.extend([
        "",
        "You may also inspect any records/, examples/, review_materials/, scripts, or logs under the same selected root when they are relevant.",
    ])
    return "\n".join(lines)


def build_question() -> str:
    task_spec = TASK_SPEC_PATH.read_text(encoding="utf-8")
    probing = build_recon_guidance()
    return MEETING_INSTRUCTIONS.format(task_spec=task_spec, probing=probing)


def main() -> None:
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]

    # Participants and judge default to DeepSeek v4 Pro at medium thinking (judge is
    # hardcoded this way in agent_meeting/judge.py); set explicitly here too so the
    # roster doesn't silently fall back to whatever model/provider each role's own
    # DEFINITION.md frontmatter happens to specify.
    participant_defaults = dict(model="deepseek-v4-flash", provider="deepseek", reasoning_effort="high")

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=12,
        participants=[
            ParticipantConfig(
                name="CorpusCartographer",
                role="Corpus structure, format, and exact-duplicate identity owner",
                role_ref="corpus-cartographer",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="VisualTaxonomist",
                role="Editorial taxonomy and visual-system owner",
                role_ref="visual-taxonomist",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="CarouselQualityAnalyst",
                role="Fixed-height carousel geometry and technical compatibility owner",
                role_ref="carousel-quality-analyst",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="DuplicateDiversityAnalyst",
                role="Exact/near-duplicate and diversity-collapse owner",
                role_ref="duplicate-diversity-analyst",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="GraphicTextRiskAnalyst",
                role="Graphic/text/UI prevalence and public-gallery risk owner",
                role_ref="graphic-text-risk-analyst",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="AutomationProbe",
                role="Tested automation-feasibility owner",
                role_ref="automation-probe",
                max_iterations=40,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="Skeptic",
                role="Constraint auditor challenging unsupported or invalid ideas",
                role_ref="skeptic-reviewer",
                max_iterations=40,
                **participant_defaults,
            ),
        ],
        planner=PlannerConfig(
            name="Planner",
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            max_iterations=25,
        ),
    )
    result = run_meeting(config, resume=resume)

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
