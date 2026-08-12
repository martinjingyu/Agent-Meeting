"""Planning meeting: 4 planner roles + a skeptical reviewer design a CPU-only image
gallery curation pipeline for C:\\pics, given a task spec and Stage-1 probing results.
The output is a plan for a downstream Executor -- this meeting does NOT run the full
pipeline. Participants may do small-scale validation (e.g. read a handful of sample
images to sanity-check a threshold) but must not attempt a full-dataset run.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline.py

If it fails partway (e.g. a transient API connection error -- the log line right before
the traceback says "resume with run_meeting(config, resume='mtg_...')"), continue it
with the same meeting_id instead of starting over:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline.py --resume mtg_14bf5d2086
Already-completed rounds are skipped entirely; only the round that was in progress
re-runs (and within it, only participants who hadn't finished yet -- see runner.py's
per-turn cache).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_en.txt"
PROBING_PATH = REPO_ROOT / "stage1_handoff_en.json"

MEETING_INSTRUCTIONS = """\
=== Meeting goal ===
This is a planning meeting, not an execution meeting. The goal is to produce a concrete plan that can be handed
directly to a later Executor. You do not need to (and should not) run the full pipeline over all 61,958 images.

Small-scale verification is allowed: for example, using the files/terminal tools to read a small number of sample
images from a dataset under C:\\pics and run a short piece of code to verify whether some threshold, heuristic
metric, or model call is actually feasible, to support your judgment -- but do not attempt to process an entire
dataset or run the full pipeline.

Give a concrete, well-argued design within your area of expertise (not generalities), and end with a conclusion
paragraph that could be written directly into the final plan.

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


def build_question() -> str:
    task_spec = TASK_SPEC_PATH.read_text(encoding="utf-8")
    probing = PROBING_PATH.read_text(encoding="utf-8")
    return MEETING_INSTRUCTIONS.format(task_spec=task_spec, probing=probing)


def main() -> None:
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]

    config = MeetingConfig(
        question=build_question(),
        participants=[
            ParticipantConfig(
                name="VisionCriteria",
                role="Visual-only criteria and boundary-case rubric",
                role_ref="vision-criteria-planner",
                max_iterations=15,
            ),
            ParticipantConfig(
                name="CPUPipeline",
                role="Windows CPU execution, dependency, and runtime planner",
                role_ref="cpu-pipeline-planner",
                max_iterations=15,
            ),
            ParticipantConfig(
                name="DiversityRanking",
                role="Deduplication, similarity, ranking, and diversity planner",
                role_ref="diversity-ranking-planner",
                max_iterations=15,
            ),
            ParticipantConfig(
                name="GalleryCurator",
                role="Gallery display quality, review package, and human QA planner",
                role_ref="gallery-display-curator",
                max_iterations=15,
            ),
            ParticipantConfig(
                name="Skeptic",
                role="Constraint auditor with veto over unsupported or invalid plan items",
                role_ref="skeptic-reviewer",
                max_iterations=15,
            ),
        ],
        aggregation_strategy="audited_llm",
        final_audit=True,
        rounds=4,
    )
    result = run_meeting(config, resume=resume)

    print(f"\nmeeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")

    for step in result["steps"]:
        if step["trigger_reason"].startswith("aggregation"):
            turn = step["turns"][0]
            print(f"\n{'=' * 20} AGGREGATED PLAN after round {turn['round']} {'=' * 20}\n{turn['output']}")
            continue
        for turn in step["turns"]:
            changes = turn.get("changes_from_prior_round")
            role_ref = turn.get("role_ref") or turn.get("role") or turn.get("strategy") or "n/a"
            header = f"{turn['agent']} ({role_ref}) -- round {turn['round']}"
            if changes:
                header += f"\n[changes from prior round] {changes}"
            print(f"\n{'=' * 20} {header} {'=' * 20}\n{turn['output']}")

    print(f"\n{'=' * 20} FINAL PLAN (after round {config.rounds}) {'=' * 20}\n")
    print(result["final_response"])


if __name__ == "__main__":
    main()
