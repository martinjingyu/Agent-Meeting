"""Planning meeting (moderator mode): same task as plan_image_gallery_pipeline.py --
4 planner roles + a skeptical reviewer design a CPU-only image gallery curation
pipeline for C:\\pics -- but here a moderator agent decides who speaks, in what order,
and how many times, instead of a fixed parallel_qa schedule. The moderator itself runs
on a high-tier model via Codex credentials (see research_agent/llm.py's provider="codex"
path) rather than the participants' regular model, since orchestration quality (asking
the right follow-up, knowing when the plan is actually done) benefits more from a
stronger model than any single planner's answer does.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_moderator.py

Resume a partial run the same way as the parallel_qa version -- moderator mode resumes
its own conversation from the saved session (agent.run(history=...)), not from a round
boundary:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_moderator.py --resume mtg_14bf5d2086
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ModeratorConfig, ParticipantConfig, run_meeting

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

As the moderator, you have convened 4 planning experts and 1 skeptical reviewer (see roster). Call on them in
whatever order you think appropriate (more than one round is fine), using meeting_set_agenda / meeting_add_notes as
needed to maintain shared context, and make sure the skeptical reviewer is asked at least once after the others have
given an initial plan, to challenge any assumption in the plan that isn't solid enough. Once you judge the plan to be
concrete, well-argued, and able to withstand scrutiny, use meeting_conclude to deliver the final plan.

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
        mode="moderator",
        # Pre-seed the roster with the same planning roles the parallel_qa version
        # uses -- the moderator still decides who speaks, in what order, and how many
        # times, it just doesn't have to spend iterations discovering roles via
        # role_list/role_load first.
        participants=[
            ParticipantConfig(name="VisionCriteria", role_ref="vision-criteria-planner", max_iterations=15),
            ParticipantConfig(name="CPUPipeline", role_ref="cpu-pipeline-planner", max_iterations=15),
            ParticipantConfig(name="DiversityRanking", role_ref="diversity-ranking-planner", max_iterations=15),
            ParticipantConfig(name="GalleryCurator", role_ref="gallery-display-curator", max_iterations=15),
            ParticipantConfig(name="Skeptic", role_ref="skeptic-reviewer", max_iterations=15),
        ],
        moderator=ModeratorConfig(
            name="Moderator",
            model="gpt-5.4",
            provider="codex",
            max_iterations=40,
        ),
    )
    result = run_meeting(config, resume=resume)

    print(f"\nmeeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")

    for step in result["steps"]:
        turn = step["turns"][0]
        if step["trigger_reason"] == "moderator session (full decision trajectory)":
            continue  # printed separately as FINAL PLAN below
        header = f"{turn['agent']} ({turn.get('role_ref')}) -- {step['trigger_reason']}"
        print(f"\n{'=' * 20} {header} {'=' * 20}\n{turn['output']}")

    print(f"\n{'=' * 20} FINAL PLAN (moderator's conclusion) {'=' * 20}\n")
    print(result["final_response"])


if __name__ == "__main__":
    main()
