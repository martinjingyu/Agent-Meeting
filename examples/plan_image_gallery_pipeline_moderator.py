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
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_cn.txt"
PROBING_PATH = REPO_ROOT / "stage1_handoff_zh.json"

MEETING_INSTRUCTIONS = """\
=== 会议目标 ===
这是一次规划（planning）会议，不是执行会议。目标是产出一份具体、可直接交给后续
Executor 执行的方案。你不需要（也不应该）跑完整的 61958 张图片的全量 pipeline。

允许做小规模验证：比如用 files/terminal 工具读取 C:\\pics 下某个数据集里的少量
样本图片，跑一小段代码验证某个阈值、某个启发式指标或某个模型调用是否真的可行，
用来支撑你的方案判断 —— 但不要尝试处理整个数据集或跑完整 pipeline。

作为主持人，你已经召集了 4 位规划专家和 1 位怀疑论评审（见 roster）。按你认为
合适的顺序调用他们（可以不止一轮），必要时用 meeting_set_agenda / meeting_add_notes
维护共享上下文，确保怀疑论评审至少在其他人给出初步方案后被问一次，用于挑战方案里
不够扎实的假设。当你认为方案已经具体、可论证、经得起质疑时，用 meeting_conclude
给出最终方案。

=== 任务原始要求 ===
{task_spec}

=== Stage 1 探索先验（已完成的数据集扫描结果，不需要重新探索） ===
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
