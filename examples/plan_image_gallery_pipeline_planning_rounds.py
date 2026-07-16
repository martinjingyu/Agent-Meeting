"""Planning meeting (planning_rounds mode): same gallery-curation task as
plan_image_gallery_pipeline.py / plan_image_gallery_pipeline_moderator.py, but with a
third orchestration style -- participants never draft a Plan themselves, they only
contribute points/suggestions/ideas each round; after every round a separate judge
model decides whether the discussion has said enough (stop) or needs another round
(continue), hard-capped at max_rounds regardless of what the judge says; once
stopped, a dedicated Planner agent synthesizes the entire multi-round discussion into
the final Plan and writes it to disk. See agent_meeting/runner.py's
_run_planning_rounds/_run_judge_step/_run_planner_step and agent_meeting/judge.py.

Model split (fixed by the framework, not configured here): participants and the judge
run on DeepSeek v4 Pro at medium thinking; the planner runs on Codex gpt-5.6-sol at
high thinking via research_agent's codex credentials, since only the planner needs to
reason over the whole transcript and produce something implementation-ready.

Run with:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_planning_rounds.py

Resume a partial run the same way as the other examples (already-completed rounds are
skipped; the planner step never runs mid-round, so resume never has to worry about a
partially-run planner):
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_pipeline_planning_rounds.py --resume mtg_14bf5d2086
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_cn.txt"
PROBING_PATH = REPO_ROOT / "stage1_handoff_zh.json"

MEETING_INSTRUCTIONS = """\
=== 会议目标 ===
这是一次规划（planning）会议，不是执行会议。目标是产出一份具体、可直接交给后续
Executor 执行的方案。你不需要（也不应该）跑完整的 61958 张图片的全量 pipeline。

这次会议的流程和以往不同：每一轮你只需要给出要点、建议和想法，不要自己写出
最终方案、pipeline 设计或分步骤实现顺序 —— 那是会议结束后由专门的 Planner
角色来完成的事情。如果你发现自己在写"第一步/第二步"这种带编号的实现步骤，
说明你已经越界了，应该把它改写成一条给 Planner 参考的建议。

允许做小规模验证：比如用 files/terminal 工具读取 C:\\pics 下某个数据集里的少量
样本图片，跑一小段代码验证某个阈值、某个启发式指标或某个模型调用是否真的可行，
用来支撑你的判断 —— 但不要尝试处理整个数据集或跑完整 pipeline。

=== 会议硬约束：Stage 1 先验的使用边界 ===
下面的 Stage 1 探索结果只允许用于：
* 估算规模、成本、运行时间、候选池大小和 QA 抽样量。
* 识别需要重点复核的风险区域。
* 选择可配置的初始阈值或 dry-run 校准点。
* 设计输出统计、日志和人工复核包。

下面的 Stage 1 探索结果不允许用于：
* 直接判断某张图、某个数据集或某类文件是否适合 gallery。
* 写死 dataset name 特例，例如"某数据集跳过模型/必然全是真实照片"。
* 使用文件名、路径、目录名、来源描述、时间、EXIF 或 content_note 作为视觉适配性依据。
* 把少量样本观察升级成通用硬规则。

如果你提出任何基于当前数据分布的策略，必须标注为"可配置默认值/风险提示/成本规划"，
不能写成不可违反的 suitability 规则。

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

    # Participants and judge default to DeepSeek v4 Pro at medium thinking (judge is
    # hardcoded this way in agent_meeting/judge.py); set explicitly here too so the
    # roster doesn't silently fall back to whatever model/provider each role's own
    # DEFINITION.md frontmatter happens to specify.
    participant_defaults = dict(model="deepseek-v4-pro", provider="deepseek", reasoning_effort="medium")

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=8,
        participants=[
            ParticipantConfig(
                name="VisionCriteria",
                role="Visual-only criteria and boundary-case rubric",
                role_ref="vision-criteria-planner",
                max_iterations=12,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="CPUPipeline",
                role="Windows CPU execution, dependency, and runtime planner",
                role_ref="cpu-pipeline-planner",
                max_iterations=12,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="DiversityRanking",
                role="Deduplication, similarity, ranking, and diversity planner",
                role_ref="diversity-ranking-planner",
                max_iterations=12,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="GalleryCurator",
                role="Gallery display quality, review package, and human QA planner",
                role_ref="gallery-display-curator",
                max_iterations=12,
                **participant_defaults,
            ),
            ParticipantConfig(
                name="Skeptic",
                role="Constraint auditor challenging unsupported or invalid ideas",
                role_ref="skeptic-reviewer",
                max_iterations=12,
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
