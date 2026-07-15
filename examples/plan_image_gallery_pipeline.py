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
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_cn.txt"
PROBING_PATH = REPO_ROOT / "stage1_handoff_zh.json"

MEETING_INSTRUCTIONS = """\
=== 会议目标 ===
这是一次规划（planning）会议，不是执行会议。目标是产出一份具体、可直接交给后续
Executor 执行的方案。你不需要（也不应该）跑完整的 61958 张图片的全量 pipeline。

允许做小规模验证：比如用 files/terminal 工具读取 C:\\pics 下某个数据集里的少量
样本图片，跑一小段代码验证某个阈值、某个启发式指标或某个模型调用是否真的可行，
用来支撑你的方案判断 —— 但不要尝试处理整个数据集或跑完整 pipeline。

请在你的专长范围内给出具体、可论证的设计（不是泛泛而谈），并在最后给出一个可以
写进最终方案里的结论段落。

=== 会议硬约束：Stage 1 先验的使用边界 ===
下面的 Stage 1 探索结果只允许用于：
* 估算规模、成本、运行时间、候选池大小和 QA 抽样量。
* 识别需要重点复核的风险区域。
* 选择可配置的初始阈值或 dry-run 校准点。
* 设计输出统计、日志和人工复核包。

下面的 Stage 1 探索结果不允许用于：
* 直接判断某张图、某个数据集或某类文件是否适合 gallery。
* 写死 dataset name 特例，例如“某数据集跳过模型/必然全是真实照片”。
* 使用文件名、路径、目录名、来源描述、时间、EXIF 或 content_note 作为视觉适配性依据。
* 把少量样本观察升级成通用硬规则。

如果你提出任何基于当前数据分布的策略，必须标注为“可配置默认值/风险提示/成本规划”，
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
