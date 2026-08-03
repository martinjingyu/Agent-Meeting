"""Planning meeting (planning_rounds mode): same gallery-curation task as
plan_image_gallery_pipeline.py / plan_image_gallery_pipeline_moderator.py, but with a
third orchestration style -- participants never draft a Plan themselves, they only
contribute points/suggestions/ideas each round; after every round a separate judge
model decides whether the discussion has said enough (stop) or needs another round
(continue), hard-capped at max_rounds regardless of what the judge says; once
stopped, a dedicated Planner agent synthesizes the entire multi-round discussion into
the final Plan and writes it to disk. See agent_meeting/runner.py's
_run_planning_rounds/_run_judge_step/_run_planner_step and agent_meeting/judge.py.

Model split (fixed by the framework, not configured here): participants run on
DeepSeek v4 Pro; the judge runs on Codex gpt-5.5 (agent_meeting/judge.py); the planner
runs on Codex gpt-5.6-sol at high thinking via research_agent's codex credentials,
since only the planner needs to reason over the whole transcript and produce
something implementation-ready.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_cn.txt"
RECON_REPORT_ROOT = Path(r"C:\Users\LX034\Code\DataBase\两次report\7_20_exp_report_multi_role_2")

# (relative path, "especially relevant to" hint) -- the hint names which
# participant(s) below should treat that report as a priority read in round 1,
# not an exclusive assignment: anyone may still open anything under the root.
RECON_ENTRYPOINTS = [
    ("synthesis/report.md", "everyone (index/conflict map -- not a substitute for a concrete report)"),
    ("synthesis/records/dataset_matrix.csv", "everyone"),
    ("synthesis/records/agreements_disagreements.csv", "Skeptic"),
    ("automation_probe/report.md", "CPUPipeline"),
    ("carousel_quality_analyst/report.md", "GalleryCurator, VisionCriteria"),
    ("corpus_cartographer/report.md", "DiversityRanking, CPUPipeline"),
    ("duplicate_diversity_analyst/report.md", "DiversityRanking"),
    ("graphic_text_risk_analyst/report.md", "VisionCriteria, Skeptic"),
    ("visual_taxonomist/report.md", "VisionCriteria"),
]

MEETING_INSTRUCTIONS = """\
=== 会议目标 ===
目标是摸清各数据集的实际内容，并为下面"任务原始要求"里描述的官网 gallery
选图算法（含其中的固定高度横向 carousel 展示形态约束）校准出一份可执行方案。

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

边界示例（用于判断某个 Stage 1 发现"能不能直接用"，而不是靠感觉判断）：
* 可以直接用：某张图片的原生尺寸/朝向/宽高比，在给定目标展示高度下是否需要放大、
  放大倍数是多少——这是对该图片自身几何事实的测量，可以直接作为它是否适合当前
  展示形态的判断依据（"这张图放大 8.57x，不适合"是合法结论）。
* 不能直接用：因为某个数据集里放大问题比例较高，就得出"这个数据集整体不适合
  gallery"或"这个数据集应该跳过展示适配性检查"——这是把数据集层面的统计，
  升级成了对数据集本身的判断，仍然是硬约束禁止的"写死 dataset 特例"。
* 可以直接用：Stage 1 census/exhaustive pass 给出的具体重复文件路径列表（例如某几张
  图片的 SHA 完全一致）——这是可复核的确定性事实，可以直接用来去重。
* 不能直接用：仅凭 Stage 1 报告一句"这个数据集整体偏工程/资料类"，就跳过对其中
  单张图片的实景可信度或展示适配性判断——分类描述不能替代逐图判断。

以下每一条"引用 Stage 1 证据"的要求，适用于本次会议的每一轮（不只是第 1 轮）：
每轮只要你的发言涉及基于数据分布的实质判断，都要能说出具体文件路径和证据类型，
而不能在后续轮次里退化成"凭上一轮印象"发言。

=== 第 1 轮强制任务：环境能力探针 ===
在提出任何架构建议之前，第 1 轮里每个人都必须先用 terminal 工具实际探测一遍你
这个角色会用到的候选库/模型在本机是否真的可用（不是查文档猜测，是实际 import/
调用一次），并把结果写进 shared/ 目录下的一个共享文件（如
shared/env_capability_probe.md），供其他人直接读取、不用重复探测：
* 具体加载/调用是否成功（不是"应该支持"），报错就把完整报错信息贴出来。
* 明显不可用的方案直接标记为"已排除"，不要在后续轮次反复重新提出同一个已经
  实测失败的方案。
* 如果别人已经在 shared/env_capability_probe.md 里探测过你要用的同一个库/模型，
  直接引用其结果，不要重复探测。
这一步是为了避免把"这个模型到底能不能在这台机器上跑起来"这种一次性事实，
拖到第 5-7 轮才通过试错逐个暴露——那样会浪费掉好几轮本可以用来讨论架构分歧
的时间。

=== 任务原始要求 ===
{task_spec}

=== Stage 1 探索先验（已完成的数据集扫描结果，不需要重新探索） ===
{probing}
"""


def build_recon_guidance() -> str:
    lines = [
        "=== Recon Evidence Pack: selected multi-role probing report ===",
        f"Selected report root: {RECON_REPORT_ROOT}",
        "",
        "Only use this one selected probing run for this meeting. Do not mix in the other probing report unless the meeting explicitly asks for cross-run comparison.",
        "Every participant has permission to inspect any file under the selected report root. Do not only read synthesis/report.md. Treat synthesis as an index and conflict map; your primary evidence should come from whichever concrete role report(s), records, scripts, logs, and examples you decide are most relevant to the claim you are making.",
        "",
        "Requirement for every participant, every round (not just round 1):",
        "- Decide for yourself which Stage-1 probing dimension(s) to inspect before making a material claim.",
        "- Open at least one concrete Stage-1 probing report or record under the selected root; do not rely only on the synthesis summary, and do not coast on a report you opened in an earlier round if this round's claim needs different evidence.",
        "- Briefly explain why you chose those Stage-1 artifact(s) for your role and for the current round's issue.",
        "- Cite the exact file path(s) you used and state whether the evidence is a census, deterministic full pass, sample, feasibility test, or untested hypothesis.",
        "- If a probing report recommends manual review, do not copy that into the final pipeline. Convert it into an automated policy such as AUTO_EXCLUDE, AUTO_QUARANTINE, ABSTAIN_FROM_GALLERY, audit_log_only, or a falsification/metric requirement.",
        "- If you disagree with a probing finding, run or propose a small reproducible counter-test against the artifact, not a subjective objection.",
        "",
        "Suggested entrypoints, not exclusive assignments -- the 'especially relevant to' hint tells you which",
        "report to prioritize opening in round 1 given your own role; anyone may still open anything else below:",
    ]
    for rel_path, hint in RECON_ENTRYPOINTS:
        lines.append(f"- {RECON_REPORT_ROOT / rel_path}  (especially relevant to: {hint})")
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
    extra_rounds = None
    if "--extra-rounds" in sys.argv:
        extra_rounds = int(sys.argv[sys.argv.index("--extra-rounds") + 1])

    # Participants and judge default to DeepSeek v4 Pro at medium thinking (judge is
    # hardcoded this way in agent_meeting/judge.py); set explicitly here too so the
    # roster doesn't silently fall back to whatever model/provider each role's own
    # DEFINITION.md frontmatter happens to specify.
    participant_defaults = dict(model="deepseek-v4-pro", provider="deepseek", reasoning_effort="high")

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=10,
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
