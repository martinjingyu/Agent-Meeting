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
TASK_SPEC_PATH = REPO_ROOT / "prompt_complex_v1_cn.txt"
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
=== 会议目标 ===
Task clarification: The goal is to understand the actual contents and calibrate a future selection algorithm for an official website gallery presented as a horizontal carousel with consistent image height and proportional scaling.

这是一次规划（planning）会议，不是执行会议。目标是产出一份具体、可直接交给后续
Executor 执行的方案。你不需要（也不应该）跑完整的 61958 张图片的全量 pipeline。

这次会议的流程和以往不同：每一轮你只需要给出要点、建议和想法，不要自己写出
最终方案、pipeline 设计或分步骤实现顺序 —— 那是会议结束后由专门的 Planner
角色来完成的事情。如果你发现自己在写"第一步/第二步"这种带编号的实现步骤，
说明你已经越界了，应该把它改写成一条给 Planner 参考的建议。

允许做小规模验证：比如用 files/terminal 工具读取 C:\\pics 下某个数据集里的少量
样本图片，跑一小段代码验证某个阈值、某个启发式指标或某个模型调用是否真的可行，
用来支撑你的判断 —— 但不要尝试处理整个数据集或跑完整 pipeline。

=== 本次会议的参会者构成 ===
本次会议的参会者不是通用的 pipeline 规划角色，而是直接由完成 Stage 1 探索的
六个 probing 角色本人（corpus_cartographer / visual_taxonomist /
carousel_quality_analyst / duplicate_diversity_analyst / graphic_text_risk_analyst /
automation_probe）出席，外加一个 Skeptic 角色。每个 probing 角色应该主要依据
自己当初产出的那份 report 和 records 来发言、辩护、提出建议；当其他参会者的
建议与你自己 Stage 1 的证据冲突时，应该指出冲突并引用具体文件。当你觉得某个
建议超出了你自己的 probing 范围（例如 corpus_cartographer 被要求评判审美质量），
应该明确说这不在你的专业范围内，并指出应该由哪个角色来回答。

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
