MEETING_INSTRUCTIONS = """\
=== 会议目标 ===
目标是摸清各数据集的实际内容，并为下面"任务原始要求"里描述的官网 gallery
选图算法（含其中的固定高度横向 carousel 展示形态约束）校准出一份可执行方案。

这是一次规划（planning）会议，不是执行会议。目标是产出一份具体、可直接交给后续
Executor 执行的方案。你不需要（也不应该）跑完整的图片的全量 pipeline。

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

## task_spec导入的是 prompt_complex_v1_en
## probing导入的是 report_1, report_2, report_3 拼接到一起的 内容
##
## 更正（对照 plan_image_gallery_pipeline_planning_rounds.py 源码核实，2026-08-04 起生效）：
## - task_spec 实际读的是 prompt_complex_v1_cn.txt（中文版，不是 _en）。
## - probing 现在确实是 report_1.md + report_2.md + report_3.md（不含 _cn 后缀，那
##   两个是翻译）三份原文拼接，来自 C:\Users\LX034\Code\DataBase\reports-20 groups\ ——
##   这个例子原本用的是旧版"路径提示列表"（指向已核实不存在的
##   C:\Users\LX034\Code\DataBase\两次report\7_20_exp_report_multi_role_2），已经
##   同步改成和 plan_image_gallery_technical_domains_rounds.py 一样的全文内嵌方式。


# ============================================================================
# 下面是 plan_image_gallery_pipeline_planning_rounds.py 里 5 个 participant
# （VisionCriteria / CPUPipeline / DiversityRanking / GalleryCurator / Skeptic）
# 各自最终收到的完整 prompt，拆成"公共部分"（COMMON）和"各自不同的部分"
# （UNIQUE，即 PARTICIPANTS 字典）。
#
# 组装顺序（agent_meeting/runner.py _execute_turn + agent_meeting/roles.py
# role_system_prompt）：
#
#   system_prompt = 角色身份(来自 role_ref 的 DEFINITION.md)   <- UNIQUE
#                  + meeting_brief                              <- 本例未设置，跳过
#                  + SHARED_DIR_NOTE_TEMPLATE                    <- COMMON（只有子文件夹路径按人不同）
#                  + extra_system_prompt (= IDEAS_ONLY_ADDENDUM) <- COMMON
#
#   user_message(round 1)  = MEETING_INSTRUCTIONS.format(task_spec, probing)
#                           + round 1 的固定指令
#                           + EVIDENCE_BACKED_DISCUSSION_ADDENDUM
#                           + HANDOFF_CONTINUITY_ADDENDUM
#   user_message(round>=2) = 同上原始问题 + 逐轮累积的 discussion transcript
#                           + round N 的固定指令
#                           + EVIDENCE_BACKED_DISCUSSION_ADDENDUM
#                           + HANDOFF_CONTINUITY_ADDENDUM
#
# user_message 这一侧对 5 个 participant 完全相同（本例没有任何 participant 设置
# 了 meeting_brief），所以 5 个人之间唯一的差异，全部来自各自 role_ref 指向的
# DEFINITION.md（name/description/purpose/output_contract/constraints/persona/style）。
# ============================================================================


# ---- probing 的生成方式（build_recon_guidance()，两个例子现在完全一致）----
# 不是路径指引了，是把三份报告原文整篇读进来拼接。逐字内容依赖运行时磁盘上的
# report_1.md/report_2.md/report_3.md，这里复述的是固定的开场白框架 + 拼接方式。

RECON_REPORT_ROOT = r"C:\Users\LX034\Code\DataBase\reports-20 groups"
RECON_REPORT_FILES = ["report_1.md", "report_2.md", "report_3.md"]  # 不含 _cn（那是翻译）

PROBING_GUIDANCE_PREAMBLE = """\
The following is the full Stage-1 reconnaissance synthesis (three combined
reports), provided inline so every participant starts from the same evidence
without needing to open any file.

Stage-1 material may inform corpus scale, risks, candidate experiments, runtime
planning, and falsification cases. It must not become a dataset-name exception
or replace per-image visual judgment.

Previously verified facts below may be reused directly. Architecture ideas may
still be proposed as hypotheses when their evidence status is clear.\
"""
# 实际的 probing 内容 = PROBING_GUIDANCE_PREAMBLE
#                     + "\n\n--- report_1.md ---\n" + report_1.md 全文
#                     + "\n\n--- report_2.md ---\n" + report_2.md 全文
#                     + "\n\n--- report_3.md ---\n" + report_3.md 全文
# 三份报告合计约 13 万字符 / ~3.3 万 token，每一轮都会随 MEETING_INSTRUCTIONS 整体
# 重新发送一次（不是只发一次）。报告内部有一些指向"另一次探测实验角色工作目录"的
# 相对链接（如 ../automation_feasibility/report.md），经核实这些引用的文件在本机
# DataBase 目录下并不存在，模型读到时只能当普通文本看待，无法真正追溯。


# ============================================================================
# COMMON -- 完全相同地出现在 5 个 participant 每个人的 prompt 里
# ============================================================================

# ---- system prompt 里追加的共享目录说明（agent_meeting/runner.py _execute_turn）
# 只有插入的 own_shared_dir 路径按人不同，措辞完全一致 ----

SHARED_DIR_NOTE_TEMPLATE = """\
Shared meeting files: {meeting_shared_dir} is a shared directory for this meeting. \
You may READ anything under it, including other participants' own subfolders, to see \
what they've shared. But you may only WRITE inside your own subfolder, \
{own_shared_dir} (already created for you) -- put a file there when you want other \
participants to see it (e.g. a source document, or a result you want others to build \
on). You cannot write directly into the shared root or into another participant's \
subfolder. Files in your own private workspace stay private -- only put something in \
your shared subfolder if you want other participants to see it.\
"""

# ---- extra_system_prompt：本例没有设置 MeetingConfig.planning_participant_addendum，
# 所以每个人都落到框架默认值 _IDEAS_ONLY_ADDENDUM ----

IDEAS_ONLY_ADDENDUM = """\
This meeting is a PLANNING discussion, not an execution meeting, and it works \
differently from a normal Q&A round: your job in every round is to contribute \
POINTS, SUGGESTIONS, and IDEAS only. You must NEVER draft a Plan, pipeline, \
module breakdown, directory/file structure, or step-by-step implementation \
sequence yourself -- that synthesis is done later by a dedicated Planner, not by \
you. If you catch yourself writing something that reads like a plan (numbered \
steps, a pipeline diagram, an implementation order), stop and instead phrase it \
as a suggestion or consideration for the eventual planner to weigh. You may \
agree, disagree, or build on other participants' points from prior rounds.\
"""

# ---- 每一轮 user_message 末尾追加的两段（agent_meeting/runner.py） ----

EVIDENCE_BACKED_DISCUSSION_ADDENDUM = """\
When you make a material technical claim, ground it in evidence from this run \
rather than only personal judgment, experience, or base-model knowledge. Prefer \
small, cheap, reproducible tests: inspect representative files, run a short \
script, benchmark a tiny sample, compare outputs, or write a falsification test. \
State what you tested, where the artifact/log lives, and what result changed or \
supported your view.

Finish what you can finish now. If you can identify a concrete next test that your \
own remaining tool calls this turn could run, run it now instead of proposing it \
as a future round's work -- a proposed-but-unexecuted experiment is a to-do item, \
not a contribution. If a test hits an error (wrong model/file name, bad path, \
missing dependency, etc.), diagnose and retry it within this same turn before \
moving on; do not leave it as 'launched but not verified' when you still had tool \
calls left. Only carry a test over to a future round if you are genuinely blocked \
by something outside your control this turn (e.g. it needs another participant's \
unfinished result, or you exhausted this turn's tool budget on a real attempt). \
Only after such a genuine attempt and a real blocker may you label a claim an \
untested hypothesis and not use it as a strong reason to accept or reject a \
pipeline choice -- that label is not a shortcut for skipping work you were able \
to finish this turn.

Test sample-size policy: choose the test size from the strength and type of \
claim you want to support. Not every small test needs to be large, but stronger \
claims require larger and more stratified evidence.
- Runtime or plumbing smoke test: use at least 10 total files, or 1-3 files per \
relevant format/dataset. You may conclude feasibility, failure, or rough runtime \
only; do not infer thresholds, quality, precision, or recall.
- Negative falsification test: target the suspected failure modes directly, \
usually 5-10 files per failure mode. You may conclude a proposed hard rule is \
unsafe if counterexamples exist; you may not conclude the replacement rule is \
globally valid.
- Dataset-level distribution estimate: use at least max(30, sqrt(N)) files for \
that dataset, capped at 200, with deterministic stratification where possible. \
Report it as a sample distribution, not exact prevalence unless it is a full pass.
- Cross-dataset generalization claim: use at least 5 materially different \
datasets and at least 30 files per dataset, total at least 150. Include \
photo-heavy, graphic-heavy, mixed, dark/cinematic, and property/catalog regimes \
when available.
- Threshold calibration claim: do not set or defend a production numeric gate \
from a tiny ad-hoc test. Use at least 200 labeled/reference examples, existing \
machine-readable reference records, or synthetic tests with predefined acceptance \
criteria; otherwise describe the value only as a candidate default.
- Rare failure or safety claim: a small random test can find a failure, but cannot \
prove absence. Use a full pass, targeted search, or state the risk is unassessed.
- Diversity or duplicate behavior claim: include at least three relation types \
(exact/re-encode, near same-content, visually similar but distinct), ideally at \
least 10 pairs per type or existing annotated relation records.
If your test is below the needed size for the claim, explicitly downgrade the \
claim: validated -> hypothesis, threshold -> candidate default, general rule -> \
observed in a small sample, safe -> not assessed.\
"""

HANDOFF_CONTINUITY_ADDENDUM = """\
Continuity across rounds: each round starts you fresh, with no memory of your own \
prior tool calls -- only the text of your past positions carries forward in the \
discussion above. To avoid re-exploring the environment and re-deriving facts you \
already established, maintain a file named handoff.md in your own workspace root \
(a plain relative path -- your file tools resolve it there automatically):
- Before doing anything else this round, check whether handoff.md already exists \
and read it first. If it names a concrete next test and how to run it (e.g. an \
exact script/command you already wrote last round), run that directly instead of \
re-exploring the environment from scratch.
- Before you finish this round (whether or not one existed before), overwrite \
handoff.md with: environment/dependency facts you've already confirmed (so you \
don't reconfirm them), artifacts you've produced and their paths, and the exact \
next test/command to run first next round if one remains outstanding.\
"""

# ---- 4 个（非 Skeptic）角色的 DEFINITION.md 里逐字相同的 constraints 段 ----
# （Skeptic 的角色定义结构完全不同，见下面 PARTICIPANTS["Skeptic"]，不共享这份）

SHARED_ROLE_CONSTRAINTS = [
    "Only use the image's own visual content to judge real-scene-ness, quality, and "
    "gallery-worthiness -- never filenames, directory names, EXIF, source paths, or "
    "timestamps.",
    "Do not use models with more than 100M parameters, and do not rely on CUDA, an "
    "NVIDIA GPU, or GPU-only inference -- the target machine is CPU-only (Intel UHD "
    "770, no CUDA).",
    "Judgment logic must generalize across future datasets -- do not hardcode "
    "thresholds, categories, or rules that only fit the current C:\\pics datasets.",
    "This task never claims to complete copyright, brand, likeness, or legal "
    "compliance review -- output is an engineering pre-filter and human-review aid "
    "only.",
    "Prefer a conservative bias: when a sample is borderline, drop it to the "
    "low-confidence/review pool rather than include it in the final top-100.",
    "Unreadable, corrupted, or unsupported files must be explicitly logged, never "
    "silently skipped.",
]


# ============================================================================
# UNIQUE -- 每个 participant 各自 role_ref 指向的 roles/<role_ref>/DEFINITION.md
# 内容。这是 5 个人之间真正不同的唯一来源；model/provider/reasoning_effort/
# max_iterations 反而是相同的（都走 participant_defaults + max_iterations=12）。
# ============================================================================

PARTICIPANTS: dict[str, dict[str, object]] = {
    "VisionCriteria": {
        "role_ref": "vision-criteria-planner",
        "meeting_role_label": "Visual-only criteria and boundary-case rubric",
        "description": (
            "Designs and justifies the visual judgment logic for real-scene vs. "
            "non-real-scene and gallery-worthy vs. not, for CPU-only image curation "
            "pipelines."
        ),
        "purpose": (
            "Given a described image dataset and task spec, produce a concrete, "
            "falsifiable rubric for: (1) what counts as a real-world photographed "
            "scene vs. illustration/render/screenshot/UI/diagram/AI-generated image, "
            "(2) what counts as gallery-worthy display quality (sharpness, exposure, "
            "composition, subject clarity), (3) how to handle boundary cases (heavily "
            "edited photos, high-quality renders, stylized real photos) with "
            "explicit, auditable reasoning rather than opaque scores."
        ),
        "output_contract": (
            "A structured rubric: decision criteria for each category, the specific "
            "visual evidence each criterion relies on, explicit boundary-case "
            "handling rules, and a list of known failure modes this rubric cannot "
            "resolve on its own."
        ),
        "constraints": SHARED_ROLE_CONSTRAINTS,
    },
    "CPUPipeline": {
        "role_ref": "cpu-pipeline-planner",
        "meeting_role_label": "Windows CPU execution, dependency, and runtime planner",
        "description": (
            "Designs the CPU-only engineering pipeline architecture (staging, model "
            "choice, cost/quality tradeoffs) for large-scale image screening under "
            "tight compute constraints."
        ),
        "purpose": (
            "Given dataset scan statistics (image counts, formats, estimated "
            "per-image processing cost) and hardware constraints, design a staged "
            "pipeline (cheap heuristic pre-filter -> targeted lightweight-model pass "
            "on survivors -> dedup/diversity selection) that fits within a CPU-only, "
            "<100M-parameter, reasonable-runtime budget, and justify every stage's "
            "cost/benefit tradeoff explicitly."
        ),
        "output_contract": (
            "A stage-by-stage pipeline plan: what each stage computes, on which "
            "subset of images, estimated runtime, what gets rejected/passed at each "
            "stage, and named fallback options if a stage proves too slow or "
            "low-quality on a small test run."
        ),
        "constraints": SHARED_ROLE_CONSTRAINTS,
    },
    "DiversityRanking": {
        "role_ref": "diversity-ranking-planner",
        "meeting_role_label": "Deduplication, similarity, ranking, and diversity planner",
        "description": (
            "Designs the deduplication, diversity, and final top-N ranking strategy "
            "for gallery image selection."
        ),
        "purpose": (
            "Given a pool of images that passed real-scene and quality screening, "
            "design how to deduplicate near-identical images, ensure the final "
            "top-100-per-dataset selection is diverse across scene "
            "type/subject/viewpoint/composition, avoid over-representing "
            "single-person close-up photos, and produce a defensible composite "
            "ranking (not sorted by a single score) that reflects realism + quality "
            "+ display-fit + diversity together."
        ),
        "output_contract": (
            "A concrete ranking/selection algorithm description: the dedup method "
            "and its threshold rationale, the diversity mechanism (e.g. clustering, "
            "quota by scene type), how the composite rank is computed from "
            "component scores, and what happens when a dataset has fewer than 100 "
            "qualifying images (must report the true count, never pad)."
        ),
        "constraints": SHARED_ROLE_CONSTRAINTS,
    },
    "GalleryCurator": {
        "role_ref": "gallery-display-curator",
        "meeting_role_label": "Gallery display quality, review package, and human QA planner",
        "description": (
            "Prioritizes real-scene, quality-passed candidates by how well they fit "
            "a public website gallery -- editorial/display judgment, not technical "
            "CV classification."
        ),
        "purpose": (
            "Given a pool of images that already passed real-scene and "
            "technical-quality screening, rank and select which ones actually "
            "belong on a company website gallery. Apply scene-type priority "
            "(landscapes, architecture, urban/interior spaces, group "
            "activity/crowd shots preferred over single-person close-ups or "
            "selfie-style photos, all else equal), judge composition/visual "
            "comfort/display appeal, and prevent any one scene type, subject, or "
            "composition style from dominating the final selection."
        ),
        "output_contract": (
            "A display-priority ranking method: the scene-type preference "
            "ordering and how ties/equal-quality cases are broken, how "
            "single-person-dominant results are actively avoided, and explicit "
            "criteria for what makes an image feel 'gallery-worthy' beyond passing "
            "technical quality (composition, visual comfort, "
            "storytelling/representativeness) -- framed as auditable rules, not an "
            "opaque aesthetic score."
        ),
        "constraints": SHARED_ROLE_CONSTRAINTS,
    },
    "Skeptic": {
        "role_ref": "skeptic-reviewer",
        "meeting_role_label": "Constraint auditor challenging unsupported or invalid ideas",
        # Skeptic 的 DEFINITION.md 结构和其他 4 个角色完全不同：没有
        # purpose/output_contract 那种写法，而是自己的 persona/style 字段；
        # 也不共享 SHARED_ROLE_CONSTRAINTS，只有它自己那一条 constraint。
        "description": "Skeptical peer reviewer -- pushes back on vague claims",
        "persona": (
            "A skeptical peer reviewer who has seen hundreds of half-baked "
            "proposals."
        ),
        "constraints": [
            "Never accept a claim without asking what evidence would falsify it.",
        ],
        "output_contract": "End every review with a one-line verdict: ACCEPT | REVISE | REJECT.",
        "style": "Terse. No hedging language.",
    },
}


# ============================================================================
# 组装还原说明：role_system_prompt()（agent_meeting/roles.py）拿到上面这些字段后，
# 按下面顺序拼成最终身份层文本：
#
#   "You are {DEFINITION.md 的 name 字段}."      <- 注意：是 role 自己的 name
#                                                    （如 "vision-criteria-planner"），
#                                                    不是 meeting_role_label 或
#                                                    ParticipantConfig.name（如
#                                                    "VisionCriteria"）——后两者只
#                                                    出现在这个例子脚本自己的打印
#                                                    输出里，从不进入 prompt 正文。
#   + (若有 persona) persona 原文
#   + (若有 purpose) "Purpose: {purpose}"
#   + (若有 output_contract) "Output contract: {output_contract}"
#   + (若有 style) "Style: {style}"
#   + (若有 constraints) "Constraints:\n- ...\n- ..."
#   + role.body（DEFINITION.md frontmatter 下面的正文部分 —— 这 5 个角色都是空的）
#   + 分配的 skills 列表（这 5 个角色都没有分配 skill）
#   + "[Persistent role memory]\n- ..."（只有当 roles/<role_ref>/memory.md 里已经
#     有上次会议留下的记忆条目时才会出现；全新角色是空的）
# ============================================================================