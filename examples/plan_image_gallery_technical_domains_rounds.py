"""Gallery-selection planning meeting with technology-domain participants.

This example keeps the planning_rounds interaction intentionally prose-first:
participants contribute free-form technical viewpoints, respond to each other, test
claims when useful, and revise their positions over multiple rounds. They are grouped
by technical domain rather than by stages of a pre-assumed filter-first pipeline.

Compared with plan_image_gallery_pipeline_planning_rounds.py, this example:

1. replaces the stage-oriented participant roster with technology-domain roles;
2. removes the assumption that the answer must be a threshold-first linear pipeline;
3. allows candidate technical compositions and architecture sketches while reserving
   the adopted final Plan for the dedicated Planner;
4. distinguishes the local <100M CPU constraint from optional external vision APIs;
5. does not assume or request human labels, and limits empirical claims to evidence
   available from deterministic facts, existing records, synthetic/programmatic tests,
   stability tests, disagreement analysis, and set-level proxy metrics;
6. gives only SystemsEvaluation the mandatory first-round environment survey;
7. treats Stage-1 reports as reusable evidence, not as a file every participant must
   reopen in every round;
8. replaces Skeptic with VisualAuditor -- instead of abstract pushback (which also
   had outsized influence via a REVISE/REJECT verdict that could override the
   judge's stop decision), VisualAuditor actually opens image files with the
   view_image tool and grounds other participants' visual claims against real
   pixels. It runs on Codex gpt-5.5 (provider="codex", vision_capable=True) --
   the six domain participants stay on DeepSeek v4 Pro, which has no confirmed
   vision support and would crash if it ever reached for view_image (see
   ParticipantConfig.vision_capable's docstring; that tool is only registered for
   participants with vision_capable=True).

Run:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py

Resume:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx

Skip any unfinished discussion round and synthesize immediately from the last
fully-checkpointed round:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx --planner-only

Reopen a completed meeting for more discussion:
    C:\\Users\\LX034\\miniconda3\\python.exe examples\\plan_image_gallery_technical_domains_rounds.py --resume mtg_xxxxxxxxxx --extra-rounds 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_meeting import MeetingConfig, ParticipantConfig, run_meeting
from agent_meeting.config import PlannerConfig
from agent_meeting.storage import load_meeting


REPO_ROOT = Path(__file__).resolve().parents[1]
RECON_REPORT_ROOT = Path(r"C:\Users\LX034\Code\DataBase\reports-20 groups")
RECON_REPORT_FILES = ["report_1.md", "report_2.md", "report_3.md"]


PARTICIPANT_TASK_SPEC = """\
输入与目标：

- 输入根目录为 `C:\\pics`。
- 根目录下每个一级子目录代表一个独立图片数据集；内部结构、格式、数量和
  命名方式均不可预设。
- 目标是从每个数据集中自动选出最多 100 张适合固定高度横向 carousel 的
  真实实景图片。
- 如果真正合格的图片不足 100 张，应输出实际数量，不得用低质、非实景、
  有害、重复或展示不适配的图片补足。

最终结果应综合考虑：

- 真实世界拍摄场景的可信度；
- 图片技术质量、主体清晰度、构图和公开展示价值；
- 固定高度 carousel 下的放大倍率、渲染宽度、视口占比和比例适配；
- 场景、主体、视角、色彩和构图的多样性；
- exact duplicate、near duplicate、同场景和高度相似内容；
- 避免单人近景或同一人物/场景过度占据最终结果；
- 明显网页、幻灯片、logo、UI、海报、图表、CG、AI 风格图和其他非实景内容；
- 适合自动识别的有害或敏感视觉内容。

信息使用边界：

- 判断真实性、质量、展示价值和内容安全时，只能使用图像像素、解码结果和
  单图几何事实。
- 文件名、路径、数据集名称、时间和非方向性 EXIF 不得作为视觉 suitability
  或语义类别的判断依据。
- 上述非视觉信息可以用于追溯、稳定排序、缓存、错误记录和纯工程管理。
- 如果文件名只用于提出待验证的 duplicate/family pair，最终关系仍必须由
  视觉内容确认，且不得影响真实性或质量判断。

自动化与评估边界：

- 原始数据不含人工 ground truth，本任务不得要求新增人工标注、人工调参、
  人工批准或人工复核作为生产结果生成的必要条件。
- 生产运行必须全自动。可以输出非阻塞审计材料、风险标记和不确定性信息。
- 技术验证可以使用确定性事实、已有机器可读记录、程序化/合成测试、
  metamorphic tests、无监督稳定性、方法间分歧、公开元数据和集合级代理指标。
- 没有可靠 ground truth 时，不得声称测得真实 precision、recall、审美准确率
  或内容安全召回率。外部视觉 API 的输出是模型判断或 weak reference，不是
  ground truth。

计算与服务约束：

- 本地执行模型必须小于 100M 参数，并支持 Windows CPU-only；不得依赖 CUDA、
  NVIDIA GPU 或 GPU-only 推理。
- 允许讨论在本地候选压缩后使用外部视觉 API，作为最终语义评价、比较排序
  或内容安全判断的一部分。
- 外部 API 建议必须同时讨论最大候选量、调用成本、隐私、批次/顺序偏差、
  可审计性、版本变化和纯本地 fallback。

范围声明：

- 本任务不声称自动完成版权、商标、品牌、肖像授权、隐私或法律审批。
- 会议负责技术方案论证；最终 Planner 负责把讨论综合成可执行工程方案。
"""


TECHNOLOGY_LANDSCAPE = """\
下面是技术导航，不是完整清单、推荐顺序或最终架构。参与者可以补充、反驳、
组合或放弃其中的方法。

1. 传统图像处理与图像取证
   - 颜色量化、局部熵、亮度/饱和度、动态范围、模糊、噪声、压缩；
   - 大面积近似色连通区域、矩形块、规则网格、边缘方向和空间布局；
   - OCR、文字区域、网页/幻灯片/logo/UI/海报的结构信号；
   - 频域、梯度、局部纹理和无参考图像质量指标。

2. 视觉表征与学习型模型
   - CNN 中间特征、ImageNet 分类器、轻量场景/质量分类器；
   - CLIP/MobileCLIP、自监督 embedding、视觉语义和 aesthetic 模型；
   - 轻量分类器、特征融合、弱监督和无标签表示。

3. 相似度、检索与聚类
   - SHA、pHash/dHash/colorHash、SSIM 和 near-duplicate graph；
   - k-NN、same-scene grouping、micro/macro clustering；
   - hierarchical clustering、HDBSCAN、spherical k-means、k-medoids；
   - outlier、medoid、cluster representative 和 cluster stability。

4. 多模态视觉 API 与内容安全
   - 单图语义判断、成对比较、listwise reranking 和 contact-sheet judging；
   - screenshot、slide、CG、AI 图、构图、主体、展示价值和场景标签；
   - moderation、安全分类、调用成本、隐私、一致性和 fallback。

5. 排名与集合选择
   - MMR、DPP、facility location、submodular selection、配额和约束优化；
   - cluster coverage、family cap、场景/人物/构图比例；
   - quality floor、gallery cohesion、carousel 首屏和整体节奏。

6. 无标签评估与工程
   - synthetic/programmatic fixtures、metamorphic invariants；
   - clustering stability、方法分歧、重复残留和集合级多样性指标；
   - CPU/API 成本、batching、缓存、恢复、确定性、隐私和降级。
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


MEETING_INSTRUCTIONS = """\
=== 会议目标 ===

本次会议要探索、比较并论证适合官网 gallery 自动选图任务的技术方法，
为最终 Planner 提供充分、经过多轮讨论的技术观点。

不要预设最终方案必须是逐图阈值过滤、线性 staged pipeline、单一 composite
score 排序或任何其他固定架构。参与者从各自技术领域出发，提出认为有价值的
方法，解释其作用、限制、与其他技术的组合方式，以及什么证据会改变自己的判断。

最终采用的完整方案由专门 Planner 综合全部讨论后生成。

=== 讨论方式 ===

这是规划会议，不是执行会议。参与者可以：

- 自由提出技术观点、候选方法和替代方案；
- 解释多个技术如何组合；
- 描述候选数据流、关系或候选架构；
- 使用简短流程或伪代码帮助表达技术思想；
- 运行与当前主张强度相匹配的小规模实验；
- 回应、支持、修正或反驳其他参与者；
- 撤回自己先前不成立的建议。

参与者不要：

- 把自己的建议宣布为会议已经采用的最终 Plan；
- 代替 Planner 完整裁决所有技术争议；
- 展开最终工程目录、完整模块拆分、实施里程碑和运行手册；
- 因为自己代表某个技术领域，就强行要求最终方案使用该领域的组件。

每轮可以自由组织表达，不需要填写固定表格或结构化记录。请给出完整当前观点，
并在后续轮次说明相比上一轮有哪些新增、修正或撤回。

=== 技术覆盖 ===

会议结束前应充分考虑但不强制采用以下技术域：

- 传统图像处理、空间结构和图像取证；
- 视觉表征、轻量模型和语义特征；
- 重复检测、相似度、关系建模和聚类；
- 多模态视觉模型、外部视觉 API 和内容安全；
- 排名、集合选择、配额和全局优化；
- 无标签评估、成本、隐私、可靠性和降级。

如果某个技术域最终不值得进入方案，应说明其收益不足、证据不支持、与其他方法
重复、成本过高或违反约束，而不是在讨论中完全遗漏。

=== 证据原则 ===

当你提出依赖当前数据分布、运行时间、阈值、准确率、错误率或模型能力的经验性
结论时，应引用已有证据或运行适当的小型实验。不要把很小的样本升级成通用结论。

尚未验证的算法、架构和技术组合可以作为假设提出，但必须明确其证据状态。
没有人工 ground truth 时，只能声称测得稳定性、一致性、分歧、确定性事实或代理
指标，不得把它们表述为真实准确率。

允许验证的对象包括单图信号、模型、embedding、聚类、视觉关系、外部 API、
内容安全、排序和集合选择，不限于阈值或启发式。

=== 任务描述 ===

{participant_task_spec}

=== 图片筛选技术地图 ===

{technology_landscape}

=== Stage-1 证据全文 ===

{probing}
"""


PARTICIPANT_DISCUSSION_ADDENDUM = """\
This is a multi-round technical planning discussion. Contribute free-form technical
viewpoints from your assigned domain. You may describe candidate method combinations,
data flows, architecture sketches, or short pseudocode when they help explain an idea.

Do not claim that your candidate is the meeting's adopted final Plan, and do not write
the complete implementation project, module tree, milestones, or operator manual. The
dedicated Planner will make the final cross-domain decisions after the discussion.

You represent a technical knowledge domain, not a technology vendor. It is valid to
conclude that a technique from your own domain is unnecessary, redundant, too costly,
or unsupported. Explain material limitations and what evidence would change your view.

This meeting includes VisualAuditor, a participant that can actually open image files
and report what they show. If a claim in this discussion hinges on what a specific
image or set of images actually looks like and you want it checked, write a request
naming the exact file path(s) and your question to image_review_request.md in your
own shared subfolder. VisualAuditor checks every participant's shared subfolder for
this file each round.
"""


COMMON_PARTICIPANT_PROMPT = """\
You are {name}, a technical-domain participant in a multi-agent planning meeting.

Your domain:
{domain}

Contribute natural, thoughtful prose rather than filling a rigid schema. In every
round, explain your full current position, the techniques or combinations you find
useful, their important limitations, and how relevant evidence or other participants'
arguments affect your view.

You may discuss adjacent domains when necessary, but keep your reasoning anchored in
your own technical perspective. Do not impersonate or replace another participant.
Do not force techniques from your domain into the final solution merely to justify
your presence.

Write for a human reader, not a spec sheet. Explain each technique in a real clause
(what it's for, why it matters here), not as a bare comma/slash-separated term dump --
"perceptual hashes like pHash to catch resized or recompressed copies" reads far
better than "pHash/dHash/colorHash/SSIM". Prefer several shorter, plain sentences over
one long chain of semicolon-joined clauses. This applies every round, not just your
first -- it is easy to let later rounds compress back into term-dump shorthand once
the discussion gets long; resist that.
"""


DOMAIN_BRIEFS = {
    "ClassicalVision": """\
Classical image processing, spatial image structure, image forensics, and technical
quality. Consider global and local color statistics, connected uniform regions,
entropy maps, edge orientation, rectangular/grid layout, OCR/text layout, frequency
signals, blur, exposure, compression, and the distinction between a photograph and
webpage/slide/logo/UI/graphic content. Avoid assuming a single global scalar threshold
is sufficient; examine spatial relationships, combinations, lightweight learned
classifiers, and counterexamples such as sky, snow, fog, white walls, or embedded
photos in slides.""",
    "RepresentationModels": """\
Learned visual representations and local models under the <100M CPU constraint.
Consider CNN intermediate features, ImageNet classifiers, CLIP/MobileCLIP-style
embeddings, self-supervised representations, lightweight semantic/quality/aesthetic
models, feature fusion, and what these representations can or cannot express. Focus
on generalization, domain shift, preprocessing, calibration, and interfaces to
clustering, API routing, and final selection rather than assuming representation
quality alone solves the task.""",
    "SimilarityClustering": """\
Image-to-image relationships: exact and near duplicate detection, perceptual hashes,
SSIM, retrieval, k-NN graphs, same-scene grouping, micro/macro clustering, HDBSCAN,
hierarchical methods, spherical k-means, k-medoids, outliers, medoids, cluster
stability, and representative selection. Do not assume clustering must happen only
after all per-image filtering. Examine where relationship-first or cluster-first
reasoning improves on independent scalar scoring, and where clustering is unstable
or adds no value.""",
    "MultimodalVisionAPI": """\
External visual APIs and multimodal models used after local candidate reduction.
Consider real-photo verification, screenshot/slide/logo/CG/AI detection, scene and
subject understanding, composition and display appeal, pairwise/listwise comparison,
contact-sheet judging, harmful/sensitive content, and moderation. Treat API outputs
as model judgments rather than ground truth. Address candidate budgets, cost, privacy,
batch/order bias, nondeterminism, auditability, version drift, and pure-local fallback.""",
    "RankingSelection": """\
Final set construction rather than only per-image scoring. Consider MMR, DPP, facility
location, submodular selection, cluster quotas, constrained optimization, family and
scene caps, quality floors, representative coverage, person/subject balance, gallery
cohesion, carousel geometry, first-screen behavior, and deterministic tie-breaking.
Challenge the assumption that sorting one composite scalar necessarily produces the
best collection of 100 images.""",
    "SystemsEvaluation": """\
Engineering feasibility and evaluation without human labels. In round 1, perform or
coordinate a single shared survey of the current Windows CPU environment, relevant
installed libraries/models, network/API assumptions, and reusable Stage-1 evidence;
publish reusable findings under the meeting shared directory so other participants
need not repeat them. Across later rounds, evaluate CPU/API cost, memory, batching,
cache/recovery, privacy, determinism, synthetic and metamorphic tests, clustering
stability, method disagreement, set-level proxy metrics, and fallback behavior.
Do not define the overall architecture merely because one option is already installed.""",
}


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


PLANNER_SYSTEM_PROMPT = """\
You are Planner, the synthesis agent for a multi-round technical planning meeting.

Participants were organized by technology domain and deliberately wrote free-form
technical viewpoints rather than a shared structured plan. Your job is to read the
entire discussion, inspect relevant shared artifacts, resolve disagreements, and
produce one coherent, executable gallery-selection plan.

If the initial message only includes recent rounds and gives a path to the full
transcript, you MUST read the full transcript before finalizing. Do not treat the last
few rounds as the whole meeting.

Do not mechanically include every technique. For each important technical choice,
distinguish:

- adopted primary method;
- fallback or optional method;
- rejected method and why;
- unresolved capability limit.

The final plan must not default to a threshold-first linear pipeline merely because
that is easy to write. Explicitly consider the discussion on spatial image structure,
learned representations, clustering/relationship modeling, external visual APIs,
content safety, set-level/global selection, and no-label evaluation. Adopt or reject
each major direction based on the evidence and constraints discussed.

The production system must be fully automatic and must not require new human labels,
manual tuning, manual approval, or manual review to generate results. When the
discussion lacks ground truth, state the limitation and use only justified
deterministic, synthetic, metamorphic, stability, disagreement, or set-level proxy
evidence. Never convert an API output into fictional ground truth or claim unmeasured
accuracy.

Local models must remain below 100M parameters and run on Windows CPU-only. External
vision APIs are allowed only with a bounded late-stage candidate budget, cost/privacy
analysis, audit fields, and a pure-local fallback.

Write the plan for readability before implementation detail:

1. executive summary and selected architecture;
2. a compact architecture/data-flow diagram;
3. important alternatives and explicit decisions;
4. stage-by-stage inputs, outputs, purpose, and failure behavior;
5. relationship modeling, clustering, and final set-selection logic;
6. external API and safety behavior, if adopted;
7. no-label evaluation and acceptance tests;
8. cost, privacy, caching, recovery, and degradation;
9. implementation milestones and PowerShell execution;
10. known limits and risks;
11. detailed thresholds, formulas, manifest fields, and directory layout in appendices
    rather than allowing them to obscure the main architecture.

Within every section, write like a senior engineer explaining this design to a
colleague, not like you are filling out a spec template. That means:

- Lead with reasoning, not just conclusions. Don't just state what was chosen --
  show the thinking that got there: what the obvious first approach would be, why it
  falls short given the evidence or constraints, and what that implies about the
  approach you're adopting instead. A reader should be able to follow *why* each
  major choice is correct, not just accept that it was made.
- Write connected prose, not a list of disconnected facts stitched together with
  semicolons. "We tried X hoping for Y; it didn't hold up because Z, so we moved to
  W instead" reads as one continuous argument. "X. Y. Z. W." (or "X; Y; Z; W" as one
  run-on sentence) reads as notes, not an explanation -- avoid both.
  Use bullet points only for content that is genuinely list-shaped (e.g. a set of
  parallel options), not as a substitute for explaining how ideas connect.
- Explain each technique in a real clause (what it's for, why it was adopted or
  rejected here), not as a bare comma/slash-separated term dump -- "perceptual
  hashes like pHash to catch resized or recompressed copies" reads far better than
  "pHash/dHash/colorHash/SSIM".
- Don't repeat the same caveat or constraint verbatim in every section just to be
  safe -- state it clearly once, where it's most load-bearing, and trust the reader
  to carry it forward.
- Push genuinely dense reference material (exact thresholds, formulas, manifest
  fields, directory layouts) into the appendices per item 11 above, so the main
  sections stay readable as an argument rather than a lookup table. But do not use
  the appendices as an excuse to leave the main sections' own reasoning thin --
  the appendix holds the *values*, the main section still owns the *why*.

The plan must be detailed enough for an Executor to implement without making new
methodology-level decisions. Save it as `final_gallery_selection_plan_technical_domains.md`
in your workspace using file tools, then respond with the same content.
"""


def build_question() -> str:
    return MEETING_INSTRUCTIONS.format(
        participant_task_spec=PARTICIPANT_TASK_SPEC,
        technology_landscape=TECHNOLOGY_LANDSCAPE,
        probing=build_recon_guidance(),
    )


def make_domain_participant(name: str) -> ParticipantConfig:
    return ParticipantConfig(
        name=name,
        role=f"Technology domain: {name}",
        system_prompt=COMMON_PARTICIPANT_PROMPT.format(
            name=name,
            domain=DOMAIN_BRIEFS[name],
        ),
        model="deepseek-v4-flash",
        provider="deepseek",
        reasoning_effort="high",
        max_iterations=12,
    )


def parse_cli() -> tuple[str | None, int | None, bool]:
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]

    extra_rounds = None
    if "--extra-rounds" in sys.argv:
        extra_rounds = int(sys.argv[sys.argv.index("--extra-rounds") + 1])

    planner_only = "--planner-only" in sys.argv
    if planner_only and not resume:
        raise SystemExit("--planner-only requires --resume <meeting_id>")
    if planner_only and extra_rounds is not None:
        raise SystemExit("--planner-only cannot be combined with --extra-rounds")

    return resume, extra_rounds, planner_only


def main() -> None:
    resume, extra_rounds, planner_only = parse_cli()

    participants = [
        make_domain_participant("ClassicalVision"),
        make_domain_participant("RepresentationModels"),
        make_domain_participant("SimilarityClustering"),
        make_domain_participant("MultimodalVisionAPI"),
        make_domain_participant("RankingSelection"),
        make_domain_participant("SystemsEvaluation"),
        ParticipantConfig(
            name="VisualAuditor",
            role="Grounds other participants' visual claims by actually opening images",
            role_ref="visual-evidence-auditor",
            meeting_brief=VISUAL_AUDITOR_MEETING_BRIEF,
            # Must be codex, not deepseek -- this is the only provider path in this
            # codebase that can actually deliver image content to the model
            # (research_agent's Responses API image support + view_image tool).
            model="gpt-5.5",
            provider="codex",
            reasoning_effort="high",
            max_iterations=16,
            # The only participant here with view_image in its tool registry -- every
            # other participant defaults to vision_capable=False, so even if one of
            # them tried to call it, it simply wouldn't be offered.
            vision_capable=True,
        ),
    ]

    config = MeetingConfig(
        question=build_question(),
        mode="planning_rounds",
        max_rounds=5,
        participants=participants,
        planner_inline_rounds=None,
        planning_participant_addendum=PARTICIPANT_DISCUSSION_ADDENDUM,
        planner=PlannerConfig(
            name="Planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            max_iterations=20,
        ),
    )

    if planner_only:
        checkpoint = load_meeting(resume)
        if checkpoint.get("mode") != "planning_rounds":
            raise SystemExit(
                f"--planner-only requires a planning_rounds meeting; "
                f"{resume} has mode={checkpoint.get('mode')!r}"
            )
        if checkpoint.get("status") != "in_progress":
            raise SystemExit(
                f"--planner-only requires an in-progress meeting; "
                f"{resume} has status={checkpoint.get('status')!r}"
            )

        completed_rounds = sum(
            1 for step in checkpoint.get("steps", [])
            if step.get("decided_by") == "judge"
        )
        if completed_rounds < 1:
            raise SystemExit(
                f"--planner-only found no fully checkpointed discussion rounds in {resume}"
            )

        # run_meeting() resumes at completed_rounds + 1. Making max_rounds equal
        # to the number already completed leaves that range empty, so
        # _run_planning_rounds() proceeds directly to _run_planner_step() while
        # preserving every checkpointed round in the Planner's transcript.
        config.max_rounds = completed_rounds
        print(
            f"[planner-only] {resume}: using {completed_rounds} completed round(s); "
            "skipping unfinished/later rounds and starting Planner"
        )

    result = run_meeting(config, resume=resume, extra_rounds=extra_rounds)

    print(f"\nmeeting_id: {result['meeting_id']}")
    print(f"saved to: runs/{result['meeting_id']}.json")

    for step in result["steps"]:
        if step["decided_by"] == "judge":
            turn = step["turns"][0]
            verdict = "STOP" if turn.get("stop") else "continue"
            print(
                f"\n{'=' * 20} JUDGE after round {turn['round']}: "
                f"{verdict} {'=' * 20}"
            )
            print(f"reasoning: {turn.get('reasoning', '')}")
            if turn.get("unresolved_issues"):
                print(f"unresolved_issues: {turn['unresolved_issues']}")
            if turn.get("override_reason"):
                print(f"OVERRIDE: {turn['override_reason']}")
            continue

        if step["decided_by"] == "planner":
            continue

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
