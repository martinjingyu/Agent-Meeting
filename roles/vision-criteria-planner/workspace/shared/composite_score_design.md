# 纯启发式方案：实景可信度 / 质量 / 展示适配 三类复合分的具体化设计

> 面向 Executor 的最终设计说明。回答：分怎么算、为什么不设死阈值、边界场景怎么保护、Executor 怎么在结果中解释判断。

---

## 1. 三类复合分的定位差别

| 维度 | 范围 | 核心问题 | 过线后参与排序？ | 过线前怎么处理？ |
|------|------|----------|-----------------|-----------------|
| **S_realism** (实景可信度) | 0–100 | "这是真实相机拍的吗？" | 必须 ≥50 才进入候选 | <30→淘汰；30–49→复核池 |
| **S_quality** (展示质量) | 0–100 | "这张图清晰/曝光/构图够好吗？" | 必须 ≥40 才参与排序 | <30→淘汰；30–39→复核池 |
| **S_display** (展示适配性) | 0–100 | "这张图适合官网 gallery 展示吗？" | 仅影响排序权重，**不淘汰** | 始终不单独淘汰 |

**关键原则：** S_display 只在 S_realism ≥ 50 **且** S_quality ≥ 40 之后才参与排名。不存在"展示不佳所以淘汰"——只存在"展示不佳所以排名靠后"。

---

## 2. 分层决策逻辑（不是单阈值，是信号组合 + 层叠判断）

### 2.1 整体架构：三层渐进

```
第1层（Stage A 早筛）：固定阈值硬拒绝
  只杀死"100%不是"的情况
  例：min_side < 64 → 缩略图；sharpness < 2.0 → 极度模糊
  第1层不过的 → 淘汰，不进后续

第2层（Stage A/B 交界）：自适应软过滤
  按数据集分布调整阈值，标记低置信度但不杀死
  例：该数据集 sharpness P25 = 8.0 → 阈值 = max(3.0, 8.0×0.4) = 3.2，低于3.2的标记 LOW_CONFIDENCE
  第2层不过的 → 标记 LOW_CONFIDENCE → 不进 top-100，但进复核池

第3层（Stage B）：复合分计算
  对幸存者算三维分，但：
  - 不含任何"一票否决"逻辑
  - 边界场景只降分不拒绝
  - 极端值做降级而非硬切
```

### 2.2 分是怎么算的

**S_realism（实景可信度）—— 基于"自然场景信号适中度"**

核心思想：真实场景的信号值落在适中范围；极端值（太高或太低）都提示非实景。

```
S_realism = 50（基线，对应 AMBIGUOUS）

=== 加分信号（同时出现才生效）===
if 0.02 ≤ edge_ratio ≤ 0.15  AND  10 ≤ colorfulness ≤ 50：
    S_realism += 20    # 自然纹理 + 自然色彩 = 强实景信号
if entropy > 6.0：
    S_realism += 10    # 信息丰富
if 60 ≤ brightness_mean ≤ 200  AND  brightness_std > 25：
    S_realism += 10    # 自然光照变化
if sharpness 在 60–400 之间：
    S_realism += 10    # 清晰但不异常

=== 降分信号（各自独立触发）===
if edge_ratio < 0.005：
    if colorfulness < 5：S_realism -= 40  # 纯色占位
    else：               S_realism -= 20  # 可能CGI平滑
if edge_ratio > 0.35  AND  colorfulness > 70：
    S_realism -= 40    # 彩色UI/截图
if edge_ratio > 0.40  AND  entropy < 4.0：
    S_realism -= 30    # 文档/信息图
if colorfulness > 80：
    S_realism -= 15    # 过饱和渲染
if brightness_mean < 20  OR  brightness_mean > 230：
    S_realism -= 25    # 过曝/欠曝
if sharpness > 600：
    S_realism -= 10    # 过度锐化（可能AI/渲染）

最终：clamp(0, 100)
```

**S_quality（展示质量）—— 基于"摄影参数舒适度"**

核心思想：清晰、色彩适中、对比自然、构图均衡 = 高质量。

```
S_quality = 50（基线）

if 60 ≤ sharpness ≤ 400：        S_quality += 15
elif sharpness < 20：             S_quality -= 20
elif sharpness > 600：            S_quality -= 5

if 15 ≤ colorfulness ≤ 50：      S_quality += 10
elif colorfulness < 5：           S_quality -= 15
elif colorfulness > 70：          S_quality -= 10

if 30 ≤ brightness_std ≤ 70：     S_quality += 10
elif brightness_std < 15：        S_quality -= 15  # 雾感/无层次

if 6.0 ≤ entropy ≤ 7.5：          S_quality += 10
elif entropy < 4.0：              S_quality -= 15

if horizontal_balance < 0.15：    S_quality += 10  # 构图稳
elif horizontal_balance > 0.35：  S_quality -= 10  # 构图明显偏

最终：clamp(0, 100)
```

**S_display（展示适配性）—— 基于"场景类型偏好"**

核心思想：优先展示空间/环境/群体，降权单人近景/产品白底/纯记录式。

```
S_display = 50（基线）

=== 加分项（可叠加）===
if 检测到 landscape（低edge + 中colorfulness + 宽幅）：   S_display += 15
if 检测到 architecture（高edge + 低color + 垂直边缘多）：  S_display += 12
if 检测到 interior（中edge + 中亮度 + 中低色彩）：        S_display += 10
if 检测到 group_activity（face_count ≥ 3）：               S_display += 15
if 检测到 urban/cityscape（高edge + 高熵 + 高对比）：      S_display += 10

=== 减分项（各自独立）===
if 检测到 single_portrait（脸面积>10% 且 face_count==1）：  S_display -= 20
if 检测到 product_packshot（低edge + 低std + 低color）：   S_display -= 15
if 检测到 selfie_like（单人近景 + 低quality + 低edge）：   S_display -= 20
if 检测到 documentary_only（构图偏 + 低quality）：          S_display -= 10

# 质量加成（展示适配感来自质量高的图）
S_display += S_quality * 0.15
# 实景加成（确认是实景的图展示更放心）
S_display += S_realism * 0.10

最终：clamp(0, 100)
```

---

## 3. 哪些信号只能做辅助、不能做硬拒绝

| 信号 | 可以做 | 绝对不能做 |
|------|--------|-----------|
| **sharpness** | 极度模糊 (< 2.0) 硬拒绝 | 单独用 sharpness 判断"非实景"——截图 sharpness 8.1 和实景 6.9 完全重叠 |
| **colorfulness** | >80 时降 S_realism；<5 且 edge 也低时硬拒绝纯色块 | **单独用 colorfulness < 5 拒绝一张图**——白墙室内照 colorfulness=5.9 |
| **edge_ratio** | >0.4+低熵硬拒绝文档；<0.005+低彩硬拒绝纯色 | 单独用 edge_ratio 拒绝——实景 edge_ratio 范围 0.05–0.25，与截图 0.15–0.50 有重叠区 |
| **brightness_std** | <15 时降 S_quality（雾感/无层次） | 硬拒绝"低对比度图"——真实雾景、雪景、清晨校园 |
| **face_count** | 群像加分、单人肖像降 S_display | 作为淘汰依据（Haar Cascade 不可靠，且人像并非"不要"而是"少要"） |
| **skin_ratio** | 单人近景提示（弱信号） | 硬拒绝——浅色墙/雕塑会假阳性 |
| **entropy** | <3.0 且 brightness_std < 10 硬拒绝纯色占位 | 单独用 entropy 拒绝——真实低纹理场景（雪地、天空、白墙） |

---

## 4. 高风险边界情况的保守处理原则

### 4.1 低饱和室内（白墙教室、走廊、实验室）

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| colorfulness=5–10（低）, edge_ratio=0.01–0.08（低至中）, entropy=5.5–6.5（中） | colorfulness 低 → 被误判为文档，edge_ratio 低 → 被误判为纯色 | **核心保护：** edge_ratio > 0.01 时 colorfulness 阈值自动放松到 2.0（在自适应阈值中已实现）。S_realism 中 edge_ratio 在 0.02–0.15 范围内加分 20（不依赖 colorfulness）。 |
| **结果** | 不会被 R5（文档规则）误杀，也不会被纯色规则误杀。S_realism 通常在 50–70（PROBABLY_REAL），S_quality 受 sharpness 影响可能 30–50（FAIR）。最终：复核池或候选但低排名。 |

### 4.2 雾景 / 雪景

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| brightness_mean=150–220（高亮）, brightness_std=10–25（低对比）, colorfulness=3–15（低彩）, edge_ratio=0.01–0.06（低） | brightness_std 低 → 被误判为"雾感无层次"而降太多分；colorfulness 低 → 被误判为文档 | **核心保护：** brightness_std < 20 仅降 S_quality（-15）但不拒绝。colorfulness < 8 且 edge_ratio > 0.015 时触发保护规则，colorfulness 拒绝阈值自动放松到 2.0。S_realism 按 edge_ratio 正常计算。 |
| **结果** | S_realism 通常在 50–70（可接受），S_quality 可能在 30–45（偏低）。真实雾景/雪景如果构图好（horizontal_balance 中），S_display 可获 landscape 加分。最终：大概率进复核池或靠后入选。 |

### 4.3 实验室 / 暗室内景

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| brightness_mean=20–60（暗）, sharpness=5–30（低至中）, colorfulness=5–20（低至中）, edge_ratio=0.02–0.15（正常） | brightness_mean 低 → 被误判为过曝/欠曝；sharpness 低 → 被误判为模糊 | **核心保护：** brightness_mean < 40 且 edge_ratio > 0.02 时触发保护规则，仅降 S_quality（-5）而非拒绝。sharpness 用自适应阈值（数据集 P25×0.4 而非全局固定值），暗光数据集自动宽松。 |
| **结果** | S_realism 通常 60–80（可确认实景），S_quality 可能 25–45。最终：如果 dataset 整体偏暗，自适应阈值会宽松；否则进复核池。 |

### 4.4 舞台暗光 / 聚光灯场景

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| brightness_mean=20–60（暗）, brightness_std=40–80（高对比=聚光灯效果）, sharpness=10–50（中等）, edge_ratio=0.05–0.20（正常） | brightness_mean 低 + brightness_std 高 → 信号组合独特，容易被误判为"异常" | **核心保护：** brightness_std > 30 时"低亮度"根本不触发降低（因为自然光照变化大）。S_realism 按正常加分（edge_ratio 适中 + entropy 高）。 |
| **结果** | S_realism 60–80，S_quality 受 sharpness 影响。如果 sharpness > 20，S_quality 可到 50+。正常入选。 |

### 4.5 电影剧照 / 活动摄影

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| 所有信号都在"优秀实景"范围，与真实照片无法区分 | 无法区分"剧照"和"实景"——但剧照本身就是真实相机拍摄的 | **不做特殊处理。** 电影/活动剧照 = 真实相机拍摄。如果有版权/品牌问题，那是法律合规审查的范围，本 pipeline 不负责。 |
| **结果** | 正常入选。 |

### 4.6 CGI / 高质量渲染

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| sharpness=100–500（高）, colorfulness=30–80（高）, edge_ratio=0.03–0.15（自然）, entropy=6.0–7.5（丰富） | 所有信号与真实照片几乎相同，这是 CPU-only 启发式的**已知盲区** | **不依赖启发式区分。** 策略：(1) 数字域数据集标记为 TYPE_B_CGI_HIGH_RISK；(2) S_realism 计算时额外 -15 偏移（审慎偏置）；(3) 所有候选自动标注 "CGI_risk" 标志；(4) 全部复核后人工终审。 |
| **结果** | 不会在 pipeline 中被自动拒绝，但会被标记为风险。人工决定。 |

### 4.7 图文截图混合（截图内嵌照片、PDF 中的图片）

| 信号表现 | 风险 | 处理原则 |
|----------|------|----------|
| 整体 edge_ratio=0.20–0.50（高，来自 UI 边框/文字），内部有照片区域 | 规则可能拒掉整个截图（正确行为），但如果 edge_ratio 刚好在边界（0.25–0.35）可能漏过 | **edge_ratio > 0.30 且 colorfulness > 60 → NON_REAL（R1 规则）。** edge_ratio 0.25–0.30 且 entropy > 5.5 的 → 标记 AMBIGUOUS 进复核池。整体的 edge_ratio 高就是截图/信息图的证据，不因为"里面有一张照片"而赦免。 |
| **结果** | edge_ratio > 0.30 拒绝；0.25–0.30 进复核池。只有整个图像本身是照片（低 edge_ratio）才通过。 |

---

## 5. Executor 在结果中如何解释判断

### 5.1 输出字段设计（每张幸存图片）

每张幸存图片输出一个 JSON 记录，包含完整的判断链和人工可读理由：

```json
{
  "filepath": "truro_school/IMG_4521.jpg",
  "signals": {
    "sharpness": 124.3,
    "edge_ratio": 0.032,
    "colorfulness": 28.5,
    "entropy": 6.8,
    "brightness_mean": 128.0,
    "brightness_std": 45.2,
    "aspect_ratio": 1.33,
    "min_side": 480,
    "face_ratio_area": 0.0,
    "face_count": 0,
    "horizontal_balance": 0.08
  },
  "scores": {
    "S_realism": 85,
    "S_quality": 72,
    "S_display": 68,
    "S_final": 71.4
  },
  "realism_label": "REAL",
  "quality_label": "GOOD",
  "display_labels": ["architecture_exterior", "urban_cityscape"],
  "selected": true,
  "selected_reason": "S_final=71.4 (高); REAL+GOOD 过线; architecture_exterior 加分; 簇5配额未满 (3/15)",
  "not_selected_reason": null,
  "boundary_notes": [],
  "cgi_risk": false,
  "human_review_priority": "low"
}
```

被淘汰图片：

```json
{
  "filepath": "roland_berger/chart_042.png",
  "scores": {
    "S_realism": 22,
    "S_quality": 35,
    "S_display": 18
  },
  "selected": false,
  "selected_reason": null,
  "not_selected_reason": "S_realism=22 (NON_REAL, <30淘汰); 具体: edge_ratio=0.42+colorfulness=5.1+entropy=3.8 → 文档/信息图",
  "boundary_notes": ["low_color_but_natural: false"],
  "cgi_risk": false,
  "human_review_priority": "none"
}
```

### 5.2 标准化判断解释编码

| 场景 | 在结果中怎么写 | 示例 |
|------|--------------|------|
| 低饱和室内被保护 | `"boundary_notes": ["low_color_but_natural: edge_ratio=0.025 正常, 保护激活"]` | 不会出现在淘汰/拒绝理由中 |
| 雾景被降质量分 | `"not_selected_reason": "S_quality=38 (<40复核池); brightness_std=18 低对比→降15分"` | 明确告知降分来源 |
| 单人近景被降展示分 | `"not_selected_reason": "S_display=24 (-20单人近景惩罚); 簇7配额未满但与cluster_pruned同场景"` | 告知是配额/惩罚双因素 |
| CGI 风险标记 | `"cgi_risk": true, "selected_reason": "S_final=65.2; 但cgi_risk=true → 人工复核确认"` | 入选但标注风险 |
| 图文截图被拒绝 | `"not_selected_reason": "R1触发: edge_ratio=0.38>0.35 AND colorfulness=82>70 → 彩色UI/截图"` | 直接引用规则 ID |
| 复核池进入原因 | `"not_selected_reason": "AMBIGUOUS(realism=48) + GOOD(quality=55) → 复核池"` | 按照判定矩阵 |

### 5.3 聚合解释（数据集级别统计）

在 `aggregate_stats.json` 中提供：

```json
{
  "truro_school": {
    "total": 36266,
    "hard_rejected": 8240,
    "low_confidence": 5480,
    "survivors": 22546,
    "review_pool": 1240,
    "top100_selected": 100,
    "top100_face_ratio": 0.12,
    "top100_display_labels": {
      "landscape": 28,
      "architecture_exterior": 22,
      "interior_space": 18,
      "group_activity": 15,
      "urban_cityscape": 12,
      "single_portrait": 5
    },
    "boundary_cases_triggered": {
      "low_color_but_natural": 342,
      "low_contrast_hazy": 89,
      "low_light_indoor": 156
    },
    "failure_warnings": []
  }
}
```

这样人工审查者可以：
1. 一眼看出每数据集 top-100 的构成（场景类型分布）
2. 确认边界保护规则是否被触发（如 low_color_but_natural 触发了多少次）
3. 如果 face_ratio 过高（>0.5），知道需要检查肖像偏见过重
4. 直接看到每个具体判断的理由编码

---

## 6. 总结：Executor 要实现的核心

| 需求 | 实现方式 |
|------|---------|
| 三维复合分 | 三层可解释公式（见 §2.2），每层都有 clamp 和明确信号组合 |
| 非单阈值判断 | 层叠逻辑：硬拒绝（固定）→ 软过滤（自适应）→ 复合分（求和）|
| 边界场景保护 | 4 条保护规则（§4），在自适应阈值和 S_realism 计算中嵌入 |
| 判断可解释 | 每张图输出完整信号、分数、selected_reason/not_selected_reason（§5.1）|
| 整体可控 | 聚合统计（§5.3）+ 人工复核池 + CGI 风险标记 |
