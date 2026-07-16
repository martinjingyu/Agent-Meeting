# 视觉判断逻辑框架：实景图 / 非实景图 / 适合 Gallery / 不适合展示

> 本框架基于 Stage 1 跨 11 数据集、220+ 样本的实证验证。所有判据仅依赖图像视觉内容，使用 CPU-only 可计算的启发式信号。优先保证可解释性（每条判断都有具体视觉证据）和保守偏置（边界样本进入复核池而非自动入选）。

---

## 目录

1. [核心判断轴与评分维度](#1-核心判断轴与评分维度)
2. [实景 vs 非实景 — 5 级分类体系](#2-实景-vs-非实景--5-级分类体系)
3. [Gallery 展示质量 — 4 级质量体系](#3-gallery-展示质量--4-级质量体系)
4. [最终适配性判定矩阵](#4-最终适配性判定矩阵)
5. [边界样本处理规则（10 条，含审计依据）](#5-边界样本处理规则)
6. [硬拒绝 / 低可信 / 待复核实例](#6-硬拒绝--低可信--待复核实例)
7. [已知失效模式与不可解决的问题](#7-已知失效模式)
8. [性能预算与工程落地建议](#8-性能预算)

---

## 1. 核心判断轴与评分维度

每个图像在 **三个独立轴** 上评估，每个轴产生一个标签 + 视觉证据字符串。

### 1.1 轴 A：实景可信度（Real-Scene Trustworthiness）

判断图像是否由真实相机在物理世界中拍摄。

| 维度 | 计算方法 | 实景典型范围 | 非实景典型范围 | 分离能力 |
|------|----------|-------------|----------------|----------|
| **Edge Ratio** | Canny 边缘像素数 / 总像素数 | 0.05–0.25 | 0.15–0.50 | **最强单一信号** |
| **Colorfulness** | Hasler-Susstrunk 指标 | 8–80 | 5–140 | **单独使用会反向误导** |
| **Image Entropy** | 灰度直方图信息熵 | 6.0–7.8 | 3.0–7.5 | 低熵时强信号 |
| **Brightness Std** | 像素亮度帧内标准差 | 40–80 | 10–40（文档）或 60–120（图形） | 辅助 |
| **Aspect Ratio** | 宽/高 | 0.5–3.0 | <0.25 或 >4.0 | 极值时强信号 |
| **Sharpness** | Sobel 梯度均值 (@640px) | 3.0–100+ | 2.0–150+ | **无法单独分离** |

### 1.2 轴 B：展示质量（Display Quality）

判断图像是否达到官网 gallery 展示的清晰度、曝光、色彩和尺寸标准。

| 维度 | 计算方法 | 说明 |
|------|----------|------|
| **Sharpness** | Sobel 梯度均值 @640px | 使用数据集自适应百分位阈值 |
| **Colorfulness** | Hasler-Susstrunk | 区分"低彩但自然"和"低彩且非实景" |
| **Exposure** | 亮度均值 (0-255) | 识别过曝/欠曝 |
| **Resolution** | 最短边像素数 | 淘汰过小图像 |
| **Entropy** | 灰度信息熵 | 识别空白/占位图 |

### 1.3 轴 C：Gallery 展示适配性（Gallery Suitability）

最终决策：轴 A + 轴 B 的交叉判定矩阵。

---

## 2. 实景 vs 非实景 — 5 级分类体系

### 2.1 Level 1: NON_REAL（确定非实景）— **硬拒绝**

判断依据是以下任一规则命中。每条规则对应具体视觉证据，可直接审计。

| 规则 ID | 条件 | 视觉证据 | 捕获的对象 |
|---------|------|----------|-----------|
| **R1** | `edge_ratio > 0.35 AND colorfulness > 70` | 高边缘密度 + 高饱和度 = 彩色 UI/图形 | 彩色截图、信息图、UI 界面 |
| **R2** | `aspect_ratio < 0.25 OR aspect_ratio > 4.0` | 极端宽/高比 | 横幅广告、极窄条幅 |
| **R3** | `entropy < 3.5 AND brightness_std < 15` | 近均匀色块 | 纯色占位图、空白图、损毁图 |
| **R4** | `sharpness_640 < 2.0` | 极度模糊，无可用内容 | 严重失焦/运动模糊图 |
| **R5** | `colorfulness < 4 AND edge_ratio > 0.20` | 低色彩 + 高边缘 = 文字文档 | 白底文档、扫描件、信息图表 |
| **R6** | `shortest_side < 150px` | 物理尺寸过小 | 缩略图、头像、图标 |

> **为什么 R1 而不是 colorfulness 单阈值？** 实证数据：kpmg 截图 colorfulness 中位=60.0，比 m_immobilier 实景（中位=29.2）更高。colorfulness 单阈值会误杀白墙室内照而放过彩色截图。只有 edge_ratio + colorfulness 组合才能正确分离。

### 2.2 Level 2: PROBABLY_NON_REAL（可能非实景）— **低可信 → 复核池**

判断依据：不满足 Level 1 的硬阈值，但复合软分 < 0.25。

**复合软分公式：**
```
composite = 0.25 × colorfulness_score + 0.25 × entropy_score 
          + 0.25 × brightness_std_score + 0.25 × (1 - min(edge_ratio/0.3, 1.0))
```

各分量归一化方法：
- `colorfulness_score`: 线性映射，20→0.0, 60→1.0，截断到 [0,1]
- `entropy_score`: 线性映射，4.0→0.0, 7.0→1.0，截断到 [0,1]
- `brightness_std_score`: 线性映射，10→0.0, 50→1.0，截断到 [0,1]
- `edge_ratio_penalty` = 1 - min(edge_ratio/0.3, 1.0)

**处理：** 进入复核池，不自动淘汰（可能误判）。对于 `roland_berger`、`tuv_rheinland` 等含大量信息图的数据集，此级别占比可能高达 30-40%。

### 2.3 Level 3: AMBIGUOUS（不确定）— **待复核 + 可能需模型辅助**

判断依据：复合软分 0.25–0.40，或在 0.40–0.55 但某个信号异常。

典型场景：
- 真实背景 + 文字覆盖（如海报中的照片）
- 高质感 3D 渲染（`digital_domain` 常见）
- 严重滤镜/过度后期照片
- 低光/雾霾场景（colorfulness 和 sharpness 同时偏低）

**处理：** 全部进入复核池。对于 `digital_domain` 数据集，此级别图像需额外标注"CGI risk"。

### 2.4 Level 4: PROBABLY_REAL（可能是实景）— **低可信 → 可候选**

判断依据：复合软分 0.40–0.55，所有信号在正常范围内。

典型场景：
- 轻度编辑的真实照片
- 低光室内场景（如 m_immobilier 的室内照，sharpness 可低至 3.0）
- 裁剪后的照片

**处理：** 进入 gallery 候选列表，但标注为"需复核"。最终排序时权重低于 REAL 级别。

### 2.5 Level 5: REAL（确定实景）— **正常候选**

判断依据：复合软分 ≥ 0.55，且同时满足：
- `colorfulness ≥ 8`（排除文档）
- `edge_ratio < 0.35`（排除 UI）
- `entropy ≥ 4.5`（排除纯色）
- `sharpness_640 ≥ max(2.0, dataset_p2_sharpness)`（排除极度模糊）

**处理：** 正常进入 gallery 候选排序。

---

## 3. Gallery 展示质量 — 4 级质量体系

### 3.1 质量等级定义

| 等级 | Sharpness @640px | Colorfulness | 亮度均值 | 最短边 | 视觉证据 |
|------|-----------------|-------------|---------|--------|---------|
| **EXCELLENT** | > 30.0 | ≥ 20 | 60–220 | ≥ 800px | 细节清晰，色彩自然，曝光正常 |
| **GOOD** | 15.0–30.0 | 8–20 | 40–60 或 220–240 | 500–800px | 略微柔和/饱和度偏低但可接受 |
| **FAIR** | 8.0–15.0 | 3–8 | < 30 或 > 240 | 300–500px | 明显模糊/色彩问题 → 复核 |
| **POOR** | < max(2.0, dataset_p2) | < 3 | < 20 或 > 250 | < 200px | 不可用 → 硬拒绝 |

### 3.2 数据集自适应阈值说明

**为什么不能用全局阈值？** 实证数据：
- m_immobilier（室内房产照）p5 sharpness = 3.0
- maior_capital（高端外景）p5 sharpness = 22.9
- 同一指标在不同数据集间差异达 **7×以上**

**做法：**
1. 先用所有图像计算该数据集的 sharpness 分布
2. 取 p2（第 2 百分位）和 p5（第 5 百分位）
3. POOR 阈值为 `max(2.0, dataset_p2_sharpness)`
4. FAIR 阈值为 `max(5.0, dataset_p5_sharpness)`

### 3.3 复合质量分

```
Quality = 0.40 × sharpness_norm + 0.25 × colorfulness_norm 
        + 0.15 × brightness_norm + 0.10 × size_norm + 0.10 × entropy_norm
```

各分量基于数据集内部分布归一化到 [0,1]。

---

## 4. 最终适配性判定矩阵

| 质量 \ 实景等级 | REAL | PROBABLY_REAL | AMBIGUOUS | PROBABLY_NON_REAL | NON_REAL |
|----------------|------|---------------|-----------|-------------------|----------|
| **EXCELLENT** | ✅ GALLERY | ✅ GALLERY | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **GOOD** | ✅ GALLERY | ✅ GALLERY | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **FAIR** | ⚠️ REVIEW | ⚠️ REVIEW | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **POOR** | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT | ❌ REJECT | ❌ REJECT |

### 额外排序偏好（不淘汰，仅影响排序）

1. **场景类型优先**：风景/建筑/室内环境 > 单人肖像 > 特写/微距
2. **群体优先**：多人活动 > 单人 > 无人
3. **自然光照优先**：brightness_std > 30（自然光照变化）> 平光
4. **多样性优先**：同一 dhash 簇只保留质量最高的 1 张
5. **肖像配额**：最终 100 张中单人肖像 ≤ 10 张

---

## 5. 边界样本处理规则

每条规则包含：边界情况描述、视觉证据、操作、为什么这样处理。

### B1: 高质感 3D 渲染（如建筑设计可视化）

| 字段 | 内容 |
|------|------|
| **视觉证据** | 高 sharpness、丰富色彩、完美曝光 — 但纹理过于均匀，缺乏传感器噪点，边缘过于完美 |
| **操作** | 无法通过启发式单独区分。标记 `digital_domain` 为高风险数据集。所有候选自动标注 "CGI risk" 标志 |
| **为什么** | 这是 CPU-only 启发式的已知盲区。必须人工审核 |
| **示例** | `digital_domain` 中的 CGI 建筑渲染图 |

### B2: 严重滤镜 / HDR 照片

| 字段 | 内容 |
|------|------|
| **视觉证据** | colorfulness > 100（超出自然范围）AND edge_ratio < 0.10（过度平滑） |
| **操作** | 降级一级：EXCELLENT→GOOD→FAIR→REVIEW |
| **为什么** | colorfulness > 100 在实景中极为罕见（kpmg 截图可达 140+，但实景很少超过 80） |

### B3: AI 生成图像

| 字段 | 内容 |
|------|------|
| **视觉证据** | 完美 sharpness、超现实光照、不可能几何、过度平滑纹理 |
| **操作** | 无法可靠检测。所有指标 > 90th 百分位的图像标记为 "potential AI" |
| **为什么** | 启发式+小模型的公认盲区。需人工审核 |

### B4: 屏幕截图中的照片

| 字段 | 内容 |
|------|------|
| **视觉证据** | 整体 edge_ratio 高（来自 UI 边框和文字），但内部有照片区域 |
| **操作** | edge_ratio > 0.30 → NON_REAL。整个图像是截图，不是照片 |
| **为什么** | 任务要求图像本身是照片，不是"包含照片的图像" |

### B5: 文档 / 信息图扫描

| 字段 | 内容 |
|------|------|
| **视觉证据** | colorfulness < 12 AND edge_ratio > 0.20 → 白底文字/图表 |
| **操作** | R5 规则捕获大部分。补充：colorfulness < 12 AND edge_ratio > 0.20 → NON_REAL |
| **为什么** | 文档有特征性高边缘密度+低色彩模式 |

### B6: 小尺寸缩略图 / 头像

| 字段 | 内容 |
|------|------|
| **视觉证据** | 最短边 < 150px |
| **操作** | NON_REAL（gallery 不适用） |
| **为什么** | 放大后像素化，不适合展示 |

### B7: 近似重复图像

| 字段 | 内容 |
|------|------|
| **检测方法** | dhash（8×9→64bit），Hamming 距离 ≤ 4 视为同簇 |
| **操作** | 每簇只保留质量最高的 1 张 |
| **为什么** | gallery 不能出现 5 张几乎相同的照片 |

### B8: 单人肖像 / 活动照

| 字段 | 内容 |
|------|------|
| **视觉证据** | aspect_ratio 接近 1:1 或 3:4，中央区域高 sharpness，背景模糊 |
| **操作** | 排序降权。最终 100 张中 ≤ 10 张 |
| **为什么** | 企业官网 gallery 应优先展示场景和团队活动 |

### B9: 损毁 / 无法读取文件

| 字段 | 内容 |
|------|------|
| **视觉证据** | Pillow 打开报错（truncated header, unsupported format 等） |
| **操作** | 记录到 `read_errors.log`（含文件名和错误信息），不静默跳过 |
| **为什么** | 任务明确要求日志记录而不是忽略 |

### B10: 低熵占位图

| 字段 | 内容 |
|------|------|
| **视觉证据** | entropy < 3.0，brightness_std < 10 |
| **操作** | NON_REAL |
| **为什么** | 没有有意义的视觉内容 |

---

## 6. 硬拒绝 / 低可信 / 待复核实例

### 6.1 应作为硬拒绝的样本

| 类别 | 视觉特征 | 对应规则 | 预期捕获 |
|------|---------|---------|---------|
| 彩色 UI 截图（kpmg_forensic） | edge_ratio > 0.35, colorfulness > 70 | R1 | ~95% 截图 |
| 白底信息图（roland_berger） | colorfulness < 4, edge_ratio > 0.20 | R5 | ~80% 信息图 |
| 纯色占位图 | entropy < 3.5, brightness_std < 15 | R3 | ~99% |
| 极端横幅/条幅 | aspect_ratio > 4.0 或 < 0.25 | R2 | ~99% |
| 缩略图/头像 | shortest_side < 150px | R6 | ~99% |

### 6.2 应作为低可信 / 待复核的样本

| 类别 | 视觉特征 | 预期级别 | 建议处理 |
|------|---------|---------|---------|
| 低光室内照（m_immobilier） | sharpness 3-8, colorfulness 8-20 | AMBIGUOUS 或 PROBABLY_REAL | 允许候选但标注需复核 |
| 高质感渲染（digital_domain CGI） | 所有指标优秀但 dataset 已知高风险 | AMBIGUOUS + "CGI risk" 标志 | 必须人工审核 |
| 严重滤镜照片 | colorfulness > 100, edge_ratio < 0.10 | PROBABLY_REAL 但降级 | 降级一级 |
| 轻微模糊照片 | sharpness 8-15, 其他指标正常 | PROBABLY_REAL + FAIR | 进入复核池 |
| 嵌入文字的实景 | 整体 edge_ratio 0.25-0.35, 但内容像实景 | AMBIGUOUS | 进入复核池 |
| 单人近景肖像 | aspect_ratio ~0.75-1.0, 背景模糊 | REAL/EXCELLENT 但肖像 | 排序降权 |

### 6.3 特殊情况说明

**电影剧照：** 电影剧照的视觉特征与实景几乎相同（真实相机拍摄）。无法通过启发式区分。解决方案：检查文件名模式（可选）或人工标注。默认视为正常实景候选。

**低饱和室内图（如白墙教室）：** colorfulness 可低至 5.9，但 edge_ratio 也低（<0.15）。不会被 R5 误杀。属于 REAL 级别的正常候选。

**人像/活动照：** 视觉上属于实景，不淘汰但排序降权。肖像配额 ≤ 10%。

**产品照：** 如果是真实拍摄的产品照片（非 3D 渲染），属于实景。如果是纯白底产品展示图（colorfulness 可能偏低），需检查 edge_ratio。低 edge_ratio + 中低 colorfulness → 真实产品照。

---

## 7. 已知失效模式

### 7.1 本框架无法解决的问题

| 失效模式 | 影响 | 缓解措施 |
|---------|------|---------|
| **高保真 3D 渲染 vs 真实照片** | 启发性信号完全重叠 | 标记高风险数据集（`digital_domain`），人工审核 |
| **AI 生成图像** | 同上，且 AI 质量持续提升 | 指标极端优秀时标记 "potential AI" |
| **照片内嵌截图** | 截图中的照片区域可能通过 heuristic 检查 | 整体 edge_ratio > 0.30 即拒绝整个图像 |
| **极端后期/滤镜** | 色彩异常但其他指标正常 | colorfulness > 100 + edge_ratio < 0.10 降级 |
| **黑白/单色照片** | colorfulness 接近 0 | 需单独规则：edge_ratio < 0.15 且 entropy > 5.0 视为真实 |

### 7.2 跨数据集分布漂移的影响

| 数据集 | Sharpness p5 | 全局固定阈值 (5.0) 的影响 |
|--------|-------------|------------------------|
| m_immobilier | 3.0 | 会误杀约 10% 的实景室内照 |
| maior_capital | 22.9 | 无损 |
| kpmg_forensic | 2.7 | 部分截图被保留（但会被 R1 杀死） |

**结论：** 必须使用数据集自适应阈值。全局固定阈值不可接受。

### 7.3 需要人工复核的高风险数据集

| 数据集 | 风险 | 预计通过率 | 建议人工复核量 |
|--------|------|-----------|-------------|
| digital_domain | CGI 风险最高 | ~30%（含人工） | 全部候选（500-800 张） |
| roland_berger | 大量信息图混合 | ~40% | 复核池中的候选 |
| tuv_rheinland | 图表/文档混合 | ~30% | 复核池中的候选 |
| kpmg_forensic | 几乎全截图 | ~5% | 全量（仅 80 张） |

---

## 8. 性能预算与工程落地建议

### 8.1 单图计算成本

| 步骤 | 操作 | 每图耗时（CPU） | 备注 |
|------|------|----------------|------|
| 1 | 读取 + 缩放到 640px | ~0.01s | Pillow + LANCZOS |
| 2 | Sharpness（Sobel 梯度均值） | ~0.02s | OpenCV，640px 灰度图 |
| 3 | Edge Ratio（Canny） | ~0.01s | OpenCV，低分辨率 |
| 4 | Colorfulness（Hasler-Susstrunk） | ~0.01s | 基于 Lab 色彩空间 |
| 5 | Entropy + Brightness Std | ~0.005s | 灰度直方图 |
| 6 | Aspect Ratio + Resolution | ~0.001s | 直接从图像尺寸 |
| 7 | Composite Score 计算 | ~0.001s | 纯数值运算 |
| 8 | dhash 计算（去重用） | ~0.01s | 灰度缩放到 9×8 |
| 9 | MobileNetV3-Small（可选） | ~0.07s | 仅用于 AMBIGUOUS 池 |

**合计（仅启发式）：~0.06s/图，62,000 图 → ~62 分钟**
**合计（含模型）：最多额外 +0.07s × AMBIGUOUS 池大小**

### 8.2 工程流水线建议结构

```
Stage A: Heuristic Scan → 全量 62K 图，产出 per-image JSON
  ├─ 读取 → 缩放到 640px → Sharpness / Edge Ratio / Colorfulness / Entropy / Brightness / Aspect Ratio / Resolution
  ├─ R1-R6 硬拒绝 → NON_REAL (log reason)
  ├─ 计算复合软分 → REAL / PROBABLY_REAL / AMBIGUOUS / PROBABLY_NON_REAL
  └─ dhash → 计算 hamming 距离
  └─ 输出: {file, sharpness, edge_ratio, colorfulness, entropy, brightness_std, aspect_ratio, resolution, real_level, quality_level, dhash, composite_score, reject_reason}

Stage B: Gallery Readiness → 仅对 Non-rejected 图像
  ├─ 计算 Quality 分
  └─ 判定矩阵 → GALLERY_READY / REVIEW_NEEDED / NOT_SUITABLE

Stage C: Diversity & Selection
  ├─ dhash 去重（每簇保留最高 Quality 的 1 张）
  ├─ 按 scene_type 分组（通过 edge_distribution / brightness_std 推断室内/室外）
  ├─ 肖像识别（aspect_ratio + 中央 sharpness 模式）
  └─ 配额: 每数据集最多 100 张，肖像 ≤ 10 张

Stage D: Output
  ├─ top_100_per_dataset/{dataset}/ (图片副本或符号链接)
  ├─ statistics.json (逐数据集统计)
  ├─ review_pool/{dataset}/ (待人工审核)
  └─ read_errors.log
```

### 8.3 保守偏置策略汇总

| 场景 | 操作 |
|------|------|
| 复合软分在边界附近 (±0.05) | 降级到更保守的级别 |
| 数据集分布不稳定（样本 < 200） | 用全局默认值代替数据集自适应 |
| 模型预测与启发式矛盾 | 采用更保守的结论 |
| 无法读取的文件 | 记录日志，不静默忽略 |
| 数据集候选不足 100 张 | 输出实际数量，不填充低质量替代品 |
| 无法判断实景/非实景 | 进入复核池，不自动淘汰或通过 |

---

> **版本历史**
> - v1.0 (Round 1): 初始框架，基于假设性判据
> - v2.0 (Round 2): 基于实证数据校正 colorfulness 判据，引入 edge_ratio
> - v2.1 (当前): 整合 10 条边界规则，明确硬拒绝/低可信/待复核实例，添加性能预算
