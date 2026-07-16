# Stage 2 规划方案 — 可直接交付 Executor

> 基于 Stage 1 跨 11 数据集、220+ 样本的实证验证。CPU-only、可复现、可复核。

---

## 1. 视觉判断逻辑总纲

详细的视觉定义框架见 `visual_definition_framework.md`。核心总结如下：

### 1.1 实景/非实景判定（3 层级联）

**第 1 层：硬阈值拒绝（~0.05s/图）**
```
R1: edge_ratio > 0.35 AND colorfulness > 70  → NON_REAL（彩色 UI/信息图）
R2: aspect_ratio < 0.25 OR aspect_ratio > 4.0 → NON_REAL（极端横幅）
R3: entropy < 3.5 AND brightness_std < 15     → NON_REAL（纯色占位）
R4: sharpness_640 < 2.0                       → NON_REAL（过糊）
R5: colorfulness < 4 AND edge_ratio > 0.20    → NON_REAL（文档）
R6: min(width, height) < 150                  → NON_REAL（太小）
```

**第 2 层：软分排序（剩余候选）**
```
composite = 0.25×colorfulness_norm + 0.25×entropy_norm 
          + 0.25×brightness_std_norm + 0.25×(1 - min(edge_ratio/0.3, 1.0))
```
- ≥ 0.55 → REAL
- 0.40–0.55 → PROBABLY_REAL
- 0.25–0.40 → AMBIGUOUS
- < 0.25 → PROBABLY_NON_REAL

**第 3 层：可选 MobileNetV3-Small 验证（仅 AMBIGUOUS 池，+~0.07s/图）**
- 1.5M 参数，已验证 CPU 可用
- 仅对 AMBIGUOUS 候选执行，重新评分后决定归属

### 1.2 Gallery 质量判定（4 级，数据集自适应阈值）

| 等级 | Sharpness | Colorfulness | Brightness | 最短边 |
|------|-----------|-------------|-----------|--------|
| EXCELLENT | > 30 | ≥ 20 | 60-220 | ≥ 800px |
| GOOD | 15-30 | 8-20 | 40-60或220-240 | ≥ 500px |
| FAIR | max(2.0, p2)-15 | 3-8 | 30-40或240-250 | ≥ 300px |
| POOR | < max(2.0, p2) | < 3 | < 30或>250 | < 300px |

> **关键**：Sharpness 下限使用 `max(2.0, dataset_p2)`，即数据集的第 2 百分位。这防止真实室内照片（m_immobilier p5=3.0）被全局阈值错误淘汰。

---

## 2. CPU-only 可实现流水线

### 2.1 总体架构：三阶段级联

```
Phase A (启发式预筛) → Phase B (模型验证，可选) → Phase C (多样性与排序)
  全量 62K 图           仅候选 ~15K 图               仅候选 ~15K 图
  ~70 min               ~18 min (MobileNetV3)         ~5 min
```

### 2.2 Phase A：启发式预筛（必选，CPU-only）

**输入**：全量 61,958 张图像
**步骤**：

A1. **文件扫描与验证**
   - 递归遍历 C:\pics\{dataset}/
   - 尝试 Pillow open
   - 支持的格式：.jpg, .jpeg, .png, .webp, .bmp, .gif
   - 失败的记录到 `logs/read_errors.log`（路径、错误类型、格式）
   - **不静默跳过任何文件**

A2. **缩略图计算**
   - 最长边缩放到 640px（保持比例），BILINEAR
   - 全部计算在该缩略图上进行

A3. **信号提取**（每图 ~0.05-0.09s）
   - `sharpness_640` = Sobel 梯度均值
   - `colorfulness` = Hasler-Susstrunk 指标
   - `brightness` = 像素均值（0-255）
   - `brightness_std` = 像素亮度标准差
   - `aspect_ratio` = width / height
   - `entropy` = 灰度直方图信息熵
   - `edge_ratio` = Canny 边缘像素 / 总像素
   - `dhash` = 8×9 差异哈希 → 64bit 整数

A4. **硬阈值拒绝**（规则 R1-R6 如上）
A5. **软分计算**（对通过 R1-R6 的候选）
A6. **数据集统计收集**
   - 统计通过/拒绝数量
   - 计算 p2/p5/p50/p95 分位数（用于 Phase B 的自适应阈值）
   - 保存到 `results/{dataset}/dataset_stats.json`

**输出**：
- `candidates_{dataset}.json` — 通过 A4 的候选列表（含信号值）
- `rejected_{dataset}.json` — 被拒绝列表（含拒绝原因）
- `dataset_stats.json` — 数据集统计

### 2.3 Phase B：模型验证（可选，仅在以下情况执行）

**触发条件**：
- 数据集含有大量 AMBIGUOUS 候选（如 `digital_domain`、`roland_berger`）
- Phase A 后候选数仍大于 3000
- Executor 手动指定 `--use-model`

**模型**：MobileNetV3-Small（timm 实现）
- 参数：1.5M（远 < 100M 限制）
- 速度：已验证 ~0.07s/图 CPU
- 操作：抽取 576-dim 特征向量，通过零样本分类器或简单的线性探针计算 photo_probability

**如果 MobileNetV3-Small 也不够**（已知盲区 CGI/AI）：
- 不额外尝试大模型
- 标记高风险候选 → 人工复核池
- 记录 `"model_check_inconclusive"` 到 CSV

### 2.4 Phase C：多样性与最终排序

**C1. 近似去重**
```
dhash Hamming 距离 ≤ 4 → 同簇
加速：前 8bit 哈希分桶，只桶内比较（避免 O(n²)）
每簇保留复合质量分最高的图像
其余标记为 "duplicate_suppressed"
```

**C2. 场景聚类**
- HSV 3D 直方图（8×4×4 箱 → 128-dim），余弦距离
- DBSCAN 聚类（eps=0.25, min_samples=2）

**C3. 场景配额**
- 单人肖像 ≤ 10 张
- 同一位置/角度 ≤ 3 张
- 每类场景至少选 1 张（如果存在）

**C4. 圆桌选择**
1. 按场景簇排序，每个簇取当前最高分候选
2. 轮转取分，直到 100 张或资源耗尽
3. 如果不足 100 → 输出实际数量，不填充

**排序公式**
```
final_rank = 按以下优先级：
  1. 实景等级（REAL > PROBABLY_REAL）
  2. 质量等级（EXCELLENT > GOOD）
  3. 质量复合分（sharpness_norm × colorfulness_norm）
  4. 场景偏好的加分（非肖像 +0.05）
  5. model_score（如有，仅作为同分决胜）
```

---

## 3. 输出目录与文件结构

```
workspace/
├── run_pipeline.py              # 主入口
├── config.yaml                  # 数据集特定配置
├── phase_a_heuristics.py        # Phase A 实现
├── phase_b_classifier.py        # Phase B 模型验证
├── phase_c_diversity.py         # Phase C 多样性与排序
├── quality_scoring.py           # 质量评分函数
├── diversity_utils.py           # dhash/DBSCAN 工具
├── visual_definition_framework.md  # 本文件
│
├── logs/
│   ├── pipeline_run_{timestamp}.log   # 完整日志
│   └── read_errors.log               # 不可读取文件
│
└── results/
    ├── {dataset}/
    │   ├── top100_gallery.csv         # 最终 gallery 选择
    │   ├── review_pool.csv            # 复核池
    │   ├── rejected_all.csv           # 全部拒绝原因
    │   ├── candidates.json            # Phase A 候选
    │   └── dataset_stats.json         # 统计
    ├── all_datasets_stats.json        # 聚合统计
    └── sampling_report.json           # 抽样复核结果
```

### top100_gallery.csv 字段
```
relative_path, dataset, rank, real_label, quality_label, gallery_label,
composite_score, sharpness, colorfulness, brightness, edge_ratio, entropy,
aspect_ratio, width, height, dhash, scene_cluster_id, rejection_reason,
notes
```

### 拒绝原因词汇表
- `non_real_R1`(edge+color), `non_real_R2`(aspect), `non_real_R3`(low-entropy)
- `non_real_R4`(blurry), `non_real_R5`(document), `non_real_R6`(tiny)
- `low_quality_sharpness`, `low_quality_exposure`, `low_quality_size`
- `duplicate_suppressed`, `diversity_limited`, `portrait_cap`
- `read_error`, `unsupported_format`

---

## 4. 日志与抽样复核

### 4.1 日志要求
- 每 1000 张图像输出进度（dataset, processed/total, count_R1..R6）
- 所有拒绝记录到 `rejected_all.csv`（路径 + 拒绝原因 + 信号值）
- 所有不可读文件记录到 `read_errors.log`（路径 + 错误）
- 运行结束时输出摘要：各数据集通过率/拒绝率/入选率

### 4.2 抽样复核机制
1. 从每个数据集的 `top100_gallery.csv` 中随机抽 10%
2. 从 `review_pool.csv` 中随机抽 20%
3. 从 `rejected_all.csv` 中抽 5%（确保没有错误拒绝）
4. 输出到 `sampling_report.json`：文件路径 + 原分类 + 建议人工复核标志

---

## 5. 风险、阈值校准与跨数据集策略

### 5.1 已知风险矩阵

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CGI 误判为实景 | digital_domain 混入 gallery | 该数据集全量进入审核池 |
| 低光室内照被误拒绝 | m_immobilier 低 sharpness | 自适应 p2 阈值 |
| 高饱和度截图误判为实景 | kpmg 彩色截图 | R1 规则（edge_ratio+colorfulness） |
| 信息图通过全部检查 | roland_berger | Phase B model 验证 + review_pool |
| AI 图像无法检测 | 所有数据集 | 标记 >90th 百分位图像；已知盲区 |
| 损坏文件静默跳过 | 数据不全 | 明确日志记录 |

### 5.2 阈值校准方式

**初始化校准**（首次运行）：
- 使用 config.yaml 中的默认阈值
- Phase A 后计算每个数据集的 p2/p5/p50/p95 分位数
- 输出校准报告：`results/{dataset}/threshold_calibration.json`

**人工校准**（复核后）：
- 从 review_pool 和 sampling_report 中读取人工反馈
- 调整 config.yaml 中的数据集特定覆盖：
  ```yaml
  digital_domain:
    sharpness_floor: 5.0  # 替代默认 max(2.0, p2)
    colorfulness_min: 10  # 替代默认 8
    mobileclip_threshold: 0.6  # 更严格要求
  ```

**关键原则**：
- 阈值调节基于**视觉证据分析**，不基于"感觉"
- 每次调节记录到 config.yaml 的注释中（为什么改）
- 优先使用数据集自适应机制，减少人工调参

### 5.3 跨数据集内容差异处理策略

| 场景类型 | 特征 | 处理策略 |
|---------|------|---------|
| 纯实景（truro_school, m_immobilier） | >85% 通过 R1-R6 | Phase A 宽松下限，主要关注质量排序 |
| 混合实景+UI（boston_university） | 30-50% 通过 | R1 是关键拒绝规则 |
| 混合实景+信息图（roland_berger） | <40% 通过 | R5 + Phase B 联合判断 |
| CGI+实景混合（digital_domain） | 无法启发式分离 | 全量人工复核标志 + review_pool |
| 截图为主（kpmg_forensic） | <10% 通过 | 预期 0-5 张产出，不填充 |
| 演示幻灯+照片（thema-med） | 中等通过率 | R5 拒绝大部分，剩余 review |

---

## 6. Executor 执行检查清单

### 运行时
```
# 全量运行（推荐，~80 min + 可选 +18 min）
python run_pipeline.py --phase all --parallel 4

# 仅 Phase A（快速测试）
python run_pipeline.py --phase A

# 单数据集调试
python run_pipeline.py --datasets boston_university --phase all

# 启用模型验证
python run_pipeline.py --phase all --use-model --model mobilevitv2_050
```

### 输出检查
1. 检查 `read_errors.log` 确保无大量意外失败
2. 检查 `top100_gallery.csv` 的空值/异常值
3. 检查 `rejected_all.csv` 的拒绝分布是否合理
4. 检查 `sampling_report.json` 的复核建议
5. 对 `digital_domain` 的入选图像全量人工复核

### 常见故障排除
- **Pillow 无法打开某些文件**：检查格式是否在支持列表；记录到日志
- **Sharpness 整体偏低**：检查是否缩略图缩放参数错误（应最长边 640px）
- **Colorfulness 异常**：检查 RGB 转换是否正确（RGBA/P 模式需先转换）
- **OOM/内存不足**：Phase A 一次只加载一张缩略图；Phase B 批处理大小 32
- **并行加速**：CPU 6 核建议 --parallel 4；硬盘慢则 --parallel 2

---

## 7. 已知不可自动解决的失败模式

| 失败模式 | 原因 | 当前缓解 |
|---------|------|---------|
| 照片级 CGI 无法区分 | 启发式看不到"不自然" | digital_domain 人工复核 |
| AI 生成图像无法检测 | 无可靠 <100M 参数检测器 | 标注 >90th 百分位图像 |
| 真实照片中的版权/品牌 | 非视觉信息，无法判断 | 明确声明：输出是工程预过滤 |
| 文化/上下文敏感性 | 技术上合格但内容不适 | 人工复核是必要环节 |
| 同一数据集仅一种场景 | 多样性限制 < 100 张 | 输出实际数量，不填充 |
| 旋转/裁剪的近似重复 | dhash 对旋转敏感 | 用 HSV 直方图余弦距离补充 |

---

## 8. 结论

本方案直接基于 Stage 1 的实证数据（11 数据集、220+ 样本的 sharpness/colorfulness/edge_ratio/entropy 分布）设计。核心设计决策均有数据支撑：

1. **Edge_ratio 取代 colorfulness 作为主分离信号** — 数据证明 colorfulness 单独使用会反向误导
2. **数据集自适应百分位阈值** — 跨数据集 sharpness 分布差异可达 7×，全局阈值必然失败
3. **三阶段级联架构** — 启发式预筛（~70min）→ 可选模型验证（+18min）→ 多样排序（~5min），避免对全量 62K 图做重推理
4. **保守偏置** — 所有 AMBIGUOUS/近边界样本进入复核池，预期产出从 0（kpmg）到 100（truro_school）诚实报告
5. **完全可审计** — 每个拒绝有明确原因代码，每张入选图有完整信号值轨迹，抽样复核机制内置
