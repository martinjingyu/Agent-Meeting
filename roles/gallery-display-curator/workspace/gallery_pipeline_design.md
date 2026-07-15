# Gallery Display Curator — 官网 Gallery 图片挑选流水线设计方案

## 1. 设计原则与判断逻辑论证

### 1.1 "实景图 vs 非实景图"的判断逻辑

**核心判断依据**：基于图片的视觉内容特征，判断其是否呈现"可由真实相机拍摄得到的现实世界场景"。

| 特征维度 | 实景图倾向 | 非实景图倾向 |
|----------|-----------|-------------|
| 色彩分布 | 自然连续色调，Hasler-Susstrunk 彩度通常在 8-60 之间 | 极高彩度(>100)或极低彩度(<5)；大块纯色区域 |
| 边缘结构 | Sobel 梯度分布自然，有景深导致的焦点模糊 | 边缘过于锐利(文字/线条)或过于平滑(3D渲染/插画) |
| 纹理复杂度 | 自然纹理（树叶、布料、墙面等），熵值分布广 | 大面积平滑渐变(渲染)或重复图案(UI/图表) |
| 宽高比 | 通常在 0.5-2.5 之间（标准相机传感器比例） | 极端宽高比(<0.3 或 >3.5)多为横幅/长图 |
| 噪声特征 | 传感器噪声自然分布 | 过度降噪(平滑)或零噪声(CGI) |

**边界情况处理**：
- **高质量3D渲染/建筑效果图**：无论是启发式方法还是<100M参数的小模型，当前都无法可靠区分。digital_domain 数据集的全部候选需标记人工审核。
- **UI截图**：通过组合规则检测（边缘密度 > 0.35 且 彩度 > 70 → R_NON_REAL）。经验验证显示：截图（kpmg_forensic）中位彩度 = 60.0，真实照片（m_immobilier）中位彩度 = 29.2——截图比真实照片更鲜艳。
- **AI生成图**：特征介于实景和渲染之间，保守处理——若 P(real) < 0.5，归入待复核池。
- **重度修图/滤镜**：保留主体场景信息则仍算实景，但需额外质量评分。

### 1.2 关键实证发现（支撑阈值选择）

以下发现是基于当前机器的实际验证，用于指导阈值设计：

| 实证发现 | 数值 | 来源 |
|---------|------|------|
| MobileCLIP2-S0 参数量 | 74.8M (✓ <100M) | check_models.py 验证 |
| MobileCLIP2-S0 CPU 前向时间 | 1.22s/张（实测平均） | 200张推理基准测试 |
| MobileNetV3-Small CPU 前向时间 | 0.07s/张 | timm 基准测试 |
| 真实照片锐度 @640px 范围 | 1.7-35.1 | 20样本×11数据集 |
| 数据集 p5 锐度 @640px | 2.3-8.7（跨数据集） | calibrate_sharpness.py |
| 阈值=2.0 假拒绝率 | ~0.5%（仅1/20在2个数据集上） | 验证——安全的保守值 |
| 阈值=5.0 假拒绝率 | 10-50%（真实照片被误拒绝） | 验证——过于激进 |
| 截图 vs 真实照片中位彩度 | 60.0 vs 29.2 | kpmg_forensic vs m_immobilier |

### 1.3 "适合官网 Gallery 展示"的判断标准

需同时满足四个维度：

1. **真实实景可信度**：模型 P(real) ≥ 0.5，优选 ≥ 0.7。纯启发式模式下用组合规则替代。
2. **画面质量**：
   - 锐度（归一化 Sobel）≥ max(2.0, dataset_p2)（640px尺度）
   - 无严重过曝/欠曝（平均亮度在 20-245 之间，8-bit）
   - 无极端宽高比（0.3-3.5 范围内）
3. **展示适配性**：
   - 主体/场景可辨识
   - 画面清晰、曝光自然、构图稳定、视觉舒适
4. **官网场景偏好**（按优先级从高到低）：
   - 风景/自然 → 建筑/城市 → 室内空间 → 多人活动/群像 → 单人照/特写
   - 同等条件下，优先选择场景类型更丰富、叙事性更强的图片

### 1.4 "不适合官网展示"的拒绝原因分类

| 拒绝码 | 含义 | 触发条件 |
|--------|------|---------|
| R_BLURRY | 模糊 | Sobel 锐度 < max(2.0, dataset_p2)（640px尺度） |
| R_ASPECT_RATIO | 极端宽高比 | 宽高比 < 0.25 或 > 4.0 |
| R_EXPOSURE | 曝光异常 | 平均亮度 < 10 或 > 248 |
| R_NON_REAL | 非实景（UI截图检测） | 边缘密度 > 0.35 **且** 彩度 > 70 |
| R_INFOGRAPHIC | 疑似信息图 | 彩度 < 4 **且** 熵 < 4.0 |
| R_TEXT_HEAVY | 文字密集 | Sobel 边缘像素 > 50% **且** 彩度 < 10 |
| R_DUPLICATE | 近似重复 | dhash 汉明距离 ≤ 4（并查集全局聚类） |
| R_LOW_QUALITY | 综合低质 | 综合质量分 Q 低于数据集阈值 |
| R_OVER_SELECTED | 相似场景过多 | 同一场景类型配额已满（多样性轮选） |
| R_REVIEW_POOL | 边界待复核 | P(real) 0.3-0.5 或 Q 在截止线 ±10% |
| R_CORRUPT | 文件损坏 | 无法读取/解码 |

## 2. 流水线架构：三阶段级联

```
Phase A: 启发式预筛选（全量 61,958 张，~70 分钟）
  → 快速剔除明显不合格图片
  → 输出候选池（约 12,000-18,000 张）

Phase B: 神经分类器（仅候选池，速度依赖于基准测试）
  → 前置200张速度基准测试决定模式
  → 模型零样本实景分类 + 质量评分

Phase C: 多样性排序与选择（后处理）
  → DBSCAN 聚类 + 圆形轮选
  → 每数据集 Top 100（不足则如实输出）
  → 统计报告 + 待复核池
```

### Phase A: 启发式预筛选（全量图片）

**输入**：所有图片文件
**处理步骤**：

```
A1. 读取与解码
    - PIL.Image.open() → RGB array (max 640px longest side)
    - 记录无法读取/损坏文件 (R_CORRUPT)
    - 跳过缩略图（最短边 < 300px → R_LOW_QUALITY）

A2. 极低质量剔除（快速拒绝，硬阈值）
    - Sobel 锐度 < 2.0  → R_BLURRY（640px尺度下的安全阈值）
    - 宽高比 < 0.25 或 > 4.0 → R_ASPECT_RATIO
    - 平均亮度 < 10 或 > 248 → R_EXPOSURE

A3. 非实景启发式剔除（组合规则）
    - R4: 边缘密度 > 0.35 且 彩度 > 70 → R_NON_REAL（UI截图检测）
    - R5: 彩度 < 4 且 熵 < 4.0 → R_INFOGRAPHIC（信息图/文档）
    - 边缘密度 > 50% 且 彩度 < 10 → R_TEXT_HEAVY（文字密集）

A4. 数据集自适应阈值
    - 每个数据集独立计算百分位数：p2、p5、p10、p25、p50
    - 锐度下限：max(2.0, dataset_p2)
    - 彩度下限（仅用于R4组合规则）：max(4, dataset_p5)
    - 小数据集（<300张）：使用固定全局阈值（不计算百分位）

A5. 近似重复检测
    - 计算 dhash (8×9 → 64-bit)
    - 并查集（Union-Find）全局聚类，汉明距离 ≤ 4
    - 每个聚类仅保留综合评分最高的1张进入候选项
    - 被拒绝图片记录：rejection_reason = "near_duplicate_of: {保留图片路径}"

A6. 候选池输出
    - 通过 A2-A4 所有过滤（且非 A5 重复）的图片进入 Phase B 候选池
    - 每个聚类仅保留最佳1张
    - 预期候选池大小：约 12,000-18,000 张（数据集差异大）
```

### Phase A 和 Phase B 之间的速度基准测试

在候选池上运行200张进行速度基准测试，用于运行时决定 Stage B 的模式：

```python
# 伪码
benchmark_mean = benchmark(model, 200_random_candidates)
if benchmark_mean < 0.8s:
    mode = "FULL"          # 对所有候选进行模型推理
elif 0.8s <= benchmark_mean <= 2.0s:
    mode = "REDUCED"       # 对 Q_heuristic 前50%的候选推理，其余用启发式分数传播
else:
    mode = "HEURISTICS_ONLY"  # 跳过模型，使用纯启发式质量分数
```

### Phase B: 神经分类器（仅候选池）

**模型选择**：按速度基准测试结果决定

| 模型 | 参数量 | 实测速度 | 用途 |
|------|-------|---------|------|
| MobileNetV3-Small (timm) | 1.5M | 0.07s/张 | 实景零样本分类 + 多样性特征提取 |
| MobileCLIP2-S0 (dfndr2b) | 74.8M | 1.22s/张 | 零样本实景分类（FULL模式时使用） |

**零样本实景分类流程**（MobileNetV3-Small 或 MobileCLIP2-S0）：

```
实景提示词（5条）：
- "a real photograph of an outdoor landscape or nature scene"
- "a real photograph of a building, cityscape or architecture"
- "a real photograph of an indoor room or interior space"
- "a real photograph of people in a group activity or event"
- "a real photograph of a natural scene with real world objects"

非实景提示词（5条）：
- "a computer generated 3D render, CGI or digital rendering"
- "a screenshot of a website, document or presentation"
- "a graphic design, illustration, icon or infographic"
- "a chart, diagram or data visualization"
- "a painting, drawing, digital art or cartoon"

计算方式：
- 分别对 5 条实景和 5 条非实景提示词计算文本特征
- 两组各自取平均并归一化
- 计算图片特征与两组平均特征的余弦相似度
- P(real) = softmax([real_sim, non_real_sim])
```

**综合质量评分**：

有模型（FULL/REDUCED 模式）：
```
Q = 0.40 × P(real)
  + 0.15 × sharpness_norm
  + 0.10 × colorfulness_norm
  + 0.10 × entropy_norm
  + 0.15 × brightness_norm
  + 0.10 × (1 - edge_ratio_norm)
```

无模型（HEURISTICS_ONLY 模式）：
```
Q_heuristic = 0.25 × sharpness_norm
            + 0.25 × colorfulness_norm
            + 0.15 × entropy_norm
            + 0.20 × (1 - |brightness_norm - 0.5| × 2)
            + 0.15 × (1 - edge_ratio_norm)
```

权重理由：P(real) 获得最高权重（0.40），因为神经模型能捕捉启发式无法区分的边界情况（CGI vs 真实、高质量渲染 vs 照片）。其余项作为正则化器——防止模型仅因 P(real) 高而选择模糊/过暗的图像。

**REDUCED 模式的特征传播**：
- 前50%候选用模型推理获得 P(real)
- 后50%候选：用前50%候选的特征做 k-NN（k=3，余弦距离），加权平均传播 P(real)
- 记录在 CSV 中：`p_real_source = "model" | "knn_propagated"`

### Phase C: 多样性排序与最终选择

**步骤**：

```
C1. 特征提取
    - 模型：MobileNetV3-Small（timm，1.5M参数，0.07s/张）
    - 特征维度：1024 → PCA降至 32 维（用于聚类速度）
    - 仅对 Phase B 候选运行（约5,000-18,000张）

C2. DBSCAN 聚类
    - eps=0.5, min_samples=2
    - 无固定K值——自动适应每个数据集的内容分布
    - 单元素聚类 = 视觉独特的异常值（保留）

C3. 场景层级分配（基于聚类中位像素统计，无需额外模型）
    | 层级 | 判断规则 | 配额（100张中） |
    |------|---------|----------------|
    | 景观/城市/建筑 | 中位宽高比 ≥ 1.6 且 中位亮度 > 120 | 0-35 |
    | 群体活动/事件 | 中位彩度 > 25 且 中位宽高比 1.2-1.8 | 0-25 |
    | 室内空间 | 中位亮度 ≤ 110 且 中位宽高比 < 1.6 | 0-30 |
    | 单人/肖像 | 中位宽高比 1.1-1.5 且 中位彩度 > 35 | ≤10（软性） |
    | 物体/产品特写 | 中位彩度 < 18 或 中位宽高比接近1.0 | 0-10 |
    | 文字密集型/标识 | 不符合以上任何条件 | 0-5 |

C4. 圆形轮选算法
    1. 按 composite_score 在每个 DBSCAN 聚类内降序排序
    2. 初始化 selected = [], used_clusters = set()
    3. 对 轮次 = 1 到 10（或直到 selected 达到100个项目）：
       a. 对每个 DBSCAN 聚类（每轮随机顺序）：
          - 如果该聚类本轮已贡献K个项目：跳过
          - 从该聚类选取最高分的未选择候选
          - 如果候选的场景层级是"肖像"且已选肖像数 ≥ 10：应用 -0.10 diversity_bonus
          - 将候选加入 selected
       b. 如果没有聚类能贡献项目：终止
    4. 如果 selected < 100：从剩余项目中按 composite_score 降序补充
    5. 最终检查：如果有场景层级占比超过35%：从过表示层级降级，从欠表示层级升级

C5. 肖像限制（软性约束，非硬性上限）
    - Haar 级联面部检测（不可靠，仅作参考信号）
    - 对标记为肖像的图像应用 diversity_bonus = -0.05 至 -0.10
    - 绝不因面部检测而硬性拒绝图片
    - 所有面部检测结果记录在 CSV 中供人工审核
    - 最终100张中肖像占比建议 ≤ 20%（人工可覆盖）

C6. 永不填充政策
    - 如果所有筛选后合格图片 < 100 张，如实输出实际数量
    - 绝不为了凑满 100 张而选入边界样本
    - 报告原因："去重+真实照片分类+质量筛选后合格图像不足"
```

## 3. 数据集专属策略

| 数据集 | 图片数 | Stage B策略 | 特殊处理 |
|--------|-------|------------|---------|
| truro_school | 36,266 | 分层（候选池约8k-10k，全部处理） | 最大数据集；全部候选跑模型（非分层抽样） |
| m_immobilier | 5,000 | **跳过 Stage B** | 已知全为真实房产照片；重点在去重和多样性；激进 dhash (Hamming ≤ 6 用于同房源聚类) |
| maior_capital | 5,000 | 标准 | 文件名含 "pianta"/"planimetria" 仍需视觉判断 |
| tara_guerard | 4,971 | 标准 | 博客图形混杂；严格质量过滤 |
| roland_berger | 2,997 | 强制运行 | 高信息图比例（~50%）；彩度 p15 激进过滤 |
| boston_university | 2,722 | 标准 | UI截图由 R4 (边缘密度+彩度) 捕获 |
| digital_domain | 1,669 | **强制运行** + 标记人工审核 | VFX/CGI与真实照片边界模糊——全部候选需人工审核 |
| tuv_rheinland | 1,465 | 强制运行 | 56% PNG 信息图；彩度 p20 激进过滤 |
| ul_solutions | 1,515 | 标准 | 产品照 vs 环境照；极端宽高比过滤横幅 |
| thema-med | 273 | 标准（全局阈值） | 小数据集（<300张）；去重优先（重复率10.5%） |
| kpmg_forensic | 80 | 标准（全局阈值） | 小数据集；全量标记→人工审核，预期入选0-3张 |

## 4. 输出格式设计

### 输出目录结构
```
workspace/output/
├── logs/
│   ├── pipeline_run.log          # 运行日志（含每步耗时）
│   └── errors.log                # 错误/异常/损坏文件记录
├── per_dataset/{dataset}/
│   ├── top100.csv                # 最终入选 Top 100（含排序、评分、原因）
│   ├── rejected.csv              # 被拒绝图片清单及原因
│   ├── review_pool.csv           # 待复核池（边界情况）
│   └── stats.json                # 数据集统计
├── aggregate/
│   ├── all_top100.csv            # 所有数据集的 Top 100 汇总
│   ├── all_rejected.csv          # 所有被拒绝图片汇总
│   ├── all_review_pool.csv       # 所有待复核图片汇总
│   └── summary_report.json       # 全局汇总统计
└── sampling_check/
    └── validation_report_10pct.json  # 10%抽样检查报告
```

### CSV 字段设计
```
relative_path             # 图片相对路径（如 "m_immobilier/12345_1.jpg"）
dataset                   # 数据集名称
image_format              # 文件格式
width, height             # 原始分辨率
result                    # final_select | rejected | review_pool
rejection_reason          # 拒绝码（R_BLURRY / R_NON_REAL / ...）
p_real                    # 实景概率（或 None 如果纯启发式模式）
p_real_source             # model | knn_propagated | heuristic_only
quality_score             # 综合质量分 Q 或 Q_heuristic
sharpness                 # Sobel锐度值（640px尺度）
colorfulness              # Hasler-Susstrunk 彩度值
entropy                   # 图像熵
brightness                # 平均亮度（8-bit）
aspect_ratio              # 宽高比
edge_ratio                # Sobel边缘像素比例
dhash                     # 64位差值哈希（十六进制）
dhash_cluster_id          # dhash 近似重复聚类ID
dbscan_cluster_id         # DBSCAN 语义聚类ID
scene_hierarchy           # Landscape/Urban/Indoor/Group/Portrait/Object/Text
face_count                # Haar级联检测到的人脸数量
face_largest_area         # 最大人脸占图片面积比
diversity_bonus           # 多样性调整（正数=加分，负数=降权）
final_rank                # 最终排序名次（仅入选图片）
```

## 5. 预计运行时间

| 阶段 | 最佳情况 | 预期情况 | 最差情况 |
|------|---------|----------|----------|
| 前置基准测试（200张） | 2分钟 | 2分钟 | 2分钟 |
| Phase A（全量62K张） | 60分钟 | 70分钟 | 90分钟 |
| Phase B 快速（FULL模式） | 2.5小时 | 3.5小时 | 5小时 |
| Phase B 中等（REDUCED模式） | 2小时 | 2.5小时 | 4小时 |
| Phase B 慢速（HEURISTICS_ONLY） | 0 | 0 | 0 |
| Phase C（所有数据集） | 8分钟 | 10分钟 | 15分钟 |
| **总计（Stage B FULL）** | **~3.7小时** | **~4.8小时** | **~6.5小时** |
| **总计（Stage B REDUCED）** | **~3.2小时** | **~3.8小时** | **~5.5小时** |
| **总计（HEURISTICS_ONLY）** | **~1.2小时** | **~1.4小时** | **~1.8小时** |

## 6. 运行说明

### 环境准备
```powershell
cd C:\Users\LX034\Code\Agent-Meeting\roles\gallery-display-curator\workspace

# 依赖检查
python -c "import PIL, numpy, torch, open_clip, timm, cv2; print('All dependencies available')"

# MobileCLIP2-S0 权重首次下载（约 50MB）
python -c "import open_clip; open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')"
```

### 执行流水线
```powershell
# 分步执行（推荐，可检查中间结果）
python phase_a_heuristic_filter.py    # 阶段 A：启发式预筛选
python phase_b_model_scoring.py       # 阶段 B：模型评分（含速度基准测试）
python phase_c_final_selection.py     # 阶段 C：多样性排序与选择

# 或一步执行
python run_pipeline.py
```

### 在新数据集上复用
1. 将新图片放在 `C:\pics\{new_dataset_name}\` 下
2. 修改 `run_pipeline.py` 中的数据集列表，添加 `new_dataset_name`
3. 运行流水线——自适应阈值和速度基准测试会自动适配新数据
4. 检查 `output/per_dataset/{new_dataset_name}/review_pool.csv` 中的边界情况
5. 如有需要，在 `config.json` 中为特定数据集覆盖阈值参数

## 7. 抽样检查与可复核性

### 抽样方案
每个数据集按结果分层抽样：
- 入选 Top 10：抽 5 张
- 入选 Top 11-100：抽 10 张
- 待复核池：抽 10 张
- 被拒绝池（按拒绝原因分层）：抽 15 张
- 总计每数据集 40 张，全量 440 张

### 检查内容
- 分类是否正确（真实照片 vs 非真实）
- 质量评分是否合理
- 多样性选择是否覆盖不同场景
- 记录错误类型：误判、漏判、阈值不合适等

### 可追溯性
每张图片的处理结果可追溯到：
- 原始文件路径 → 人工定位查看
- 处理阶段 → 启发式拒绝 / 模型拒绝 / 多样性拒绝
- 具体拒绝原因 → 机器可读码 + 人工可读描述
- 评分详情 → 6个维度分数可查
- 同聚类图片 → dhash 和 DBSCAN 聚类ID 可查

## 8. 已知限制（诚实评估）

1. **照片级CGI/AI图像**：无论是启发式方法还是<100M模型都无法与真实照片区分。digital_domain（VFX工作室）需要强制性人工审核。

2. **dhash对旋转敏感**：旋转/裁剪后的近似重复可能被遗漏。DBSCAN语义聚类可以部分补偿。

3. **Haar级联面部检测不可靠**：深色皮肤、侧面脸部、遮挡脸部、艺术光照均会导致漏检。仅用作软性多样性信号，不用于硬性拒绝。

4. **边缘比率的局限性**：精细纹理（树叶、布料、碎石）也会产生高边缘比率。`edge_ratio + colorfulness`组合减轻了假阳性但未消除。

5. **数据集分布变化**：m_immobilier（全真实照片）与 roland_berger（~50%信息图）需要不同的处理策略。自适应阈值部分解决此问题。

6. **无法律合规性**：此流水线不检查版权、品牌使用、肖像权或任何法律合规性。输出仅为工程预筛选，需人工策展审核。

## 9. 结论段落（可直接写入最终方案）

本方案设计了一个**三阶段级联流水线**，用于从 11 个异构数据集（共 61,958 张图片）中筛选适合官网 Gallery 展示的真实场景图片，全部在 Intel UHD 770 CPU 上运行（无 CUDA）。核心设计决策为：(1) **Phase A** 使用数据集自适应百分位数阈值（锐度下限 = max(2.0, dataset_p2)，640px 尺度），结合边缘密度+彩度组合规则（R4）检测 UI 截图和信息图，通过并查集 dhash 全局去重（汉明距离 ≤ 4）；(2) **Phase B** 采用 MobileNetV3-Small（1.5M 参数，实测 0.07s/张）或 MobileCLIP2-S0（74.8M 参数，实测 1.22s/张），并以前置 200 张速度基准测试决定 FULL/REDUCED/HEURISTICS_ONLY 三种模式；模型通过 5 条实景 + 5 条非实景零样本提示词计算 P(real)；(3) **Phase C** 使用 DBSCAN（eps=0.5, min_samples=2）在 MobileNetV3 特征（PCA 降至 32 维）上进行语义聚类，通过圆形轮选算法实现多样性选择，肖像约束为软性（-0.05 至 -0.10 diversity_bonus，非硬性上限）。关键实证发现包括：锐度阈值 = 2.0 的假拒绝率约 0.5%（安全）；UI 截图比真实照片更鲜艳（中位彩度 60.0 vs 29.2）；MobileCLIP2-S0 实测 1.22s/张（非宣称的 0.51s）。预计运行时间：含模型推理约 3.7-6.5 小时，纯启发式约 1.2-1.8 小时。数据集专属策略包括：m_immobilier 跳过 Stage B（已知全为真实照片）；digital_domain 强制 Stage B 且全部候选标记人工审核（VFX/CGI 边界模糊）；信息图密集型数据集（roland_berger、tuv_rheinland）使用激进彩度过滤。所有输出为可追溯的 CSV/JSON 格式，便于人工审核，并采用保守偏见（绝不凑数至 100 张）。已知限制：CGI/AI 照片级图像无法检测、dhash 对旋转敏感、Haar 级联面部检测不可靠、无法律合规性。输出应仅作为工程化预筛选和人工复核的输入依据。
