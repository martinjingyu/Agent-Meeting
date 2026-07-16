# Top-100 去重、多样性与排序设计

## 1. 设计原则

### 1.1 核心约束
- **禁止纯分数排序**：最终 top-100 不是按单一实景置信度降序取前 100
- **多样性优先于绝对分数**：同一场景/主体/视角的相似样本受配额限制，即使分数很高
- **视觉内容唯一依据**：不依赖文件名、路径、EXIF 做分类依据
- **CPU-only + 可复现**：全部步骤在 Intel UHD 770 上数十秒内完成，用固定 seed 保证可复现

### 1.2 整体流程（三级级联的 Stage C）

```
Stage B 输出（候选池，含视觉特征向量 + 实景分 + 质量分 + 显示适配分）
  │
  ▼
Step C1: 近似去重（特征空间密度过滤）
  │
  ▼
Step C2: 场景/主体聚类（层次化分组）
  │
  ▼
Step C3: 多样性配额分配（按场景层+质量层）
  │
  ▼
Step C4: 圆桌排序（round-robin selection）
  │
  ▼
Step C5: truro_school/房产等高相似集合的特殊处理
  │
  ▼
Step C6: 边界样本复核池 + 日志输出
```

---

## 2. Step C1: 近似去重（Near-Duplicate Dedup）

### 2.1 目标
从候选池中识别并移除**视觉近似的重复图像**，确保没有两张图像共享几乎相同的构图、主体、背景。

### 2.2 方法：特征向量余弦距离 + 密度球过滤

**前提**：Stage B 已为每张图像输出一个 **128~512 维视觉特征向量**（来自 MobileNet-V3 或 EfficientNet-Lite 的 penultimate layer 的 Global Average Pooling 输出）。

#### 2.2.1 距离度量
- 使用 **cosine distance** = 1 − cosine_similarity
- 范围 [0, 2]，0 表示完全相同方向，1 表示正交，2 表示完全相反
- 阈值默认设为 **0.15**（≈ cosine similarity 0.85）

#### 2.2.2 阈值校准原理
| 阈值 | 行为 | 适用场景 |
|------|------|----------|
| 0.10 | 极严格，几乎只去完全相同的复制 | 通用默认 |
| 0.15 | 宽松去重，捕获裁剪/滤镜/小曝光差异 | 房产/学校等高相似集 |
| 0.20 | 更宽松，可捕获同一场景不同角度但构图相似的 | 仅当 recall 优先时 |

**校准策略**：
- 对每个数据集，统计候选池内 pairwise distance 分布
- 取 **第 5 百分位距离 × 1.5** 作为自适应阈值
- 最低阈值 0.08，最高 0.25
- 若数据集内所有 pairwise distance > 0.25，视为无近似重复，跳过去重

#### 2.2.3 去重算法：贪心密度球（Greedy Density Sphere）

```
输入：N 张候选图像，每张带特征向量 v_i 和综合分 s_i
输出：去重后的候选池

1. 按 s_i 降序排序候选列表
2. 初始化保留集 accepted = []，移除集 removed = []
3. 遍历排序后的列表：
   a. 若当前图像已被标记为 removed，跳过
   b. 对当前图像 i：
      - 计算 i 与 accepted 中所有图像的最小 cosine distance d_min
      - 若 d_min < threshold，将 i 加入 removed
      - 否则，将 i 加入 accepted
4. 返回 accepted
```

**为何选择贪心而非全局最优**：
- O(N²) 复杂度 OK（N 通常 500~2000，CPU 可秒级完成）
- 贪心保高分：让高综合分的图像优先保留，低分近似品被移除
- 确定性：固定 seed + 固定排序 → 每次结果一致

#### 2.2.4 去重日志
每移除一张图像记录：
```json
{
  "removed": "img_0421.jpg",
  "kept": "img_0387.jpg",
  "cosine_distance": 0.12,
  "reason": "Near-duplicate (feature distance 0.12 < threshold 0.15)"
}
```

---

## 3. Step C2: 场景/主体聚类

### 3.1 目标
对去重后的候选池进行**无监督场景分组**，得到若干语义上相似的图像簇，用于后续配额分配。

### 3.2 方法：两阶段层次聚类

#### 3.2.1 阶段 A：粗粒度场景聚类（HDBSCAN 或 DBSCAN）
- 输入：视觉特征向量
- 距离：cosine distance
- DBSCAN 参数：
  - `eps`: 自动估计（取第 10 百分位 pairwise distance × 2）
  - `min_samples`: 2（允许单样本簇）
- HDBSCAN（若有库支持）更优，因为它自动选择 eps

#### 3.2.2 阶段 B：细粒度主体/视角分组（在粗场景簇内）
- 每个粗簇内，使用 **Agglomerative Clustering** 进一步细分
- 链接方式：ward（最小化方差）
- 簇数：由 silhouette score 在 [1, min(10, n_samples)] 上自动选择
- 目的：区分同一场景内的不同拍摄角度（正面/侧面/远景/特写）

#### 3.2.3 层次化场景标签

每张图像最终获得两级标签：
```
scene_cluster: 3          # 粗场景簇编号（全局）
viewpoint_cluster: 1      # 细视点簇编号（场景簇内）
scene_size: 45            # 粗场景簇内的样本数
viewpoint_size: 12        # 细视点簇内的样本数
```

#### 3.2.4 降级路径
若 DBSCAN/HDBSCAN 不可用（纯 CPU numpy-only 环境）：
- 使用 **K-means**（k = max(5, N/50)）作为粗聚类
- 然后对每个 K-means 簇做 **层次聚类（scipy.cluster.hierarchy）** 作为细聚类
- 或使用 **基于阈值的贪婪聚类**（见附录 A）

---

## 4. Step C3: 多样性配额分配

### 4.1 目标
确保最终 top-100 中：
- 没有单一场景或主体类型占比超过 **30%**（最多 30 张）
- 没有单一细视点簇占比超过 **15%**（最多 15 张）
- 单人近景（portrait-like）照片占比不超过 **25%**（最多 25 张）
- 至少有 **3 个不同的粗场景簇** 被代表（若候选池允许）

### 4.2 场景层级配额

| 层级 | 配额上限 | 说明 |
|------|----------|------|
| 粗场景簇 | ≤ 30 张 (30%) | 防止单一场景主导 |
| 细视点簇 | ≤ 15 张 (15%) | 防止相同视角重复 |
| 单人近景 | ≤ 25 张 (25%) | 检测方法见 4.3 |
| 最低场景数 | ≥ 3 个簇 | 若候选池有 ≥3 个簇 |

### 4.3 单人近景检测（Portrait Guard）

不依赖文件名，纯视觉判断：

| 信号 | 判定规则 |
|------|----------|
| 宽高比 (aspect ratio) | 0.7 < h/w < 1.5 且最短边 ≥ 300px |
| 主体占比 | face_region_area / total_area > 0.20（若 Stage B 跑了人脸检测） |
| 边缘密度 (edge_ratio) | < 0.05（背景模糊/虚化严重） |
| 颜色数 (colorfulness) | < 20（背景单调） |

**判断逻辑**：
```
is_portrait_like = (
    (0.7 < aspect_ratio < 1.5 and min_side >= 300) AND
    (
        (edge_ratio < 0.05 AND colorfulness < 20)  # 虚化背景单人照
        OR
        (face_ratio > 0.20)  # 人脸占比大
    )
)
```

**注意**：人脸检测在 CPU 上可用 OpenCV Haar Cascade（轻量、~5ms/图）。若 Stage B 未跑，Stage C 可补跑。若人脸检测完全不可用，则退化为仅使用 aspect_ratio + edge_ratio + colorfulness 组合判断。

### 4.4 综合分函数

最终排序综合分 `composite_score` 不是单一实景置信度，而是三信号加权：

```
composite_score = w1 * realism_confidence 
                + w2 * quality_score 
                + w3 * display_fit_score
```

其中：
- `realism_confidence` ∈ [0, 1]：五级实景可信度归一化（NON_REAL=0, PROBABLY_NON_REAL=0.15, AMBIGUOUS=0.4, PROBABLY_REAL=0.7, REAL=1.0）
- `quality_score` ∈ [0, 1]：四级质量归一化（POOR=0, FAIR=0.35, GOOD=0.7, EXCELLENT=1.0）
- `display_fit_score` ∈ [0, 1]：显示适配度（横屏宽高比、合理分辨率、无极端裁切等归一化）
- 默认权重：`w1=0.5, w2=0.3, w3=0.2`

**权重可配置**（通过配置文件），视觉评审后可调整。

---

## 5. Step C4: 圆桌排序（Round-Robin Selection）

### 5.1 核心算法

这是确保多样性**不被单一分数碾压**的关键机制。

```
输入：
  - 去重后的候选池，已分配场景簇标签
  - 配额上限（每个簇的 max_count）
  - 每张图像的 composite_score

输出：
  - 有序的 top-100 列表

算法（Round-Robin with Priority）：

1. 按 composite_score 对每个簇内的图像降序排序
2. 维护每个簇的指针（初始 = 0）
3. 维护每个簇已选计数 selected_count[cluster] = 0
4. 重复直到选出 100 张或候选池耗尽：
   a. 找到当前未达配额的簇
   b. 在这些簇中，按 composite_score 选当前指针指向的图像
   c. 所选图像加入 final_list
   d. 对应簇的 selected_count++，指针++
   e. 若某个簇的指针已到簇尾，标记该簇为耗尽
5. 返回 final_list
```

### 5.2 簇间轮转顺序

不是简单的固定轮转，而是**分数加权轮转**：

```
每轮选择时，对未达配额的簇计算 priority_score：
  priority_score = max(0, composite_score[ptr] - decay_factor * selected_count[cluster])

其中 decay_factor = 0.02（可配置）
```

这样：
- 高综合分的图像仍有优势
- 但每多选一张同一簇的图像，该簇的下一张优先权衰减
- 当某个簇的配额已满，自动跳过

### 5.3 单人近景惩罚

在轮转过程中，额外维护 `portrait_count`：
- 若当前选择的图像是 portrait-like，portrait_count++
- 当 portrait_count ≥ 25，标记所有 portrait-like 图像为不可选（即使来自不同簇）

### 5.4 补充规则

- **最少代表**：若某个簇的综合分前 3 张都低于全局阈值（如 < 0.4），整个簇可以跳过
- **质量下限**：任何入选图像必须 quality_score ≥ 0.35（至少 FAIR）
- **人工保留位**：最末 5 个名额保留给"视觉独特但分数偏低"的样本（来自 score 排名 101~150 中 scene_cluster 未被充分代表的）

---

## 6. Step C5: 高相似集合的特殊处理

### 6.1 房产类数据集（如 housing_estate, property_interior）

**问题**：同一房产的多张照片非常相似（不同房间、不同角度但整体风格一致）。

**策略**：
1. **去重阈值收严**：threshold = 0.10（比默认 0.15 更严格）
2. **粗场景簇数量上限**：最多 20 张（配额 20% 而非 30%）
3. **细视点簇配额**：最多 8 张（而非 15 张）
4. **额外维度**：在特征向量中增加对**纹理频率**和**颜色直方图**的加权，使同一面白墙不同角度不再因特征相似被过度去重
5. **强制多样性**：若某个房产有 50 张照片，最多只能入选 8 张（在粗簇配额内再设子配额）

### 6.2 truro_school 数据集

**问题**：学校场景——教学楼正面、操场、教室内部、走廊——内容高度相似，大量重复角度。

**策略**：
1. **去重阈值放松**：threshold = 0.18（避免过度去重导致数量不足）
2. **但配额更严**：粗场景簇 ≤ 15 张（15%），细视点簇 ≤ 5 张
3. **KL 散度偏置检测**：计算候选池的 scene_cluster 分布，若 KL 散度显示某个簇占比 > 候选池的 40%，则自动将该簇的配额下调至 10%（从 30%）
4. **多样化注入**：从每个簇中选取综合分前 2 张 + 随机抽取 1 张（以发现意想不到的好图）
5. **回退路径**：若去重后候选池 < 100 张，降级到 threshold = 0.22 重新运行去重，但配额不变

### 6.3 检测逻辑（自适应）

不硬编码数据集名，而是**根据聚类结果自动识别**：

```
is_high_similarity = (
    num_samples_in_largest_cluster / total_samples > 0.35
    AND
    number_of_clusters < 4
    AND
    avg_pairwise_distance < 0.25
)

if is_high_similarity:
    apply_tier_2_restrictions  # 更严配额 + 更松去重阈值
```

---

## 7. Step C6: 边界样本复核池

### 7.1 何时进入复核池

| 条件 | 进入复核池 |
|------|-----------|
| 去重中被移除但综合分 ≥ 0.70 | 进入 Review Pool（可能被误去重） |
| 在 top-150 但未被选入 top-100（因配额限制） | 进入 Review Pool（人工可替换） |
| 粗场景簇的边界样本（DBSCAN 标记为 -1 的噪声点） | 进入 Review Pool（未分类但可能独特） |
| 单人近景检测为 True 但综合分 ≥ 0.85 | 进入 Review Pool（高质量人像可能值得破例） |

### 7.2 复核池结构

```
review_pool/
├── dedup_candidates.json     # 被去重的高分样本
├── quota_excluded.json       # 因配额未进入 top-100 的样本
├── unclassified.json         # DBSCAN 噪声点
└── portrait_exceptions.json  # 高质量人像候选
```

每张记录包含：文件名、综合分、场景簇、移除原因。

---

## 8. 输出目录与文件结构

### 8.1 输出目录

```
output/{dataset_name}/
├── stage3_dedup_diversity/
│   ├── config.json                   # 运行时配置（阈值、权重等）
│   ├── dedup_log.json                # 去重记录
│   ├── cluster_assignments.json      # 每张图像的簇标签
│   ├── cluster_summary.json          # 各簇统计
│   ├── quota_report.json             # 配额分配详情
│   ├── final_ranking.json            # top-100 排序结果（含分数明细）
│   ├── final_ranking_list.txt        # 纯文件名列表（一行一个，供 Executor 使用）
│   ├── review_pool.json              # 复核池
│   ├── review_sample_grid/           # 抽样复核图（每簇代表性图像）
│   │   ├── cluster_0_grid.jpg
│   │   ├── cluster_1_grid.jpg
│   │   └── ...
│   └── pipeline_log.json             # 全流程日志
```

### 8.2 final_ranking.json 格式

```json
{
  "dataset": "housing_estate",
  "total_candidates": 850,
  "after_dedup": 720,
  "final_count": 100,
  "clusters_found": 7,
  "thresholds": {
    "dedup_cosine_distance": 0.10,
    "min_composite_score": 0.35,
    "max_per_cluster": 20
  },
  "rankings": [
    {
      "rank": 1,
      "filename": "img_0421.jpg",
      "composite_score": 0.87,
      "realism": "REAL",
      "quality": "EXCELLENT",
      "display_fit": 0.91,
      "cluster": {
        "scene": 2,
        "viewpoint": 0,
        "scene_size": 45,
        "viewpoint_size": 12
      },
      "is_portrait_like": false
    }
  ]
}
```

### 8.3 日志格式

```json
{
  "timestamp": "2025-07-11T14:30:00Z",
  "phase": "dedup",
  "event": "removed_near_duplicate",
  "kept": "img_0387.jpg",
  "removed": "img_0421.jpg",
  "distance": 0.12,
  "threshold": 0.15,
  "cluster": 2
}
```

---

## 9. 质量控制与可复现性

### 9.1 固定 seed
```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

所有涉及随机性的步骤（K-means 初始中心、随机采样等）使用此 seed。

### 9.2 断点续跑
每个步骤的输出都写入文件。若步骤输出文件已存在，跳过该步骤（通过 `--resume` 标志控制）。

### 9.3 抽样复核

| 复核类型 | 数量 | 方式 |
|----------|------|------|
| 去重正确性 | 10 组 (kept, removed) | 用 grid 图像对比 |
| 聚类合理性 | 每簇 3 张 | 拼图展示簇内一致性 |
| top-100 多样性 | 按簇 + 视点分布检查 | 自动生成统计报告 |
| 单人近景检测 | 所有标记为 portrait 的样本 | 人工快速浏览确认 |

### 9.4 缺失数据处理
- 若 Stage B 未输出特征向量 → 该图像被标记为 `feature_missing`，不参与去重/聚类，仅根据综合分从底部补位
- 若某个簇在配额限制后为 0 张 → 在日志中 WARNING 记录
- 若最终 top-100 不足 100 张 → 报告真实数量，不填充

---

## 10. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 特征向量区分度低 | 不同场景被聚到同一簇 | 降低 eps，或增加特征维度 |
| 单人近景误判 | 高质量人像被排除 | 复核池可恢复，阈值可调 |
| 房产数据过度去重 | 同一房产只剩 1~2 张 | 放松去重阈值 + 子配额 |
| truro_school 候选池不足 100 | 无法凑满 top-100 | 报告真实数量，不填充 |
| 聚类结果不稳定 | 每次运行簇标签不同 | 固定 seed + 记录簇中心 |
| 高相似集合未被自动识别 | 多样性不足 | 留复核池拦截 |

---

## 附录 A：降级聚类方案（纯 numpy + scipy）

若 DBSCAN/HDBSCAN 不可用：

### A.1 基于距离阈值的贪婪聚类
```python
def greedy_cluster(features, threshold=0.25):
    """贪心聚类：从最高分样本开始，聚合距离 < threshold 的样本"""
    n = len(features)
    assigned = [False] * n
    clusters = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i+1, n):
            if not assigned[j]:
                d = cosine_distance(features[i], features[j])
                if d < threshold:
                    cluster.append(j)
                    assigned[j] = True
        clusters.append(cluster)
    return clusters
```

### A.2 基于 K-means + 层次聚类
```python
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

# 粗聚类
kmeans = KMeans(n_clusters=max(5, n//50), random_state=42)
coarse_labels = kmeans.fit_predict(features)

# 对每个粗簇做细聚类
fine_labels = {}
for c in set(coarse_labels):
    mask = coarse_labels == c
    if sum(mask) <= 1:
        fine_labels[c] = [0]  # 单样本簇
    else:
        Z = linkage(features[mask], method='ward')
        fine_labels[c] = fcluster(Z, t=min(5, sum(mask)//2), criterion='maxclust')
```

---

## 附录 B：配置示例

```json
{
  "_comment": "Stage C 配置文件 - diversity-ranking-planner",
  "seed": 42,
  "dedup": {
    "method": "greedy_density_sphere",
    "distance_metric": "cosine",
    "default_threshold": 0.15,
    "adaptive_percentile": 5,
    "adaptive_multiplier": 1.5,
    "min_threshold": 0.08,
    "max_threshold": 0.25,
    "high_similarity_threshold": 0.10
  },
  "clustering": {
    "method": "dbscan",
    "eps_auto_percentile": 10,
    "eps_auto_multiplier": 2.0,
    "min_samples": 2,
    "fallback_method": "kmeans_hierarchy"
  },
  "quota": {
    "max_per_scene_cluster": 30,
    "max_per_viewpoint_cluster": 15,
    "max_portrait_like": 25,
    "min_scene_clusters_represented": 3,
    "high_similarity_max_per_scene": 20,
    "high_similarity_max_per_viewpoint": 8,
    "truro_school_max_per_scene": 15,
    "truro_school_max_per_viewpoint": 5,
    "kl_divergence_threshold": 0.4
  },
  "ranking": {
    "weights": {"realism": 0.5, "quality": 0.3, "display_fit": 0.2},
    "round_robin_decay": 0.02,
    "min_quality_threshold": 0.35,
    "min_composite_for_auto_entry": 0.4,
    "human_reserve_slots": 5
  },
  "portrait_detection": {
    "aspect_ratio_min": 0.7,
    "aspect_ratio_max": 1.5,
    "min_side_px": 300,
    "max_edge_ratio": 0.05,
    "max_colorfulness": 20,
    "face_area_ratio_min": 0.20
  },
  "review_pool": {
    "include_dedup_high_score": true,
    "dedup_high_score_threshold": 0.70,
    "include_quota_excluded": true,
    "quota_excluded_range": 150,
    "include_unclassified": true,
    "include_portrait_exceptions": true,
    "portrait_exception_threshold": 0.85
  },
  "output": {
    "grid_sample_per_cluster": 3,
    "save_intermediate": true
  }
}
```

---

*本文档是 Stage 2 规划方案中"去重、多样性与排序"部分的详细展开。配合 visual_definition_framework.md 和 stage2_pipeline_plan.md 共同构成完整交付。*
