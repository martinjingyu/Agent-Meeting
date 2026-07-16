# Executor: Stage C 去重、多样性、排序 — 可执行规划方案

> 从 Stage B 输出的候选池（含 visual feature vector、realism_confidence、quality_score、display_fit_score）出发，为每个数据集输出 ≤100 张 gallery 图片 + 复核池 + 可解释字段。
> 
> 本方案兼容：房产类（室内室外高度重复）、学校活动类（大量相似合影/活动照）、信息图混杂类（掺插图文图表）—— 均不依赖语义标签。

---

## 1. 输入输出

### 输入（Stage B 输出格式，每个数据集一个 CSV）

| 字段 | 类型 | 说明 |
|------|------|------|
| filepath | str | 绝对路径 |
| img_width, img_height | int | 像素 |
| feature_vector | float[576] | MobileNetV3-Small 倒数第二层 embedding |
| realism_confidence | float [0,1] | Stage B 分类器输出 |
| quality_score | float [0,1] | Stage B 质量分支输出 |
| display_fit_score | float [0,1] | Stage B 显示适配分支输出 |
| stage_a_flags | str | Stage A 通过的信号标记 |

### 输出（每个数据集）

1. **`{dataset}/top_100_gallery.csv`** — 最终入选的 ≤100 张
2. **`{dataset}/review_pool.csv`** — 复核池（人工可替换用）
3. **`{dataset}/dedup_diversity_report.json`** — 去重/聚类/配额统计
4. **`pipeline_stageC_state.json`** — 断点续跑状态

---

## 2. Stage C 子步骤（总计时 ~10–15 min CPU）

```
Stage B CSV
    │
    ▼
C1: 复合分计算 & 排序
    │
    ▼
C2: 近似去重（贪心密度球）
    │
    ▼
C3: 场景/主体聚类（DBSCAN on 特征向量）
    │
    ▼
C4: 单人近景检测（Haar Cascade）
    │
    ▼
C5: 多样性配额 + 圆桌轮转排序
    │
    ▼
C6: 复核池构建 + 可解释字段填充
    │
    ▼
C7: 统计报告输出
```

---

## 3. C1 — 复合分计算

```python
composite_score = (
    w_realism * realism_confidence +
    w_quality * quality_score +
    w_display * display_fit_score
)
```

**默认权重**（可配置于 `pipeline_config.yaml`）：

| 参数 | 默认值 | 理由 |
|------|--------|------|
| w_realism | 0.50 | 实景可信度最重要 |
| w_quality | 0.30 | 展示质量次之 |
| w_display | 0.20 | 显示适配作为加分 |

> 注意：composite_score 只用于簇内排序和复核池门槛判定，**不直接决定 top-100 入选**。入选由配额+圆桌机制决定。

---

## 4. C2 — 近似去重（贪心密度球）

### 算法

```
输入: N 张图片, 每张有 feature_vector + composite_score
步骤:
1. 按 composite_score 降序排列候选池
2. 初始化 kept = []
3. 遍历每张图 img_i:
   a. 计算 img_i 与 kept 中所有图的 cosine_distance
   b. 若 min_distance >= dedup_threshold: kept.append(img_i)
   c. 否则: 丢弃到 review_pool（标记原因="近似去重"）
4. 返回 kept 作为去重后的候选池
```

### 阈值策略（per-dataset 自适应）

| 数据集类型 | 阈值 | 理由 |
|-----------|------|------|
| 房产类 (real_estate, digital_domain) | 0.10 | 室内外场景高度重复，严去重 |
| 普通数据集 (design, ecommerce) | 0.12 | 中等多样性 |
| 学校活动类 (truro_school) | 0.18 | 大量相似合影，太严则不足100 |
| 信息图混杂类 | 0.08 | 图文/图表需明确区隔 |

**自适应计算**（当 per-dataset 类型未明确时）：
```python
# 取第 5 百分位的 pairwise cosine distance 作为 base
all_distances = pairwise_cosine_distances(feature_vectors)
base = percentile(all_distances, 5)
dedup_threshold = min(max(base * 1.5, 0.08), 0.25)

# 特殊: truro_school 风格自动检测（若 max_cluster_ratio>0.5 且 avg_pairwise_dist<0.20）
if is_highly_repetitive(feature_vectors):
    dedup_threshold = max(dedup_threshold, 0.18)
```

### 去重记录字段

每张被去重的图记录：
- `dedup_reason`: "近似去重"
- `dedup_kept_neighbor`: 保留的那张图的 filepath
- `dedup_distance`: cosine_distance 值

---

## 5. C3 — 场景/主体聚类（两层级联）

### 粗场景簇（Scene Cluster）

```
算法: DBSCAN
特征: 576-dim feature_vector (PCA 压缩到 64-dim 可加速，可选)
metric: cosine
eps: 0.20 (自适应见下)
min_samples: 3
```

**eps 自适应**:
```python
# 计算候选池的平均 pairwise distance 的 30 百分位
avg_dist_p30 = percentile(pairwise_distances, 30)
eps = max(0.15, min(avg_dist_p30 * 0.8, 0.35))
```

### 细视点簇（Viewpoint Sub-cluster）

对于粗场景簇内图片数量 > 15 的大簇，执行二次聚类：

```
算法: DBSCAN (same feature vector)
metric: cosine
eps: 0.10（固定，更精细）
min_samples: 2
```

### 噪声点处理

DBSCAN 标记为 -1（噪声）的图片：
- 不强制丢弃，作为独立簇（size=1）参与配额分配
- 在复核池中标记为 "DBSCAN噪声点"

---

## 6. C4 — 单人近景检测

### 检测方法（纯视觉，CPU 5ms/张）

```python
# 使用 OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

is_portrait = False
if len(faces) == 1:
    x, y, w, h = faces[0]
    face_ratio = (w * h) / (img_width * img_height)
    aspect_ratio = img_width / img_height
    short_edge = min(img_width, img_height)
    
    # 单人近景判定条件（满足以下多数）：
    is_portrait = (
        face_ratio > 0.08          # 面部占画面 > 8%
        and abs(aspect_ratio - 0.75) < 0.3  # 接近 3:4 或 4:3 的人像比例
        and short_edge >= 300       # 非缩略图
        and len(faces) == 1         # 仅一人脸
    )
```

### 为何不用 deepface / 更重的人脸模型
- CPU-only + 速度约束：Haar Cascade 5ms/张
- 我们不需要识别身份，只需检测"单人近景过载"风险
- 边界案例（多人照中的单人特写）由配额兜底

---

## 7. C5 — 多样性配额 + 圆桌轮转排序（核心）

### 7.1 配额层级

| 层级 | 上限 | 说明 |
|------|------|------|
| 粗场景簇 | ≤ 30张 (30%) | 同一场景类型不超过 30% |
| 细视点簇 | ≤ 15张 (15%) | 同一视角/站位不超过 15% |
| 单人近景 | ≤ 25张 (25%) | 单人特写不超过 25% |
| 最低场景簇数 | ≥ 3个 | 至少来自 3 个不同场景簇 |

### 7.2 高相似集合自适应（当候选池多样性不足时）

检测条件（自动触发更严配额）：
```
max_cluster_ratio > 0.35       # 最大簇占比 > 35%
AND num_clusters < 4           # 总簇数 < 4
AND avg_pairwise_distance < 0.25  # 整体相似度高
```

触发后的严配额：
```
场景簇上限 → 20张 (20%)
视点簇上限 → 8张 (8%)
单人近景上限 → 15张 (15%)
```

### 7.3 KL 散度偏置检测

对每个粗场景簇，计算其 composite_score 分布相对于全候选池的 KL 散度：
```python
kl_div = kl_divergence(cluster_score_dist, global_score_dist)
if kl_div > 0.5:
    # 该簇分数分布偏差大 → 下调该簇配额 × 0.8
    scene_quota[cluster_id] = int(scene_quota[cluster_id] * 0.8)
```

### 7.4 圆桌轮转（Round-Robin with Priority）— 最终排序算法

这是**取代纯分数排序**的核心机制。

#### 数据准备

```
输入: 去重后的候选池
      - 每张有 composite_score, scene_cluster_id, view_cluster_id, is_portrait
      - 每个 scene_cluster_id 有配额上限 scene_quota[cluster]
      - 每个 view_cluster_id 有配额上限 view_quota[cluster]
      - 全局单人近景配额上限 portrait_quota = 25
```

#### 算法

```
1. 在每个簇内，按 composite_score 降序排列该簇的图片列表
2. 维护每个簇的指针 ptr[cluster] = 0
3. 维护每个簇已选计数 selected_scene[cluster] = 0
4. 维护每个细视点簇已选计数 selected_view[view_cluster] = 0
5. 维护已选单人近景计数 selected_portrait = 0
6. 初始化 distance_decay = {}  # 记录每张图上轮被跳过的次数

7. 循环直到选满 100 张 或 所有簇耗尽：
   a. 计算每个未达配额簇的 priority_score:
      priority = max(0, composite_score[cluster[ptr]] - decay_factor × selected_scene[cluster])
      decay_factor = 0.02（可配置）
   
   b. 选 priority_score 最高的簇的当前 ptr 指向的图片
   
   c. 检查配额约束：
      - 若选中后 selected_scene[cluster] > scene_quota[cluster] → 跳过该簇
      - 若选中后 selected_view[view_cluster] > view_quota[view_cluster] → 跳过该簇
      - 若 is_portrait 且 selected_portrait + 1 > portrait_quota → 跳过该图
      - 跳过时，该簇 ptr 不动但记录一次跳过（distance_decay[cluster] += 1）
   
   d. 真实选中：
      - 图片加入 final_list
      - selected_scene[cluster] += 1
      - selected_view[view_cluster] += 1
      - if is_portrait: selected_portrait += 1
      - ptr[cluster] += 1
      - 若 ptr[cluster] >= len(cluster_images[cluster])，标记该簇为耗尽
      
8. 若循环结束时 final_list < 100：
   - 放宽单人近景配额到 35
   - 从被配额跳过的图片中按 composite_score 补选
   - 记录日志: "数据集 {dataset} 最终入选 {n} 张，未达 100"

9. 返回 final_list
```

### 7.5 truro_school 特殊策略

当数据集检测为 "学校活动类"（由 Stage A 的 heuristic 信号比率判断）：
```
dedup_threshold = 0.18
场景簇上限 = 15张
视点簇上限 = 5张
单人近景上限 = 20张

每簇选择逻辑：top-2 by composite_score + 随机 1 张（m=1, seed=42）
若最终 < 100，降级重跑：dedup_threshold = 0.22
```

---

## 8. C6 — 复核池构建

以下图片进入 review_pool（每张标注入选或落选原因）：

| 来源 | 原因标记 | 说明 |
|------|---------|------|
| 被去重的高分样本（composite ≥ 0.70） | "高分被去重" | 人工可替换近似重复 |
| 因配额未入选的 top-150（去重后） | "配额限制" | 若有替换需求 |
| DBSCAN 噪声点 | "DBSCAN噪声点" | 质量好但无法聚类的图 |
| 高质量人像（composite ≥ 0.85 且被 quota 拦下） | "高质人像配额" | 可能适合 gallery |
| 因单人近景配额未入选的图片 | "单人近景配额" | 人工评估是否保留 |
| Stage A/B 边界样本（AMBIGUOUS + FAIR 以上） | "边界样本待复核" | 人眼判断 |

复核池大小建议：每数据集 ≤ 150 张（含去重落选）。

---

## 9. C7 — 可解释输出字段

### top_100_gallery.csv 每行

| 字段 | 示例 | 说明 |
|------|------|------|
| rank | 1 | 最终入选序号（1-based） |
| filepath | C:\pics\real_estate\img042.jpg | |
| composite_score | 0.823 | 见 C1 |
| realism_confidence | 0.91 | Stage B 输出 |
| quality_score | 0.78 | Stage B 输出 |
| display_fit_score | 0.75 | Stage B 输出 |
| scene_cluster_id | 3 | C3 粗场景簇编号 |
| view_cluster_id | 3-1 | C3 细视点簇编号 |
| is_portrait | False | C4 是否单人近景 |
| selected_reason | "场景配额+圆桌入选" | 为什么入选 |
| dedup_passed | True | 是否通过 C2 去重 |
| dedup_neighbor | "" | 若被去重，保留的邻居路径 |

### review_pool.csv 每行

| 字段 | 示例 |
|------|------|
| filepath | C:\pics\... |
| composite_score | 0.812 |
| not_selected_reason | "高分被去重" |
| dedup_distance | 0.09 |
| dedup_kept_neighbor | C:\pics\...\kept_img.jpg |
| scene_cluster_id | 3 |
| review_priority | "高" | 基于 composite_score 分档：≥0.8→高, 0.7-0.8→中, <0.7→低 |

### dedup_diversity_report.json 统计

```json
{
  "dataset": "real_estate",
  "total_input": 5200,
  "survived_stage_a": 2400,
  "survived_stage_b": 2100,
  "after_dedup": 890,
  "final_gallery_count": 100,
  "scene_clusters": {
    "total": 8,
    "distribution": {"0": 30, "1": 22, "2": 15, "3": 10, "4": 8, "5": 7, "6": 5, "7": 3},
    "max_ratio": 0.30
  },
  "portrait_count": 22,
  "dedup_statistics": {
    "total_removed": 1210,
    "high_score_removed": 45,
    "threshold_used": 0.10
  },
  "review_pool_size": 142,
  "quota_used": "normal",
  "quota_triggered_tight": false,
  "warnings": []
}
```

---

## 10. 跨数据集兼容性说明

| 数据集类型 | 核心挑战 | 应对策略 |
|-----------|---------|---------|
| **real_estate**（房产，~5000张） | 大量相似室内外角度 | dedup_threshold=0.10 严去重；视图点簇二次聚类区分不同房间 |
| **truro_school**（学校，~7650张） | 大量合影/活动照；59%通过率 | dedup_threshold=0.18 松；每簇 top-2+random1；配额更紧 |
| **digital_domain**（CGI，~2400张） | 高质量渲染易误判 | 标记高风险；AMBIGUOUS 全进复核池；去重阈值 0.10 |
| **design**（设计，~5800张） | 信息图/图文混杂 | dedup_threshold=0.08 严；PCA 特征增强区分纯图文 |
| **ecommerce**（电商，~600张） | 白底产品图过多 | 质量分对白底图降权；场景聚类天然分离不同产品 |
| **普通数据集** | 多样性适中 | 默认参数 0.12 |

---

## 11. 脚本模块结构

```
pipeline/
├── stage_c/
│   ├── __init__.py
│   ├── compute_composite.py   # C1 复合分
│   ├── dedup.py               # C2 贪心密度球去重
│   ├── cluster.py             # C3 DBSCAN 两层级联聚类
│   ├── face_detector.py       # C4 Haar Cascade 单人近景
│   ├── quota_manager.py       # C5 配额计算 + KL 散度偏置检测
│   ├── round_robin.py         # C5 圆桌轮转排序
│   ├── review_pool.py         # C6 复核池构建
│   ├── report.py              # C7 统计 JSON + CSV 输出
│   └── run_stage_c.py         # Stage C 主入口（单数据集执行）
├── config/
│   └── pipeline_config.yaml   # 全局配置 + per-dataset 覆盖
├── run_pipeline.py            # 主入口（全数据集循环 + 断点续跑）
└── pipeline_state.json        # 断点状态文件
```

---

## 12. 断点续跑设计

`pipeline_state.json` 内容：
```json
{
  "completed_datasets": ["real_estate", "design"],
  "current_dataset": "truro_school",
  "current_step": "C5",
  "intermediate_files": {
    "truro_school": {
      "post_dedup": "cache/truro_school/post_dedup.csv"
    }
  },
  "started_at": "2026-07-15T18:30:00",
  "last_updated": "2026-07-15T19:45:00"
}
```

启动时检测：若 `current_dataset` 非空，从中断步骤恢复，不再重跑已完成数据集。

---

## 13. 风险说明

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| DBSCAN eps 自适应不当导致欠/过聚类 | 中 | 配额失效 | 日志输出 eps 值和簇分布；允许人工调整 eps 后重跑 C3 |
| Haar Cascade 漏检/误检多人近景 | 低 | 单人近景配额不精确 | 配额是软上限，漏检影响小；复核池可人工纠正 |
| truro_school 最终不足 100 张 | 中 | 展示空缺 | 降级阈值重跑；最终如实报告数量 |
| 某数据集去重后候选池 < 100 | 低 | 无法达满额 | 如实报告数量，不补入低质量图片（保守原则） |
| 不同数据集间特征分布差异大 | 中 | 聚类质量波动 | per-dataset 自适应 eps；report 中输出 DBSCAN 质量指标 |

---

## 14. Executor 执行步骤

```
1. pip install opencv-python numpy scikit-learn scikit-image pyyaml
   # （Stage A/B 依赖已安装，Stage C 新增 scikit-learn）

2. 确认 Stage B 输出 CSV 格式与预期一致
   # 字段: filepath, feature_vector, realism_confidence, quality_score, display_fit_score

3. 配置 pipeline_config.yaml
   # 设置 per-dataset 的 dedup_threshold、scene_quota、专属 flag

4. python pipeline/run_pipeline.py --stage C --datasets all
   # 执行全量 Stage C，~10-15 min

5. 检查各数据集输出:
   - logs/stage_c/*.log
   - outputs/{dataset}/dedup_diversity_report.json
   - outputs/{dataset}/top_100_gallery.csv
   - outputs/{dataset}/review_pool.csv

6. 人工复核 review_pool，决定替换
   # 替换操作：从 review_pool 选图替换 top_100 中指定 rank 的图片

7. (可选) python scripts/generate_report.py --html
   # 生成 HTML gallery 预览
```

---

## 15. 附录：配置模板（pipeline_config.yaml）

```yaml
stage_c:
  composite_weights:
    w_realism: 0.50
    w_quality: 0.30
    w_display: 0.20

  dedup:
    default_threshold: 0.12
    threshold_range: [0.08, 0.25]
    percentile_base: 0.05
    percentile_multiplier: 1.5

  clustering:
    dbscan_eps_ratio: 0.8
    dbscan_eps_percentile: 30
    dbscan_eps_range: [0.15, 0.35]
    min_samples: 3
    subcluster_eps: 0.10
    subcluster_min_samples: 2

  quotas:
    scene_cluster_max: 30
    view_cluster_max: 15
    portrait_max: 25
    min_scene_clusters: 3
    
    tight_mode:
      scene_cluster_max: 20
      view_cluster_max: 8
      portrait_max: 15
    tight_trigger:
      max_cluster_ratio: 0.35
      min_clusters: 4
      avg_pairwise_dist: 0.25

  round_robin:
    decay_factor: 0.02
    max_iterations: 500

  review_pool:
    high_score_threshold: 0.70
    top_n_for_quota: 150
    high_quality_portrait_threshold: 0.85

  per_dataset_overrides:
    real_estate:
      dedup_threshold: 0.10
    truro_school:
      dedup_threshold: 0.18
      scene_cluster_max: 15
      view_cluster_max: 5
      portrait_max: 20
      cluster_sample_strategy: "top2_plus_random1"
      dedup_retry_threshold: 0.22
    digital_domain:
      dedup_threshold: 0.10
      review_all_ambiguous: true
    design:
      dedup_threshold: 0.08
    ecommerce:
      dedup_threshold: 0.12
```

---

> **作者**: diversity-ranking-planner  
> **基于**: mtg_215f666019 视觉判据专家三轴框架 + CPU 流水线专家三级级联 + Stage 1 探索结论  
> **本方案已做**: 去重阈值自适应、聚类 eps 自适应、高相似集自动严配额、KL 散度偏置检测、truro_school 特殊处理、单人近景检测（纯视觉）、圆桌轮转（非纯分数排序）、复核池标注原因、可解释输出字段
