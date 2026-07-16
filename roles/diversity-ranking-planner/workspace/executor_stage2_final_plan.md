# Stage 2 Executor 可直接执行方案：去重、多样性、排序

> **基于**：视觉判据专家三轴框架 + CPU 流水线专家三级级联 + 怀疑论评审修正  
> **意图**：给出可直接编码执行的算法伪代码、参数阈值、输出格式、复核机制  
> **原则**：保守偏置（不足100如实报、不补低质图）、禁止纯分数排序、仅视觉内容为依据

---

## 0. 核心流水线概述

```
Stage B 候选池（≤2,000张/数据集，含feature_vector + 三轴分）
  │
  ▼
C1 — 复合分计算 + 去重前排序
  │
  ▼
C2 — 三信号组合近似去重（贪心密度球）
  │  dhash(0.35) + cosine(0.40) + edge_hist(0.25) → combined_distance
  │
  ▼
C3 — 双阶段场景聚类（先dhash桶 → 桶组 → DBSCAN）
  │
  ▼
C4 — 单人近景弱惩罚评分（多信号合成 portrait_penalty ∈ [0,0.15]）
  │
  ▼
C5 — 多样性配额 + 圆桌轮转排序（核心：非纯分数排序）
  │  • 桶配额 ≤5张  • 簇配额 ≤25张  • 全局 ≤100张
  │  • decay_factor=0.02  • adjusted_composite 参与轮转
  │
  ▼
C6 — 复核池构建 + 落选原因标准化编码
  │
  ▼
C7 — 输出文件（top_100_gallery.csv + review_pool.csv + report.json）
```

---

## 1. C1 — 复合分计算

### 1.1 输入字段（来自 Stage B）

| 字段 | 说明 | 来源 |
|------|------|------|
| `feature_vector` | 576-dim float[576] | MobileNetV3-Small penultimate layer |
| `realism_confidence` | float [0,1] | Stage B 实景分类器 |
| `quality_score` | float [0,1] | Stage B 质量分支 |
| `display_fit_score` | float [0,1] | Stage B 显示适配分支 |

### 1.2 复合分公式

```python
composite_score = (
    0.50 * realism_confidence +
    0.30 * quality_score +
    0.20 * display_fit_score
)
```

### 1.3 去重前排序

按 `composite_score` 降序排列候选池。此排序仅用于 C2 贪心去重（高分优先保留），**不代表最终 top-100 排名**。

---

## 2. C2 — 三信号组合近似去重

### 2.1 三信号定义

| 信号 | 权重 | 方法 | 计算量 |
|------|------|------|--------|
| **dhash** | 0.35 | 缩放到 9×8 → 差异位 → 64bit → Hamming/64 | 0.5ms/张 |
| **cosine(feature)** | 0.40 | Stage B 已有的 576-dim 特征向量 | 0.02ms/对 |
| **边缘直方图** | 0.25 | Canny → 8×8 网格边缘密度 → L1 | 0.3ms/张 |

### 2.2 组合距离

```python
def compute_combined_distance(img_a, img_b, feat_a, feat_b):
    # 1. dhash 距离
    dhash_a = dhash(img_a)        # 64-bit int
    dhash_b = dhash(img_b)
    dhash_dist = bin(dhash_a ^ dhash_b).count('1') / 64.0  # [0,1]
    
    # 2. cosine 距离（复用 Stage B 特征）
    dot = np.dot(feat_a, feat_b)
    norm_a = np.linalg.norm(feat_a)
    norm_b = np.linalg.norm(feat_b)
    cosine_dist = 1.0 - (dot / (norm_a * norm_b + 1e-10))  # [0,2], 截断到 [0,1]
    cosine_dist = min(cosine_dist, 1.0)
    
    # 3. 边缘直方图距离
    hist_a = edge_histogram(img_a)   # 64-dim (8×8 网格)
    hist_b = edge_histogram(img_b)
    edge_dist = np.sum(np.abs(hist_a - hist_b)) / 64.0  # [0,1]
    
    # 组合
    combined = (
        0.35 * dhash_dist +
        0.40 * cosine_dist +
        0.25 * edge_dist
    )
    return combined, {"dhash": dhash_dist, "cosine": cosine_dist, "edge": edge_dist}
```

### 2.3 去重算法：贪心密度球

```python
def greedy_dedup(images, threshold):
    """
    images: list of dicts {composite_score, filepath, ...}, pre-sorted desc
    threshold: combined_distance 阈值
    """
    kept = []
    removed = []
    kept_features = []
    
    for img in images:
        feat = img["feature_vector"]
        min_dist = float('inf')
        nearest_kept = None
        sub_scores = None
        
        for k_idx, k_feat in enumerate(kept_features):
            d, subs = compute_combined_distance(img, kept[k_idx], feat, k_feat)
            if d < min_dist:
                min_dist = d
                nearest_kept = kept[k_idx]
                sub_scores = subs
        
        if min_dist < threshold:
            removed.append({
                **img,
                "reason": "DEDUP_DENSE",
                "combined_distance": min_dist,
                "kept_neighbor": nearest_kept["filepath"],
                "sub_scores": sub_scores
            })
        else:
            kept.append(img)
            kept_features.append(feat)
    
    return kept, removed
```

### 2.4 阈值自适应

```python
def auto_dedup_threshold(all_images):
    """自适应计算去重阈值"""
    # 采样最多 500 对计算 pairwise distances
    n = min(len(all_images), 500)
    sampled = all_images[:n]
    distances = []
    for i in range(min(n, 100)):   # 限制 O(n²) 到约 5k 对
        for j in range(i+1, min(n, 100)):
            d, _ = compute_combined_distance(sampled[i], sampled[j],
                                             sampled[i]["feature_vector"],
                                             sampled[j]["feature_vector"])
            distances.append(d)
    
    if len(distances) < 10:
        return 0.14  # 默认值
    
    base = np.percentile(distances, 10)
    threshold = max(0.08, min(base * 1.2, 0.30))
    return threshold
```

### 2.5 per-dataset 阈值覆盖

| 数据集 | threshold | 理由 |
|--------|-----------|------|
| 默认 | 0.14 | 中等多样性通用值 |
| real_estate | 0.10 | 室内场景高度重复，严去重 |
| digital_domain | 0.10 | CGI 渲染图相似度高 |
| truro_school | 0.22 | 大量合影/活动照，松去重避免不足100 |
| design | 0.08 | 图文/图表混杂，需明确区隔 |

---

## 3. C3 — 双阶段场景聚类

### 3.1 第一阶段：dhash 粗分桶

```python
def dhash_bucket(img):
    """9×8 → 64-bit dhash"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_LINEAR)
    diff = resized[:, 1:] > resized[:, :-1]  # 8×8 差异矩阵
    bits = 0
    for i in range(8):
        for j in range(8):
            bits = (bits << 1) | (1 if diff[i, j] else 0)
    return bits
```

```python
# 合并相邻桶：Hamming 距离 ≤ merge_threshold 的桶视为同组
def merge_buckets(buckets, merge_threshold=4):
    """buckets: dict {bucket_id: [image_indices]}"""
    bucket_ids = list(buckets.keys())
    bucket_groups = []  # list of lists of bucket_ids
    assigned = set()
    
    for i, bid in enumerate(bucket_ids):
        if bid in assigned:
            continue
        group = [bid]
        assigned.add(bid)
        for j in range(i+1, len(bucket_ids)):
            if bucket_ids[j] in assigned:
                continue
            d = hamming_distance(bid, bucket_ids[j])
            if d <= merge_threshold:
                group.append(bucket_ids[j])
                assigned.add(bucket_ids[j])
        bucket_groups.append(group)
    
    # 构建最终桶组
    groups = []
    for bg in bucket_groups:
        group_images = []
        for bid in bg:
            group_images.extend(buckets[bid])
        groups.append(group_images)
    return groups
```

### 3.2 第二阶段：桶组内 DBSCAN + 跨桶合并

```python
def cluster_images(images, groups):
    """
    groups: list of lists of image indices (来自 dhash 桶组)
    返回: cluster_id 映射
    """
    n = len(images)
    cluster_ids = [-1] * n
    
    # 桶内 DBSCAN
    sub_cluster_counter = 0
    for group in groups:
        if len(group) <= 5:
            # 小桶组直接视为一个细簇
            for idx in group:
                cluster_ids[idx] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # 大桶组内 DBSCAN
            feats = np.array([images[idx]["feature_vector"] for idx in group])
            from sklearn.cluster import DBSCAN
            db = DBSCAN(metric='cosine', eps=0.15, min_samples=2).fit(feats)
            for i, idx in enumerate(group):
                label = db.labels_[i]
                if label == -1:
                    cluster_ids[idx] = -1  # 噪声
                else:
                    cluster_ids[idx] = sub_cluster_counter + label
            sub_cluster_counter += len(set(db.labels_[db.labels_ >= 0]))
    
    # 跨桶合并：对每个非噪声簇计算质心，再做一次 DBSCAN
    valid_clusters = set(c for c in cluster_ids if c >= 0)
    centroids = {}
    for c in valid_clusters:
        indices = [i for i, cid in enumerate(cluster_ids) if cid == c]
        feats = np.array([images[i]["feature_vector"] for i in indices])
        centroids[c] = np.median(feats, axis=0)
    
    if len(centroids) >= 2:
        centroid_ids = list(centroids.keys())
        centroid_feats = np.array([centroids[c] for c in centroid_ids])
        db2 = DBSCAN(metric='cosine', eps=0.20, min_samples=1).fit(centroid_feats)
        
        # 重新映射 cluster_id
        merged_cluster_counter = 0
        mapping = {}
        for i, cid in enumerate(centroid_ids):
            new_id = db2.labels_[i]
            if new_id not in mapping:
                mapping[new_id] = merged_cluster_counter
                merged_cluster_counter += 1
            new_cid = mapping[new_id]
            # 更新所有属于该簇的图片
            for j, cid_j in enumerate(cluster_ids):
                if cid_j == cid:
                    cluster_ids[j] = new_cid
    
    return cluster_ids
```

### 3.3 噪声点处理

DBSCAN 标记为 -1 的图片：
- composite_score ≥ 0.6：保留为单张独立簇进入配额轮转
- composite_score < 0.6：直接进入复核池（NOISE_CLUSTER）

---

## 4. C4 — 单人近景弱惩罚评分

### 4.1 三信号合成 portrait_penalty

**不依赖 Haar Cascade 稳定性**，使用多信号连续惩罚：

```python
def portrait_penalty(img):
    """返回 [0, 0.15] 的连续惩罚值"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # ── 信号1：边缘集中度 (weight 0.40) ──
    edges = cv2.Canny(gray, 50, 150)
    cx, cy = w // 2, h // 2
    crop_h, crop_w = int(h * 0.2), int(w * 0.2)
    center = edges[cy-crop_h:cy+crop_h, cx-crop_w:cx+crop_w]
    
    center_density = center.sum() / max(center.size, 1) if center.size > 0 else 0
    full_density = edges.sum() / (h * w)
    concentration = center_density / max(full_density, 1e-6)
    s1 = min(max((concentration - 2.0) / 4.0, 0), 1.0)
    
    # ── 信号2：宽高比 (weight 0.30) ──
    aspect = w / h
    if 0.6 <= aspect <= 1.1:
        s2 = 1.0 - abs(aspect - 0.75) / 0.35
    else:
        s2 = 0.0
    
    # ── 信号3：面部检测 (weight 0.30, 可选) ──
    s3 = 0.0
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 1:
            x, y, fw, fh = faces[0]
            face_ratio = (fw * fh) / (w * h)
            if face_ratio > 0.05:
                s3 = min(face_ratio / 0.2, 1.0)
    except Exception:
        pass  # 无检测器时 s3=0, 自动降权
    
    # ── 综合 ──
    penalty = 0.40 * s1 + 0.30 * s2 + 0.30 * s3
    return min(penalty * 0.15, 0.15)
```

### 4.2 调整后的复合分

```python
adjusted_composite = composite_score - portrait_penalty
```

- 单人近景明显：penalty ≈ 0.10–0.15 → 排名自然下降
- 群像/风景/建筑：penalty ≈ 0.00–0.03 → 不受影响
- **不硬阻断**：高分单人照仍可能入选，但竞争力降低

---

## 5. C5 — 多样性配额 + 圆桌轮转排序（核心）

### 5.1 配额层级

| 层级 | 默认上限 | 理由 |
|------|---------|------|
| 桶配额 (bucket_id) | ≤5张 | 同一 dhash 桶组内限制 |
| 簇配额 (scene_cluster) | ≤25张 | 同一场景簇不超过 25% |
| **全局上限** | **100张** | 最终 gallery 大小 |
| 最低簇数 | ≥3个（若候选池有≥3簇） | 确保多样性 |

### 5.2 圆桌轮转算法

这是**取代纯分数排序**的核心机制。

```python
def round_robin_selection(images, cluster_ids, bucket_ids, 
                           bucket_quota=5, cluster_quota=25,
                           global_quota=100, decay_factor=0.02):
    """
    images: list of dicts (含 adjusted_composite, filepath, ...)
    cluster_ids: list of int, 每张的最终簇 ID
    bucket_ids: list of int, 每张的桶组 ID
    """
    # 按簇分组
    clusters = {}
    for i, img in enumerate(images):
        cid = cluster_ids[i]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append((i, img))
    
    # 簇内按 adjusted_composite 降序
    for cid in clusters:
        clusters[cid].sort(key=lambda x: -x[1]["adjusted_composite"])
    
    # 指针和计数
    ptr = {cid: 0 for cid in clusters}
    selected_bucket = {}
    selected_cluster = {}
    final_list = []
    not_selected_reasons = {}
    
    # 有效簇列表
    active_clusters = set(clusters.keys())
    
    for round_idx in range(500):  # 安全上限
        if len(final_list) >= global_quota:
            break
        if not active_clusters:
            break
        
        candidates = []
        for cid in active_clusters:
            if ptr[cid] >= len(clusters[cid]):
                continue
            idx, img = clusters[cid][ptr[cid]]
            
            # 计算 priority: adjusted_composite - decay × 已选数
            priority = img["adjusted_composite"] - decay_factor * selected_cluster.get(cid, 0)
            
            candidates.append((priority, cid, idx, img))
        
        if not candidates:
            break
        
        # 选 priority 最高的
        candidates.sort(key=lambda x: -x[0])
        _, best_cid, best_idx, best_img = candidates[0]
        
        bid = bucket_ids[best_idx]
        
        # 检查配额
        bucket_count = selected_bucket.get(bid, 0)
        cluster_count = selected_cluster.get(best_cid, 0)
        
        if bucket_count >= bucket_quota:
            # 桶配额满 → 跳过，记录原因
            not_selected_reasons[best_img["filepath"]] = "QUOTA_BUCKET"
            ptr[best_cid] += 1
            continue
        
        if cluster_count >= cluster_quota:
            # 簇配额满 → 跳过，记录原因
            not_selected_reasons[best_img["filepath"]] = "QUOTA_CLUSTER"
            ptr[best_cid] += 1
            continue
        
        # 入选
        final_list.append(best_img)
        selected_bucket[bid] = bucket_count + 1
        selected_cluster[best_cid] = cluster_count + 1
        ptr[best_cid] += 1
    
    # 记录未入选原因（余下未遍历的）
    for cid in active_clusters:
        while ptr[cid] < len(clusters[cid]):
            idx, img = clusters[cid][ptr[cid]]
            if img["filepath"] not in not_selected_reasons:
                not_selected_reasons[img["filepath"]] = "QUOTA_GLOBAL"
            ptr[cid] += 1
    
    return final_list, not_selected_reasons
```

### 5.3 若最终不足 100 张

```python
if len(final_list) < global_quota:
    print(f"WARNING: 数据集仅入选 {len(final_list)} 张，未达 {global_quota}")
    # 如实报告，不填充
```

### 5.4 truro_school 与 real_estate 差异化配额

| 参数 | 默认 | real_estate | truro_school |
|------|------|-------------|--------------|
| dedup_threshold | 0.14 | 0.10 | 0.22 |
| dhash_merge_hamming | 4 | 3 | 6 |
| bucket_quota | 5 | 3 | 8 |
| cluster_quota | 25 | 25 | 15 |
| portrait_penalty_max | 0.15 | 0.15 | 0.10 |

---

## 6. C6 — 复核池构建

### 6.1 标准化落选原因编码

| Code | 含义 | 触发条件 |
|------|------|---------|
| DEDUP_DENSE | 近似去重 | combined_distance < threshold |
| QUOTA_BUCKET | 桶配额溢出 | 该桶已选满 bucket_quota |
| QUOTA_CLUSTER | 簇配额溢出 | 该簇已选满 cluster_quota |
| QUOTA_GLOBAL | 全局 100 满 | 已选 100 张后剩余 |
| NOISE_CLUSTER | DBSCAN 噪声点 | cluster_id=-1 且 composite<0.6 |
| BORDERLINE_A | Stage A/B 边界 | 非 REAL 且 quality<GOOD |
| KEPT_IN_GALLERY | ✅ 入选 | 入选 top-100 |

### 6.2 复核池入选条件

```python
def build_review_pool(all_images, selected_paths, reasons):
    pool = []
    for img in all_images:
        fp = img["filepath"]
        if fp in selected_paths:
            continue
        
        reason = reasons.get(fp, "UNKNOWN")
        
        include = (
            (reason == "DEDUP_DENSE" and img["composite_score"] >= 0.65) or
            (reason in ("QUOTA_BUCKET", "QUOTA_CLUSTER") and img["adjusted_composite"] >= 0.70) or
            (reason == "NOISE_CLUSTER" and img["composite_score"] >= 0.60) or
            (reason == "BORDERLINE_A" and img["quality_score"] >= 0.4) or
            (img["composite_score"] >= 0.80 and img["portrait_penalty"] >= 0.08)
        )
        
        if include:
            pool.append({
                "filepath": fp,
                "composite_score": img["composite_score"],
                "portrait_penalty": img["portrait_penalty"],
                "adjusted_composite": img["adjusted_composite"],
                "reason": reason,
                "review_priority": "高" if img["composite_score"] >= 0.75 else "中"
            })
    
    pool.sort(key=lambda x: -x["composite_score"])
    return pool[:150]
```

---

## 7. C7 — 输出文件结构

### 7.1 输出目录

```
outputs/{dataset_name}/
├── top_100_gallery.csv      # 最终入选 ≤100 张
├── review_pool.csv           # 复核池 ≤150 张
├── stageC_report.json        # 统计报告
├── dedup_log.json            # 去重详细记录
└── cluster_assignments.json  # 每张图的聚类标签
```

### 7.2 top_100_gallery.csv

```csv
rank,filepath,composite_score,adjusted_composite,realism_confidence,quality_score,display_fit_score,scene_cluster_id,bucket_id,portrait_penalty,selected_reason
1,C:\pics\real_estate\img042.jpg,0.891,0.881,0.95,0.85,0.82,3,2,0.01,BUCKET_HIGH_SCORE
2,C:\pics\truro_school\group_03.jpg,0.782,0.732,0.80,0.78,0.75,7,12,0.05,CLUSTER_ROUND_ROBIN
...
```

**selected_reason 编码**：

| Code | 含义 |
|------|------|
| BUCKET_HIGH_SCORE | 桶内 composite 排名靠前入选 |
| CLUSTER_ROUND_ROBIN | 圆桌轮转中因簇轮选入选 |
| PORTRAIT_LOW_PENALTY | 单人近景但惩罚低 |
| NOISE_HIGH_SCORE | 噪声点但 composite≥0.7 |

### 7.3 review_pool.csv

```csv
filepath,composite_score,portrait_penalty,adjusted_composite,combined_distance,kept_neighbor,bucket_id,cluster_id,not_selected_reason,review_priority
C:\pics\real_estate\img038.jpg,0.812,0.02,0.792,0.09,C:\pics\real_estate\img042.jpg,2,3,DEDUP_DENSE,高
C:\pics\truro_school\group_15.jpg,0.795,0.06,0.735,,,12,7,QUOTA_BUCKET,中
```

### 7.4 stageC_report.json

```json
{
  "dataset": "real_estate",
  "total_input_from_stage_b": 2100,
  "after_dedup": 890,
  "clusters_found": 8,
  "final_gallery_count": 100,
  "under_100": false,
  "cluster_distribution": {"0": 25, "1": 20, "2": 15, "3": 12, "4": 10, "5": 8, "6": 6, "7": 4},
  "portrait_in_gallery": 18,
  "dedup_threshold_used": 0.10,
  "dedup_removed_high_score": {"count": 45, "threshold": 0.65},
  "review_pool_size": 142,
  "quota_mode": "normal",
  "tight_mode_triggered": false,
  "warnings": []
}
```

---

## 8. 复核与替换机制

### 8.1 人工复核流程

```
Step 1: 查看 top_100_gallery.csv 预览（建议生成 HTML grid）
Step 2: 检查 review_pool.csv 中 "高" 优先级的图片
Step 3: 决定替换：
  - 从 review_pool 选图，记录替换目标 rank
  - 生成 replacement_log.csv 记录替换前后
Step 4: 重新运行排序（可选）：python run_stage_c.py --dataset X --replace replacement_log.csv
```

### 8.2 抽样校验

| 校验项 | 方法 | 样本量 |
|--------|------|--------|
| 去重正确性 | 随机选 10 组 (kept, removed) 对比 | 10 组/数据集 |
| 聚类合理性 | 每簇随机选 3 张拼图 | 3×簇数 |
| 单人近景检测 | 所有 portrait_penalty>0.08 的图 | 全量 |
| 整体多样性 | 按簇分布检查 | 自动报告 |

---

## 9. 运行步骤与依赖

### 9.1 依赖

```bash
pip install opencv-python numpy scikit-learn scikit-image pyyaml
```

> 注：无需额外深度学习框架 (PyTorch/TensorFlow)，Stage C 完全零模型推理

### 9.2 运行步骤

```bash
# Step 1: 配置
# 编辑 pipeline_config.yaml（参考下方配置模板）

# Step 2: 运行单数据集（推荐先验证小规模）
python run_stage_c.py --dataset real_estate --stage_b_csv inputs/real_estate_stage_b.csv

# Step 3: 检查输出
#   outputs/real_estate/top_100_gallery.csv
#   outputs/real_estate/stageC_report.json

# Step 4: 运行全量
python run_pipeline.py --datasets all --resume
```

### 9.3 配置模板 (pipeline_config.yaml)

```yaml
stage_c:
  seed: 42
  composite_weights:
    w_realism: 0.50
    w_quality: 0.30
    w_display: 0.20
  
  dedup:
    default_threshold: 0.14
    threshold_range: [0.08, 0.30]
    dhash_merge_hamming: 4
  
  clustering:
    dbscan_eps_bucket_inner: 0.15
    dbscan_eps_cross_bucket: 0.20
    dbscan_min_samples: 2
  
  quotas:
    bucket_quota: 5
    cluster_quota: 25
    global_quota: 100
    min_clusters: 3
  
  round_robin:
    decay_factor: 0.02
    max_iterations: 500
  
  review_pool:
    max_size: 150
    high_score_threshold: 0.65
    quota_excluded_threshold: 0.70
    noise_threshold: 0.60
  
  per_dataset_overrides:
    real_estate:
      dedup_threshold: 0.10
      dhash_merge_hamming: 3
      bucket_quota: 3
      cluster_quota: 25
    
    truro_school:
      dedup_threshold: 0.22
      dhash_merge_hamming: 6
      bucket_quota: 8
      cluster_quota: 15
      portrait_penalty_max: 0.10
    
    digital_domain:
      dedup_threshold: 0.10
    
    design:
      dedup_threshold: 0.08
    
    ecommerce:
      dedup_threshold: 0.12
```

---

## 10. 风险、阈值校准与小规模验证

### 10.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 特征向量区分度不足导致欠聚类 | 中 | 配额失效 | 日志输出 eps 值和簇分布；允许手动重跑 C3 |
| dhash 桶合并过激导致跨场景合并 | 低 | 本应不同的图归同桶 | 默认 hamming=4，truro_school=6 已区分 |
| portrait_penalty 误伤群像/建筑 | 低 | 分数轻微下降 | 惩罚值小（≤0.03），影响有限 |
| 某数据集去重后候选池 < 100 | 中 | 无法满额 | 如实报告数量，不补低质图 |
| 跨数据集特征分布差异大 | 高 | 聚类质量波动 | 自适应 eps；per-dataset 覆盖 |

### 10.2 小规模验证建议

```bash
# 建议：先在一个小型数据集上验证完整流程
# 1. 选 ecommerce（~600张）作为验证集
python run_stage_c.py --dataset ecommerce --stage_b_csv inputs/ecommerce_stage_b.csv

# 2. 检查输出指标
#   - final_gallery_count（期望 ≤100）
#   - cluster_distribution（期望 ≥3 个簇）
#   - portrait_in_gallery（期望 < 50%）

# 3. 目视抽检 20 张 top-100 图片
#   - 有无近似重复？
#   - 场景多样性是否合理？

# 4. 调参后跑 real_estate（房产，~5000张，挑战最大）
python run_stage_c.py --dataset real_estate --stage_b_csv inputs/real_estate_stage_b.csv

# 5. 最终全量
python run_pipeline.py --datasets all
```

### 10.3 阈值校准的 fallback 机制

```python
# 若 final_gallery_count < 60（严重不足）：
# 自动触发宽松模式：
fallback_config = {
    "dedup_threshold": min(original_threshold * 1.5, 0.30),
    "bucket_quota": min(original_bucket_quota * 2, 15),
    "cluster_quota": min(original_cluster_quota * 1.5, 40),
    "portrait_penalty_max": 0.05,
}
# 记录日志：f"数据集 {dataset} 触发 fallback，原阈值 {t1}，新阈值 {t2}"
```

---

## 11. 可复现性保证

```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# sklearn 聚类固定 seed
from sklearn.cluster import DBSCAN
db = DBSCAN(metric='cosine', eps=0.15, min_samples=2, n_jobs=1)
```

所有随机步骤使用固定 seed 42，确保每次运行结果一致。

---

## 附录：脚本模块结构建议

```
pipeline/
├── stage_c/
│   ├── __init__.py
│   ├── compute_composite.py      # C1
│   ├── dedup.py                  # C2 (含 dhash + edge_histogram)
│   ├── cluster.py                # C3 (dhash桶 → DBSCAN)
│   ├── portrait_penalty.py       # C4
│   ├── quota_round_robin.py      # C5
│   ├── review_pool.py            # C6
│   ├── report.py                 # C7
│   └── run_stage_c.py            # 入口（单数据集）
├── config/
│   └── pipeline_config.yaml
├── run_pipeline.py               # 全数据集循环
└── scripts/
    └── generate_gallery_preview.py  # HTML grid 生成（可选）
```

---

> **本方案三版本演进**：
> 1. `dedup_diversity_ranking_design.md` — 原始设计（贪心去重 + DBSCAN + 圆桌轮转）
> 2. `executor_dedup_diversity_ranking_plan.md` — 可执行规划（含 Haar Cascade + 自适应 eps）
> 3. **本文件 (executor_stage2_final_plan.md)** — 修正版（三信号去重 + dhash桶聚类 + 弱惩罚 + 标准化编码）
>
> Executor 应**直接使用本最终版**编码实现。
