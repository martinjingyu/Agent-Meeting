# Stage C 修正版：去重/多样性/排序（不依赖重模型、不依赖脆弱人脸检测）

> 对应怀疑论评审的 5 项挑战。本修订不改变 Stage A/B 的已有设计，仅修正 Stage C 中评审指出的脆弱点。  
> 基调：**宁可多留复核池、不依赖不稳定信号、不足100如实报**。

---

## 修订摘要

| 原方案问题 | 修订方案 | 动机 |
|-----------|---------|------|
| C4 单人近景：Haar Cascade 阈值硬、不稳定 | 改为 **多信号弱惩罚**：面占比 + 主体占比 + 长宽比 + 边缘密度联合评分 → portrait_penalty ∈ [0,0.15] 加入复合分 | 不依赖单一面部检测输出 |
| C2 去重：仅依赖 feature_vector | 改为 **三信号组合**：感知哈希(dhash) + 特征向量 cosine + 布局边缘直方图 → 综合距离 | 单一特征不可靠 |
| C3 聚类：仅用 feature_vector | 改为 **多尺度聚类**：先 dhash 分桶（粗分组），桶内 DBSCAN 调优 | 无深特征时更稳健 |
| 单人近景：硬配额硬阻断 | 改为 **端口罚分** → portrait_penalty 加权入 composite → 天然降序挤出 | 软控制、不硬阻断 |
| 配额 by 场景簇：依赖聚类精度 | 改为 **配额分两级**：桶级配额 + 聚类簇配额互补；truro_school 和房产分别配置 | 聚类失效时有兜底 |

---

## 1. 近重复检测：三信号组合距离

### 信号构成

```python
combined_distance = (
    w_dhash * dhash_distance / 64.0 +          # 归一化到 [0,1]
    w_feat  * cosine_distance(feat_a, feat_b) + 
    w_edge  * edge_histogram_distance(img_a, img_b)
)
# 默认权重: w_dhash=0.35, w_feat=0.40, w_edge=0.25
```

### 各信号说明

| 信号 | 方法 | 计算量 | 区分力 |
|------|------|--------|--------|
| **dhash** | 缩放到 9×8 → 差异位 → 64bit → Hamming / 64 | 0.5ms/张 | 能抵抗轻微裁切/压缩，对大体相同但颜色不同敏感 |
| **cosine(feature)** | 576-dim 特征向量 cosine | 0.02ms/对 | 语义近似的图也会拉近（同一场景不同角度） |
| **边缘直方图** | Canny + 8×8 网格边缘像素密度 → L1 | 0.3ms/张 | 区分构图相近但色彩不同的图 |

### 贪心去重算法不变，但距离改为 combined_distance

```python
# 阈值自适应：取候选池 pairwise combined_distance 的 10 百分位 × 1.2
base = percentile(all_combined_distances, 10)
dedup_threshold = min(max(base * 1.2, 0.08), 0.30)

# 去重时记录三个子信号值以便复核
dedup_detail = {
    "combined_distance": 0.14,
    "dhash_dist": 12,           # raw Hamming, /64 = 0.19
    "feat_dist": 0.09,
    "edge_hist_dist": 0.11,
    "kept_neighbor": "..."
}
```

**为什么三个信号互补**：纯 dhash 对裁切敏感；纯特征对光照/场景变化敏感；纯边缘直方图对平面图案敏感。三者组合在 CPU-only 场景下接近中等深度特征的区分力。

---

## 2. 场景聚类：先桶后簇（双阶段，不依赖深特征）

### 第一阶段：dhash 粗分桶（快速、零模型）

```
对每张图片:
  1. 计算 dhash (9×8 → 64 bits)
  2. 以 64-bit 值为桶 ID（精确匹配）
     → 完全相同或极近似的图落同一桶
  3. 合并相邻桶：Hamming 距离 ≤ 4 的桶视为同组
     （例如桶 A: 0b101010... 与桶 B: 0b101110... 差 1 bit → 合并）
```

**作用**：快速聚合肉眼近似的图片，不依赖任何模型。桶大小反映重复密集度。

### 第二阶段：桶组内 DBSCAN（调优，用特征向量）

```
对每个桶组（含 M 张图）:
  1. 若 M ≤ 5 → 不聚类，整个桶组视为一个细簇
  2. 若 M > 5 → DBSCAN(feature_vectors, metric=cosine, eps=0.15, min_samples=2)
     → 进一步分离同一桶内的不同主体/角度
```

**合并跨桶的相似图片**（解决裁切/滤镜导致的不同桶）：

```python
# 对所有桶组质心（取组内 median feature vector）做二次 DBSCAN
# eps=0.20, min_samples=1 → 合并不需要满足密度要求
桶组_质心 = [median(features of group g) for g in bucketed_groups]
跨桶聚类 = DBSCAN(cosine, eps=0.20, min_samples=1).fit(桶组_质心)
```

### 最终聚类 ID

```python
final_cluster_id = f"{cross_bucket_cluster_id}.{in_bucket_subcluster_id}"
# 例如 "3.1" = 跨桶簇 3 下的子簇 1
# 噪声点: "-1.0"
```

**优势**：dhash 桶保证肉眼相同的图一定在同一组（即使特征向量相近但不够近），而跨桶 DBSCAN 解决滤镜/轻微变换问题。

---

## 3. 单人近景：弱惩罚评分（不依赖稳定人脸检测）

### 信号合成 portrait_penalty ∈ [0, 0.15]

不再用 Haar Cascade 硬判定"是/否单人近景"，而是计算一个连续惩罚值：

```python
def portrait_penalty(img) -> float:
    # ---- 信号1: 主体占比（基于边缘密度在中心区域的集中度） ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    
    # 中心区域 (40%-60% 的宽高范围)
    cx, cy = w//2, h//2
    crop = edges[cy-int(h*0.2):cy+int(h*0.2), cx-int(w*0.2):cx+int(w*0.2)]
    center_density = crop.sum() / (crop.size + 1) if crop.size > 0 else 0
    full_density = edges.sum() / (h * w)
    
    # 若边缘集中在中心区域（主体居中的典型特征）
    center_concentration = center_density / (full_density + 1e-6)
    s1 = min(max((center_concentration - 2.0) / 4.0, 0), 1.0)  # 2x 集中 → 开始贡献
    
    # ---- 信号2: 宽高比 ----
    aspect = w / h
    # 人像比例附近 (0.67-1.0) 加分
    aspect_penalty = 0
    if 0.6 <= aspect <= 1.1:
        aspect_penalty = 1.0 - abs(aspect - 0.75) / 0.35
    
    # ---- 信号3: 面部检测作为弱提示（可选，不依赖） ----
    # 如果 OpenCV Haar Cascade 可用且检测到单面且面占比 > 8% → 加分
    face_signal = 0.0
    if face_detector_available:
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        if len(faces) == 1:
            x, y, fw, fh = faces[0]
            face_area_ratio = (fw * fh) / (w * h)
            if face_area_ratio > 0.05:
                face_signal = min(face_area_ratio / 0.2, 1.0)  # 越大越像近景
    
    # ---- 综合 ----
    penalty = (
        0.40 * s1 +                    # 边缘集中度 (weight 0.40)
        0.30 * aspect_penalty +         # 宽高比 (weight 0.30)
        0.30 * face_signal              # 面部信号 (weight 0.30, 无检测器时 = 0)
    )
    # 映射到 [0, 0.15]
    return min(penalty * 0.15, 0.15)
```

### 如何使用

```python
adjusted_composite = composite_score - portrait_penalty
```

- 单人近景特征明显的图：penalty ≈ 0.10–0.15 → 排名下降
- 群像/风景/建筑：penalty ≈ 0.00–0.03 → 不受影响
- **不硬阻断**：若某单人近景实景+质量分足够高（eg. composite=0.95, penalty=0.12 → adjusted=0.83），仍然可能入选，但降低了与同分其他图的竞争力

### 为什么这样更健壮

1. 不依赖面部检测是否找到脸 — 信号1+2 在正面、侧面、背影单人近景都能检测（边缘集中 + 比例）
2. 即使面部检测完全不可用（权重 0→0.3），仍有 0.7 权重来自边缘集中度和宽高比
3. 连续惩罚值取代硬分类，避免了"差一点就判单人/不是单人"的阈值敏感
4. 在 round-robin 排序时，adjusted_composite 参与簇内排序，自然使单人近景排名后移

---

## 4. truro_school 与房产类的差异化配额

### truro_school（学校活动照）

| 参数 | 值 | 理由 |
|------|----|------|
| dedup_threshold | 0.22（宽松） | 大量拍摄角度相似的合影，太严会筛掉太多 |
| dhash_bucket_merge | Hamming ≤ 6 | 更激进合并桶（同类活动照归簇） |
| 桶配额 (bucket_quota) | 每个 dhash 桶组 ≤ 8 张 | 同一活动/场景不超过 8 张 |
| 最终簇配额 (cluster_quota) | ≤ 15 张 | 同一场景类型不过载 |
| portrait_penalty 上限 | 0.10（比默认 0.15 更轻） | 学校活动以人为主，太严厉会导致不足 100 |
| min_final_count | 90（宽松目标） | 若不足 90 才触发降级重跑 |
| 降级方案 | dedup=0.26, 桶配额=12, 不加 portrait_penalty | 最终如实报告数量 |

### real_estate（房产室内外观）

| 参数 | 值 | 理由 |
|------|----|------|
| dedup_threshold | 0.10（严格） | 相同房间不同拍摄位置需去重 |
| dhash_bucket_merge | Hamming ≤ 3（严格） | 同一角度和装修的图不应认为是不同场景 |
| 桶配额 (bucket_quota) | 每个 dhash 桶组 ≤ 3 张 | 严格限制同一机位的重复 |
| 最终簇配额 (cluster_quota) | ≤ 25 张 | 房产场景类型有限（客厅/卧室/厨房/外观） |
| portrait_penalty 上限 | 0.15（默认） | 房产不需要人像，正常惩罚 |
| 跨桶聚类 eps | 0.18（稍紧） | 更好分离不同房间类型 |

### 其他数据集使用默认值

```python
default_config = {
    "dedup_threshold": 0.14,
    "dhash_merge_hamming": 4,
    "bucket_quota": 5,
    "cluster_quota": 25,
    "portrait_penalty_max": 0.15,
    "min_final_count": 100,
}
```

---

## 5. 落选原因编码（标准化的 not_selected_reason）

### 编码表（每个原因有 code + 短描述 + 长描述）

| Code | 原因 | 适用场景 | 说明 |
|------|------|---------|------|
| DEDUP_DENSE | 近似去重（距离 < 阈值） | 任一被贪心去重算法丢弃的图 | 关联 kept_neighbor + combined_distance |
| QUOTA_BUCKET | 桶配额溢出（该 dhash 桶已满） | 同一 dhash 桶组超过 bucket_quota 上限 | 第 bucket_quota+1 张开始标记 |
| QUOTA_CLUSTER | 簇配额溢出（该场景簇已满） | 某最终簇已选满 cluster_quota | 即使该图分高也因簇饱和落选 |
| QUOTA_PORTRAIT | 单人近景惩罚（portrait_penalty > 0.08 导致 adjusted 偏低） | 圆桌轮转中因 adjusted_composite 低而未入选 | 注意：不是硬配额，是惩罚后排名自然下降 |
| QUOTA_GLOBAL | 数据集已达 100 张上限 | 所有配额用满后剩余的高分图 | 仅用于无其他原因时 |
| NOISE_CLUSTER | DBSCAN 噪声点（低置信度无簇归属） | -1 标签且 composite < 0.6 的图 | composite ≥ 0.6 的噪声单独进复核池 |
| BORDERLINE_A | 实景可信度 AMBIGUOUS 且质量 FAIR 以下 | Stage A/B 边界样本 | 直接进复核池 |
| KEPT_IN_GALLERY | **入选 top-100** | 所有入选图 | selected_reason 字段用 |

### 输出格式（review_pool.csv）

```csv
filepath,composite_score,portrait_penalty,adjusted_composite,combined_distance,kept_neighbor,bucket_id,cluster_id,not_selected_reason,review_priority
C:\pics\real_estate\img042.jpg,0.812,0.02,0.792,0.09,C:\pics\real_estate\img038.jpg,b3,c3.1,DEDUP_DENSE,高
C:\pics\truro_school\group_15.jpg,0.795,0.06,0.735,,-,b12,c7.2,QUOTA_BUCKET,中
```

### top_100_gallery.csv 入选原因

```csv
rank,filepath,adjusted_composite,cluster_id,selected_reason,portrait_penalty
1,C:\pics\real_estate\img101.jpg,0.891,c1.2,桶配额高分入选,0.01
2,C:\pics\truro_school\group_03.jpg,0.782,c7.1,簇轮转入选,0.05
...
```

**selected_reason 编码**：

| Reason Code | 含义 |
|------------|------|
| BUCKET_HIGH_SCORE | 同桶内 composite 排名靠前入选 |
| CLUSTER_ROUND_ROBIN | 圆桌轮转中因簇轮选入选 |
| PORTRAIT_LOW_PENALTY | 单人近景但惩罚低，排名仍靠前 |
| NOISE_HIGH_SCORE | 噪声点但 composite ≥ 0.7，优先考虑 |
| REVIEW_REPLACEMENT | 人工复核替换加入 |

---

## 6. 补充：圆桌轮转的健全版本（整合 portrait_penalty）

```python
# 每轮计算 priority 时使用 adjusted_composite 而非 raw composite
for round in range(max_iterations):
    # 计算每个活跃簇的 priority
    candidates = []
    for cluster_id in active_clusters:
        if cluster_ptr[cluster_id] >= len(cluster_images[cluster_id]):
            continue
        img = cluster_images[cluster_id][cluster_ptr[cluster_id]]
        priority = img.adjusted_composite - decay_factor * cluster_selected[cluster_id]
        candidates.append((priority, cluster_id, img))
    
    if not candidates:
        break
    
    # 选最高 priority 的
    candidates.sort(key=lambda x: -x[0])
    best_priority, best_cluster, best_img = candidates[0]
    
    # 检查配额（桶配额 + 簇配额 + 全局 100）
    if (bucket_selected[best_img.bucket_id] < bucket_quota 
        and cluster_selected[best_cluster] < cluster_quota
        and total_selected < 100):
        
        final_list.append(best_img)
        bucket_selected[best_img.bucket_id] += 1
        cluster_selected[best_cluster] += 1
        total_selected += 1
        cluster_ptr[best_cluster] += 1
    else:
        # 配额满 → 该簇 ptr 不动，记录原因
        if not_bucket_reason[best_img.filepath]:
            not_bucket_reason[best_img.filepath] = "QUOTA_BUCKET"
        cluster_ptr[best_cluster] += 1  # 跳过这张图继续
```

**关键改进**：桶配额和簇配额同时检查，且用 adjusted_composite 取代 raw composite。

---

## 7. 复核池构建逻辑

```python
def build_review_pool(all_images, selected_paths, config):
    pool = []
    for img in all_images:
        if img.filepath in selected_paths:
            continue  # 已入选
        
        reason = determine_reason(img, selected_paths, config)
        
        # 仅以下情况进复核池：
        include = (
            # 高分被去重
            (reason == "DEDUP_DENSE" and img.composite_score >= 0.65)
            # 配额溢出且分高
            or (reason in ("QUOTA_BUCKET", "QUOTA_CLUSTER") and img.adjusted_composite >= 0.70)
            # DBSCAN 噪声点且质量好
            or (reason == "NOISE_CLUSTER" and img.composite_score >= 0.60)
            # 边界样本
            or (reason == "BORDERLINE_A" and img.quality_score >= 0.4)
            # 单人近景被惩罚压分但 raw composite 高
            or (reason == "QUOTA_PORTRAIT" and img.composite_score >= 0.80)
        )
        
        if include:
            pool.append({
                "filepath": img.filepath,
                "composite_score": img.composite_score,
                "portrait_penalty": img.portrait_penalty,
                "adjusted_composite": img.adjusted_composite,
                "reason": reason,
                "review_priority": "高" if img.composite_score >= 0.75 else "中"
            })
    
    # 按 composite_score 降序排列，限制 ≤ 150 张
    pool.sort(key=lambda x: -x["composite_score"])
    return pool[:150]
```

---

## 8. 与原始方案的主要差异对照

| 方面 | 原始方案 | 修正版 | 评审对应点 |
|------|---------|--------|-----------|
| 去重距离 | 仅 feature_vector cosine | dhash(0.35) + cosine(0.40) + edge_hist(0.25) | ① |
| 聚类方式 | DBSCAN on feature_vector 直接 | dhash 分桶 → 桶内 DBSCAN → 跨桶合并 | ② |
| 单人近景 | Haar Cascade 硬判定 + 硬配额 | 多信号连续惩罚分 [0,0.15] + 软调整 | ③ |
| truro_school 策略 | dedup=0.18, 场景簇≤15 | dedup=0.22, 桶配额≤8, 簇配额≤15, penalty上限0.10 | ④ |
| 房产策略 | dedup=0.10 | dedup=0.10, 桶配额≤3, 跨桶eps=0.18 | ④ |
| 落选原因 | 5种自由文本 | 9种标准化编码 (DEDUP_DENSE, QUOTA_BUCKET, 等) | ⑤ |

---

## 9. 性能估算（CPU-only, Intel UHD 770）

| 步骤 | 单张耗时 | 总耗时（25k 候选） | 并行？ |
|------|---------|-------------------|--------|
| dhash 计算 | 0.5 ms | ~12 s | 可并行 4 workers |
| 边缘直方图 | 0.3 ms | ~8 s | 同上 |
| 特征向量 cosine（已有） | 0 新开销（Stage B 产出） | 0 | — |
| 三信号去重 | 0.03 ms/对 | ~3 min (N=25k, O(N²) 被贪心剪枝) | 否 |
| dhash 分桶 | 0.01 ms | <1 s | 否 |
| 桶内 DBSCAN | — | ~2 min | 可 per-bucket 并行 |
| portrait_penalty | 2.0 ms（含边缘+可选 Haar） | ~50 s | 可并行 |
| 圆桌轮转 | — | <1 s | 否 |
| **Stage C 总计** | | **~8 min** | |

> 注：无模型推理（Stage C 完全零模型），远优于原方案的 10-15 min 估算，且在 CPU 上稳定。

---

## 10. 对 Executor 的要求

1. **安装**：opencv-python, numpy, scikit-learn, scikit-image（已涵盖，无新增依赖）
2. **输入确认**：Stage B 必须输出 feature_vector (float[576]) 和 composite_score
3. **per-dataset 配置**：在 `pipeline_config.yaml` 中按上面的参数表配置 truro_school 和 real_estate
4. **人脸检测可选**：若 executor 选择不装 OpenCV Haar（或其他原因），portrait_penalty 自动降为 0.7权重（仅用边缘集中度+宽高比），功能仍完整
5. **日志要求**：每一步输出图数量 + 阈值 + 簇分布，便于人工核对

---

> **结论**：本修正版 Stage C 完全移除对深度模型和人脸检测的硬依赖，改用三信号组合去重 + dhash 分桶聚类 + 连续 portrait_penalty 弱惩罚。在 CPU-only 环境下每数据集约 8 分钟，25k 候选共约 1.5 小时（11 数据集串行），且所有阈值可 per-dataset 配置。不足 100 张时会如实报告，不补入低质量图片。
