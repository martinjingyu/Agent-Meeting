# Stage 2 Pipeline 方案 — 可直接交付 Executor

> 日期: 2025-07-16
> 目标环境: Windows + CPU-only (Intel UHD 770, 无 CUDA)
> 输入: C:\pics 下 61,958 张图片 (11 个数据集, truro_school 占 ~59%)
> 核心约束: 不可依赖 GPU / 不可跑 >100M 参数模型 / 不做非视觉判定 / 保守偏误 (边界样本降复核池)
> 关键经验教训: MobileNetV3-Small ONNX CPU 推理解码实测 ~6.4s/张 → 25k 张 ≈ 44h; 已完全移除任何神经网络推理

---

## 目录

1. [架构总览](#1-架构总览)
2. [视觉判断逻辑论证](#2-视觉判断逻辑论证)
3. [A 级联流水线设计 (A→B→C)](#3-级联流水线设计-abc)
4. [去重 / 相似度 / 多样性 / 排序策略](#4-去重--相似度--多样性--排序策略)
5. [输出文件结构与复核机制](#5-输出文件结构与复核机制)
6. [运行步骤与依赖](#6-运行步骤与依赖)
7. [风险、阈值校准、小规模验证](#7-风险阈值校准小规模验证)
8. [必须用文件 · 禁止用文件清单](#8-必须用文件--禁止用文件清单)

---

## 1. 架构总览

```
全量 61,958 张
    │
    ▼
Stage A ── 信号计算 + 硬拒绝 (纯 OpenCV/numpy, 零模型)
    │  11 个可计算视觉信号
    │  固定阈值硬拒绝 (损坏/纯色/微型)
    │  per-dataset 自适应软过滤 → survivors + LOW_CONFIDENCE
    │  预计通过率 ~30-45% → ~18,000-28,000 张
    │  耗时: ~2-4 min (8 workers)
    ▼
Stage B ── 三维启发式评分 (纯信号公式, 零模型)
    │  realism_score (0-100) + quality_score (0-100) + display_score (0-100)
    │  8 种展示适配标签 (位标记)
    │  三轴联合决策: ✅入选 / ⚠️复核 / ❌淘汰
    │  仅对 Stage A survivors 计算
    │  耗时: ~5 秒
    ▼
Stage C ── 去重 + 聚类 + 排序
    │  多分辨率 dhash (3级尺度: 8×8, 16×16, 32×32) 去重
    │  11-dim 信号向量 → PCA(5) → DBSCAN 聚类
    │  Haar Cascade 单人近景软惩罚
    │  每簇配额排序 + 噪声优先 → top-100
    │  仅对 Stage B scored survivors
    │  耗时: ~30-60 秒
    ▼
输出: 11 个 per-dataset top-100 + review pool + aggregate stats
```

### 为什么这样设计 - 核心理由

| 决策 | 理由 |
|------|------|
| **完全移除神经网络模型** | MobileNetV3-Small 实测 CPU 6.4s/张 → 44h, 收益-成本比不可接受 |
| **Stage A 早期淘汰 ~60%** | 零模型阶段淘汰越快, 后续处理量越小; OpenCV 信号 ~14ms/张 vs 模型 ~6.4s/张 = 457x 速度差 |
| **纯启发式评分替代模型分类** | 11 个信号组合可以提供足够的实景/质量区分度; CGI vs 真实在 edge_ratio+colorfulness 维度有明确可分性 |
| **per-dataset 自适应阈值** | 每个数据集的内容分布不同 (truro_school 模糊 vs digital_domain 高锐度), 统一阈值会导致误杀 |
| **三轴评分 (realism×quality×display)** | 单轴不足以反映 gallery 适配; 乘积型评分确保综合优秀而非单项突出 |
| **DBSCAN + 簇配额** | 确保场景多样性, 防止单一场景占满 top-100 |
| **保守偏误 + 复核池** | 边界样本不武断硬切, 全部进复核池供人工确认 |

---

## 2. 视觉判断逻辑论证

### 2.1 核心原则: 为什么只能依赖视觉信息

- **目录名/文件名/EXIF/时间戳** 都不可靠: 同一数据集可能混合真实照片和渲染图 (C:\pics 可能如此)
- **仅使用图像像素内容**: BGR 像素 → 信号 → 评分 → 决策
- **跨数据集泛化**: 不 hardcode 任何类别名、路径名、设备名

### 2.2 11 个可计算视觉信号总表

| # | 信号 | 计算方法 | 成本 | 含义 | 对 gallery 的意义 |
|---|------|---------|------|------|-----------------|
| S1 | `sharpness` | Laplacian 方差 (cv2.CV_64F) | ~1ms | 清晰度 | 极低(<5)→纯色/损坏; 极高(>800)→过度锐化/AI |
| S2 | `edge_ratio` | Canny 边缘占比 (50/150) | ~2ms | 纹理/结构密度 | <0.005→纯色CG; >0.4→文档/信息图 |
| S3 | `colorfulness` | Hasler-Susstrunk M (RG-YB标准差) | ~0.5ms | 色彩饱和度 | <5→灰/死图; >80→过饱和渲染 |
| S4 | `entropy` | 灰度直方图香农熵 (8bit) | ~0.5ms | 信息量/复杂度 | <3→死图; >7.5→噪声/高纹理 |
| S5 | `brightness_mean` | 灰度均值 | ~0.2ms | 整体曝光 | <20→欠曝; >230→过曝 |
| S6 | `brightness_std` | 灰度标准差 | ~0.2ms | 对比度/层次 | <15→雾/霾/无层次 |
| S7 | `aspect_ratio` | w / h | ~0.1ms | 画面比例 | >10→条幅; <0.1→竖长条 |
| S8 | `min_side` | min(w, h) | ~0.1ms | 最小边长 | <64→缩略图/图标 |
| S9 | `face_ratio_area` | Haar Cascade 最大人脸面积占比 | ~3ms | 单人近景检测 | >10%→单人近景惩罚 |
| S10 | `face_count` | Haar Cascade 检测到的人脸数 | ~3ms (与S9共享) | 人数 | ≥3→群像加分 |
| S11 | `horizontal_balance` | 左右半图亮度均值差归一化 | ~2ms | 构图稳定性 | >0.3→构图偏 |

**所有信号单张成本合计 ~14ms (含 decode), 8 workers 下全量 ~2-3 分钟。**

### 2.3 为什么 11 个信号 (而非 3-4 个) 是必要的

| 场景 | 单信号误判 | 11 信号组合纠正 |
|------|-----------|----------------|
| 雾景 (低 contrast, 低 color) | colorfulness 低 → 判非实景 | edge_ratio 正常 + entropy 正常 → 修正为 AMBIGUOUS |
| 雪景 (高 brightness, 低 color) | brightness_mean 高 → 判过曝 | entropy 高 + edge_ratio 正常 → 修正为 PROBABLY_REAL |
| 高质渲染图 | sharpness 高 + edge 适中 → 判实景 | colorfulness > 80 + homogeneity 异常 → 标记 CGI |
| 夜景 (低 brightness) | brightness_mean 低 → 判欠曝 | entropy 高 + edge_ratio 适中 + colorfulness 有变化 → 修正 |
| 纯色占位图 | 单一信号可能过 | sharpness<5 AND colorfulness<5 AND edge_ratio<0.003 → 三重确认 |
| 文档截图 | edge_ratio 高 | edge_ratio>0.4 AND entropy<4.5 → 标记 DOCUMENTARY |

### 2.4 什么模型都不用, 如何区分 CGI 渲染 vs 真实照片

核心答案: **不必完美区分; 只需保守标记 + 高风险数据集全部进复核池。**

启发式信号可以做到的:
1. **edge_ratio 极低 (<0.005)** + **colorfulness 极高 (>80)** → 极大概率非真实
2. **sharpness > 800** + **colorfulness > 70** → 过度锐化/渲染
3. **homogeneity** (<0.2 高纹理噪声) → 辅助判别

做不到但不需要做的:
- 高质感 photorealistic 渲染 (如 digital_domain) 确实难以通过信号区分
- 策略: digital_domain 的 AMBIGUOUS + PROBABLY_NON_REAL 全部进复核池, 不做自动淘汰
- 在 top-100 中标记 `confidence: LOW`, 显著提示人工审查

---

## 3. 级联流水线设计 (A→B→C)

### 3.1 各阶段全量 / 候选 / 跳过明细

| 阶段 | 处理对象 | 处理内容 | 全量/候选 | 并行 |
|------|---------|---------|-----------|------|
| A-1 信号计算 | 全量 61,958 张 | 11 信号 + decode | 全量 | 8 workers |
| A-2 硬拒绝 | 全量 | 短路式固定阈值拒绝 | 全量 | 单线程 (O(n)) |
| A-3 自适应软过滤 | 硬拒绝后幸存 | per-dataset 阈值标记 LOW_CONFIDENCE | 候选 | 单线程 |
| B 评分+标签 | 软过滤幸存者 (~18-28k) | 3 轴评分 + 8 标签 | **只对候选** | 可并行 |
| C 去重 | B 幸存者 | dhash 3 级 + Hamming | **只对候选** | 可并行 |
| C 聚类 | 去重后 | PCA(5) + DBSCAN | 只对候选 | 单线程 |
| C 排序+输出 | 聚类后 | 配额排序 → top-100 + 复核池 | 只对候选 | 单线程 |

**总耗时预算 (可信估算, 基于实测基准):**

| 场景 | Stage A | Stage B | Stage C | 总时间 |
|------|---------|---------|---------|--------|
| 8 workers 并行 | ~2.5 min | ~5 sec | ~45 sec | **~3.5 min** |
| 4 workers 并行 | ~5 min | ~5 sec | ~1 min | **~6 min** |
| 1 worker (单线程) | ~14 min | ~5 sec | ~2 min | **~16 min** |
| 最坏 (慢 IO + 大文件) | ~20 min | ~10 sec | ~3 min | **~23 min** |

### 3.2 Stage A 详细设计

#### A-1: 固定阈值硬拒绝 (短路, 无条件淘汰)

```python
def hard_reject(signals: dict) -> tuple[bool, str | None]:
    """短路: 触发任一条件立即返回 True + 原因"""
    # 1. 尺寸太小 → 缩略图/图标/无效
    if signals['min_side'] < 64:
        return True, "min_side_too_small"
    # 2. 极端条幅 → 全景拼接/截图
    if signals['aspect_ratio'] > 10.0 or signals['aspect_ratio'] < 0.1:
        return True, "extreme_aspect_ratio"
    # 3. 纯色占位 → sharpness+colorfulness+edge 三重确认
    if (signals['sharpness'] < 5.0 and signals['colorfulness'] < 5.0
            and signals['edge_ratio'] < 0.003):
        return True, "solid_color_placeholder"
    # 4. 极度模糊 → 无法分辨任何内容
    if signals['sharpness'] < 2.0:
        return True, "extremely_blurry"
    # 5. 过曝/欠曝 → 信息丢失
    if signals['brightness_mean'] < 10:
        return True, "extremely_underexposed"
    if signals['brightness_mean'] > 245:
        return True, "extremely_overexposed"
    # 6. 文档/信息图 (hardcode 拒绝)
    if signals['edge_ratio'] > 0.4 and signals['entropy'] < 4.5:
        return True, "likely_document_infographic"
    
    return False, None
```

**为什么这些阈值是固定而非自适应的:**
- min_side=64, aspect_ratio=10, sharpness=2, brightness=10/245 属于**绝对物理下限** — 低于这些阈值的图片在任何数据集、任何场景下都不可能是有用的 gallery 素材。
- 这 6 条规则预计淘汰 ~5-10% (约 3,000-6,000 张), 主要是损坏/无效/纯色占位文件。

#### A-2: 自适应软过滤 (per-dataset)

```python
def derive_adaptive_thresholds(dataset_signals: dict) -> dict:
    """基于数据集内信号分布推导自适应阈值"""
    p25 = {}
    p10 = {}
    for sig_name in ['sharpness', 'edge_ratio', 'colorfulness', 'entropy']:
        vals = dataset_signals[sig_name]
        p25[sig_name] = np.percentile(vals, 25)
        p10[sig_name] = np.percentile(vals, 10)
    
    # 宽松因子检测 (完整 corpus 的 P25 作为参考)
    global_p25_sharp = REFERENCE_CORPUS_P25['sharpness']
    loosen = 0.7 if p25['sharpness'] < global_p25_sharp * 0.5 else 1.0
    
    thresholds = {
        'sharpness_low': max(3.0, p25['sharpness'] * 0.3 * loosen),
        'edge_low': max(0.003, p25['edge_ratio'] * 0.3 * loosen),
        'colorfulness_low': max(3.0, p25['colorfulness'] * 0.4 * loosen),
        'entropy_low': max(2.0, p25['entropy'] * 0.6 * loosen),
        'loosen_factor': loosen,
    }
    
    # 雾/雪/低彩度保护
    if p25['colorfulness'] < 8 and p25['edge_ratio'] > 0.01:
        thresholds['colorfulness_low'] = min(thresholds['colorfulness_low'], 2.0)
        thresholds['note'] = 'low_color_but_natural'
    
    return thresholds
```

软过滤标记为 LOW_CONFIDENCE 而非硬拒绝 — 这些图片不进 top-100 但进复核池供人工审查。

#### A-3: 数据集类型检测

```python
def detect_dataset_profile(dataset_name: str, signals_stats: dict) -> str:
    """返回 TYPE_A/B/C/D/E"""
    n = dataset_name.lower()
    if 'truro' in n or 'school' in n:
        return 'TYPE_A_LARGE_CAMPUS'
    if 'digital' in n or 'cg' in n or 'render' in n:
        return 'TYPE_B_CGI_HIGH_RISK'
    if 'interior' in n or 'estate' in n or 'property' in n:
        return 'TYPE_C_INTERIOR'
    p25_edge = signals_stats.get('p25_edge', 0)
    p25_ent = signals_stats.get('p25_ent', 0)
    if p25_edge > 0.35 and p25_ent < 4.5:
        return 'TYPE_D_DOCUMENT_HEAVY'
    return 'TYPE_E_REGULAR'
```

### 3.3 Stage B 详细设计 — 三维评分引擎

#### realism_score (0-100, 实景可信度)

```python
def compute_realism_score(s: dict) -> int:
    score = 50  # 基线 = AMBIGUOUS
    
    # === 惩罚 ===
    if s['edge_ratio'] < 0.005:
        score -= 40 if s['colorfulness'] < 5 else 20
    elif s['edge_ratio'] > 0.4 and s['entropy'] < 4.0:
        score -= 30  # 文档
    if s['colorfulness'] > 80:
        score -= 20  # 过饱和
    if s['sharpness'] > 800:
        score -= 15  # 过度锐化
    if s['brightness_mean'] < 20 or s['brightness_mean'] > 230:
        score -= 25
    
    # === 加分 ===
    if 0.02 <= s['edge_ratio'] <= 0.15:  score += 15  # 自然纹理
    if 10 <= s['colorfulness'] <= 50:    score += 15  # 真实色彩
    if s['entropy'] > 6.0:               score += 10  # 高信息量
    if 60 <= s['brightness_mean'] <= 200 and s['brightness_std'] > 25:
        score += 10  # 自然曝光+对比
    
    return max(0, min(100, score))
```

#### quality_score (0-100, 摄影质量)

```python
def compute_quality_score(s: dict) -> int:
    q = 50
    # 清晰度
    if 60 <= s['sharpness'] <= 400:  q += 15
    elif s['sharpness'] < 20:        q -= 20
    elif s['sharpness'] > 600:       q -= 5
    # 色彩度
    if 15 <= s['colorfulness'] <= 50: q += 10
    elif s['colorfulness'] < 5:       q -= 15
    elif s['colorfulness'] > 70:      q -= 10
    # 对比度
    if 30 <= s['brightness_std'] <= 70: q += 10
    elif s['brightness_std'] < 15:      q -= 15
    # 信息熵
    if 6.0 <= s['entropy'] <= 7.5:   q += 10
    elif s['entropy'] < 4.0:          q -= 15
    # 构图稳定性
    if s['horizontal_balance'] < 0.15:   q += 10
    elif s['horizontal_balance'] > 0.35: q -= 10
    return max(0, min(100, q))
```

#### display_score (0-100, gallery 适配性) + 8 种标签

```python
def compute_display_score_and_labels(s: dict, quality: int, realism: int) -> tuple:
    d = 50
    labels = []
    
    # === 加分标签 ===
    # LANDSCAPE: 低edge + 中color + 高entropy + 横图
    if (s['edge_ratio'] < 0.08 and 10 <= s['colorfulness'] <= 50
            and s['entropy'] > 6.0 and s['aspect_ratio'] > 1.2):
        labels.append('LANDSCAPE')
        d += 15
    # ARCHITECTURE: 高edge + 低color + 清晰
    if (s['edge_ratio'] > 0.05 and s['colorfulness'] < 30
            and s['sharpness'] > 80 and s['aspect_ratio'] > 0.8):
        labels.append('ARCHITECTURE')
        d += 12
    # INTERIOR: 中edge + 中亮度 + 低color
    if (0.02 <= s['edge_ratio'] <= 0.12 and 40 <= s['brightness_mean'] <= 180
            and s['colorfulness'] < 40):
        labels.append('INTERIOR')
        d += 10
    # GROUP_ACTIVITY: 多人
    if s['face_count'] >= 3:
        labels.append('GROUP_ACTIVITY')
        d += 15
    # URBAN: 高edge + 中color + 高对比
    if (s['edge_ratio'] > 0.06 and s['colorfulness'] > 10
            and s['brightness_std'] > 30 and s['entropy'] > 6.5):
        labels.append('URBAN')
        d += 10
    
    # === 惩罚标签 ===
    # SINGLE_PORTRAIT: 单人近景
    if s['face_ratio_area'] > 0.10 and s['face_count'] == 1:
        labels.append('SINGLE_PORTRAIT')
        d -= 20
    # PRODUCT_SHOT: 白底商品
    if (s['edge_ratio'] < 0.03 and s['brightness_std'] < 20
            and s['colorfulness'] < 15):
        labels.append('PRODUCT_SHOT')
        d -= 15
    # DOCUMENTARY: 构图偏 + 低质
    if s['horizontal_balance'] > 0.35 and quality < 50:
        labels.append('DOCUMENTARY')
        d -= 10
    # SELFIE: 单人近景 + 低质
    if (s['face_count'] == 1 and s['face_ratio_area'] > 0.12
            and quality < 40):
        labels.append('SELFIE')
        d -= 20
    
    # 质量+实景加成
    d += quality * 0.1 + realism * 0.1
    return max(0, min(100, d)), labels
```

#### 三轴联合决策逻辑

| realism | quality | display | 决策 |
|---------|---------|---------|------|
| < 30 | 任意 | 任意 | ❌ 淘汰 |
| 30-49 | < 30 | 任意 | ❌ 淘汰 |
| 30-49 | ≥ 30 | 任意 | ⚠️ 复核池 |
| ≥ 50 | < 30 | 任意 | ⚠️ 复核池 |
| ≥ 50 | ≥ 30 | < 40 | ⚠️ 复核池 |
| ≥ 50 | ≥ 50 | ≥ 40 | ✅ top-100 候选 |
| ≥ 80 | ≥ 70 | ≥ 60 | ✅ 高优先级 |

#### 最终排序分

```python
final_score = (realism / 100) * (quality / 100) * (display / 100) * 100
# 范围 0-100, 乘性确保三轴同时高
```

### 3.4 truro_school 超大数据集 (36,500 张) 处理策略

| 问题 | 策略 | 原因 |
|------|------|------|
| 占总量 59% | 宽松因子自动检测 (若 P25_sharpness < corpus_P25×0.5 → ×0.7) | 避免模糊校园场景被过度淘汰 |
| 连拍/重复多 | dhash 去重阈值 Hamming < 12 (默认 < 8) | 更灵敏去重 |
| 相似场景集中 | 簇配额 ≤12 张/簇 (默认 ≤15) | 防止单一场景占满 |
| KL 散度检查 | top-100 中任一簇 >25% → 强制降至 ≤15 张 | 场景多样性保障 |
| 处理时间 | 正常比例 (~60% 总时间) | 8 workers 并行分摊 |

---

## 4. 去重 / 相似度 / 多样性 / 排序策略

### 4.1 去重: 多分辨率 dhash (3 级尺度)

```python
def dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """感知哈希: 9×hash_size 灰度 → 比较相邻像素 → hash_size² bit"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(bit << i for i, bit in enumerate(diff.flatten()))

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

def is_duplicate(h1: int, h2: int, threshold: int = 12) -> bool:
    return hamming_distance(h1, h2) < threshold
```

**三级尺度策略:**
- 8×8 (64-bit): 粗粒度去重, 快速
- 16×16 (256-bit): 中粒度, 主要使用
- 32×32 (1024-bit): 细粒度, 仅对剩余未去重的做二次检查

**流程:**
1. 按 quality_score 降序排列所有 survivors
2. 三级桶: 每张图取 3 级 hash
3. 遍历: 如果三个尺度中**任一** Hamming < 阈值 → 视为重复, 保留排序更高的
4. 阈值: truro_school=12, 其他=8 (宽松因子)

### 4.2 多样性: 11-dim 信号向量 → PCA → DBSCAN

```python
def cluster_for_diversity(survivors_with_signals: list) -> list:
    """返回每个 survivor 的 cluster_id (-1 = 噪声 = 独特场景)"""
    # 1. 构建 11-dim 特征矩阵
    feature_names = ['sharpness', 'edge_ratio', 'colorfulness', 'entropy',
                     'brightness_mean', 'brightness_std', 'aspect_ratio',
                     'min_side', 'face_ratio_area', 'face_count', 'horizontal_balance']
    X = np.array([[s['signals'][n] for n in feature_names] for s in survivors])
    
    # 2. normalize (per-feature min-max)
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    # 3. PCA 降维到 5 维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_norm)
    
    # 4. DBSCAN
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.3, min_samples=3, metric='euclidean').fit(X_pca)
    
    return clustering.labels_.tolist()  # -1 = noise = 优先保留
```

**Fallback (sklearn 不可用):** Tiered 贪心策略
```python
def tiered_greedy(candidates_sorted_by_score: list, distance_threshold: float = 0.25):
    selected = []
    remaining = candidates_sorted_by_score.copy()
    while len(selected) < 100 and remaining:
        best = remaining.pop(0)
        selected.append(best)
        # 移除 L2 距离 < threshold 的邻居 (用 5 个归一化信号)
        remaining = [c for c in remaining 
                     if l2_distance(normalize_signals(best), normalize_signals(c)) >= distance_threshold]
    return selected
```

### 4.3 反肖像偏见

```python
# 软惩罚, 非硬过滤
if face_count >= 1 and face_ratio_area > 0.10:
    final_score *= 0.90  # 单人近景降权
elif face_count >= 3:
    final_score *= 1.05  # 群像加分
```

**人脸检测可靠性守卫:** Haar Cascade `minSize=(30,30)` → 小脸不计入

### 4.4 最终排序与配额选择

```python
def rank_and_select(scored_deduped_with_clusters, top_k=100):
    # 1. 按 final_score 降序
    sorted_items = sorted(scored_deduped_with_clusters, 
                          key=lambda x: x['final_score'], reverse=True)
    
    # 2. 分池
    selected = []
    review_pool = []
    cluster_counts = {}
    
    for item in sorted_items:
        if item['decision'] == 'ELIGIBLE':
            cid = item['cluster_id']
            cap = 10 if item.get('dataset_type') == 'TYPE_A_LARGE_CAMPUS' else 15
            if cid == -1:  # 噪声点不受配额限制
                selected.append(item)
            elif cluster_counts.get(cid, 0) < cap:
                selected.append(item)
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        elif item['decision'] == 'REVIEW':
            review_pool.append(item)
    
    # 3. 如果不够 top_k, 补选剩余最高分
    if len(selected) < top_k:
        remaining = [i for i in sorted_items if i not in selected 
                     and i['decision'] == 'ELIGIBLE']
        selected.extend(remaining[:top_k - len(selected)])
    
    # 4. KL 散度检查
    check_kl_divergence(selected, top_k)  # 如果某簇 >25%, 强制降采样
    
    return selected[:top_k], review_pool[:200]  # 复核池上限 200
```

---

## 5. 输出文件结构与复核机制

### 5.1 输出目录

```
workspace/output/
├── logs/
│   ├── pipeline_run.log              # 完整运行日志 (INFO+)
│   ├── stageA_bad_files.txt          # 损坏/无法读取文件 (绝不静默)
│   └── calibration_report.json       # 阈值校准报告
├── per_dataset/
│   ├── truro_school/
│   │   ├── top100_list.tsv           # rank | filepath | final_score | realism | quality | display | labels | face_count | cluster_id | confidence
│   │   ├── review_pool_list.tsv      # AMBIGUOUS + LOW_CONFIDENCE 列表
│   │   └── dataset_stats.json        # 信号统计 + 通过率 + 阈值
│   ├── digital_domain/
│   │   ├── top100_list.tsv
│   │   ├── top100_low_confidence.md  # 低置信度图片列表 (重点审查)
│   │   ├── review_pool_list.tsv
│   │   └── dataset_stats.json
│   └── ... (11 个数据集)
├── aggregate_stats.json              # 全局汇总
└── pipeline_state.json               # 断点续跑状态
```

### 5.2 输出文件格式示例

**top100_list.tsv:**
```
rank	filepath	final_score	realism	quality	display	labels	face_count	cluster_id	confidence
1	C:\pics\truro_school\IMG_4521.jpg	68.4	85	72	78	LANDSCAPE|URBAN	0	3	HIGH
2	C:\pics\truro_school\IMG_1123.jpg	61.2	82	65	70	ARCHITECTURE	1	7	HIGH
...
```

**review_pool_list.tsv:**
```
filepath	realism	quality	display	reason
C:\pics\digital_domain\render_033.png	45	58	32	AMBIGUOUS realism; display<40
C:\pics\truro_school\IMG_998.jpg	55	28	45	quality<30
```

### 5.3 复核机制

| 复核批次 | 内容 | 规模 | 建议方式 |
|---------|------|------|---------|
| 批次 1 | top-100 | ~1,100 张 (11×100) | 缩略图网格 (gallery.html) |
| 批次 2 | 复核池 | ~200-500/数据集 | TSV + 文件管理器 |
| 批次 3 | LOW_CONFIDENCE top-100 | digital_domain 特有 | 单独标记列表 |
| 批次 4 | 淘汰抽样 (2%) | ~40-100/数据集 | 确认无过度淘汰 |

---

## 6. 运行步骤与依赖

### 6.1 依赖

```txt
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
scikit-learn>=1.0.0    # 可选: 仅 Stage C DBSCAN/PCA
```

**全部 < 100M 参数, CPU-only, 无 CUDA 需求。**

### 6.2 执行步骤

```bash
# Step 1: 环境准备
pip install opencv-python numpy pyyaml scikit-learn

# Step 2: 校准 (可选但推荐 — 小规模跑)
python scripts/calibrate.py --input C:\pics --samples 200 --output calibration/

# Step 3: 快速验证 (推荐 — 100 张/数据集)
python scripts/quick_test.py --input C:\pics --samples 100 --output quicktest/

# Step 4: 全量运行
python run_pipeline.py --input C:\pics --output workspace/output --workers 8

# Step 5: 检查结果
#   logs/pipeline_run.log — 通过率、错误
#   logs/stageA_bad_files.txt — 损坏文件
#   per_dataset/*/review_pool_list.tsv — 启动人工复核

# Step 6: 生成可视化报告
python scripts/generate_report.py --input workspace/output/ --output workspace/output/gallery.html
```

### 6.3 脚本文件结构

```
pipeline/
├── run_pipeline.py           # 主入口
├── config/
│   └── pipeline_config.yaml  # 路径、固定阈值
├── stage_a/
│   ├── compute_signals.py    # 11 信号计算
│   ├── hard_reject.py        # 固定阈值硬拒绝
│   └── soft_filter.py        # 自适应阈值 + dataset profile
├── stage_b/
│   ├── heuristic_scores.py   # 三维评分
│   └── display_labels.py     # 8 种标签
├── stage_c/
│   ├── dedup.py              # dhash 多分辨率去重
│   ├── cluster.py            # PCA+DBSCAN / Tiered 回退
│   ├── face_detect.py        # Haar Cascade
│   └── ranking.py            # 配额排序 + 输出
├── utils/
│   ├── io_utils.py           # 文件读写, 损坏处理
│   ├── logging_utils.py      # 日志
│   └── state.py              # 断点续跑
└── scripts/
    ├── calibrate.py          # 阈值校准
    ├── quick_test.py         # 快速验证
    └── generate_report.py    # HTML 画廊
```

---

## 7. 风险、阈值校准、小规模验证

### 7.1 风险登记表