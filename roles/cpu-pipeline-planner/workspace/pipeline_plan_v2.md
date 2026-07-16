# Executor Pipeline Plan — CPU-only Windows 10 落地实现

> 基于: 视觉判据专家三轴判断框架 (实景可信度5级 × 展示质量4级 × gallery适配性联合决策) + Stage 1 扫描结论  
> 目标: C:\pics 下 61,958 张图片, 11 个数据集 → 每数据集 top-100 gallery + 复核池  
> 核心约束: CPU-only (Intel UHD 770, 无 CUDA), 模型 <100M params, 不依赖非视觉元数据

---

## 0. 视觉判断逻辑 — 三轴联合决策框架 (专家共识)

### 0.1 三轴定义

| 轴 | 级别数 | 级别 | 含义 |
|----|--------|------|------|
| **A: 实景可信度** | 5 级 | NON_REAL(0) / PROBABLY_NON_REAL(1) / AMBIGUOUS(2) / PROBABLY_REAL(3) / REAL(4) | 图片内容多大程度上是真实相机拍摄的自然场景 |
| **B: 展示质量** | 4 级 | POOR(0) / FAIR(1) / GOOD(2) / EXCELLENT(3) | 构图/曝光/清晰度/色彩的综合摄影质量 |
| **C: gallery适配性** | 联合 | A×B 交叉表 (见 0.2) | 是否适合放入官网 gallery |

### 0.2 联合决策矩阵 (C = f(A, B))

| A(实景) \ B(质量) | POOR(0) | FAIR(1) | GOOD(2) | EXCELLENT(3) |
|-------------------|---------|---------|---------|-------------|
| **NON_REAL(0)** | ❌ 淘汰 | ❌ 淘汰 | ❌ 淘汰 | ❌ 淘汰 |
| **PROBABLY_NON_REAL(1)** | ❌ 淘汰 | ❌ 淘汰 | ❌ 淘汰 | ⚠️ 复核池 |
| **AMBIGUOUS(2)** | ❌ 淘汰 | ⚠️ 复核池 | ⚠️ 复核池 | ⚠️ 复核池 |
| **PROBABLY_REAL(3)** | ❌ 淘汰 | ⚠️ 复核池 | ✅ top-100 | ✅ top-100 |
| **REAL(4)** | ❌ 淘汰 | ⚠️ 复核池 | ✅ top-100 | ✅ top-100 |

**关键原则:**
- **边界样本不进 top-100**: AMBIGUOUS 无论质量多高都进复核池
- **PROBABLY_NON_REAL 质量极高也进复核池**: 允许人工判断是否高质感渲染误杀
- **保守优先**: 不足 100 时不从复核池补入, 宁缺毋滥

### 0.3 启发式硬拒绝规则 (Stage A 核心)

这些不需要模型, 纯 OpenCV 即可判断:

| 拒绝原因 | 条件 | 物理含义 |
|---------|------|---------|
| 极端条幅 | `aspect_ratio > 10.0` 或 `< 0.1` | 超宽拼接 / 超长竖条 |
| 纯色占位 | `edge_ratio < 0.005` 且 `colorfulness < 5.0` | 占位图 / 纯色背景 |
| 极度模糊 | `sharpness < 5.0` | 完全跑焦 / 损坏 |
| 文档/信息图 | `edge_ratio > 0.4` 且 `entropy < 4.0` | 文本截图 / 数据表 |
| 小缩略图 | `min(w,h) < 64` | 图标 / 编码缩略图 |
| 过曝/欠曝 | `brightness_mean > 230` 或 `< 20` | 纯白/纯黑 |

**注意**: 这些只用于**硬拒绝**, 正常图片仍需要模型做 5 级实景判别。

---

## 1. 总体架构: 三级级联 + 推理回避策略

```
全量 61,958 张
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage A: 启发式预筛 (零模型)                  │
│  7 维视觉信号 + 硬拒绝规则 + 自适应阈值       │
│  预计: 18,000~25,000 幸存者 (30%~40%)         │
│  耗时: ~5 min (8 workers)                    │
│  输出: survivors + rejected + bad_files       │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage B: 轻量特征 + 分类                      │
│  MobileNetV3-Small (2.5M params, ONNX CPU)    │
│  仅对 Stage A 幸存者推理                      │
│  输出: 576-dim 特征向量 + 5 级实景 + 4 级质量 │
│  耗时: ~3-5 hours                             │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage C: 去重 + 多样性 + 排序                 │
│  dhash 去重 → DBSCAN 聚类 → 反人脸偏见       │
│  → 配额排序 → 每数据集 top-100 + 复核池       │
│  耗时: ~10-15 min                             │
└──────────────────────┬───────────────────────┘
                       ▼
  输出: 11 个 per-dataset top-100 + review_pool + 统计
```

### 为什么这样分级?

| 方案 | 耗时估计 | 问题 |
|------|---------|------|
| 全量 MobileNetV3-Small (61,958 张) | ~12-18 小时 | 太长, 且大量低质图浪费推理 |
| 全量启发式 + 直接排序 | ~2 小时 | 无语义理解, CGI/AI/高质渲染无法区分 |
| **本方案 A→B→C** | **~4-6 小时** | **最优: 早淘汰 + 针对性模型 + 最后精选** |

---

## 2. Stage A: 启发式预筛 (零模型, CPU-only 最优)

### 2.1 每张图片提取的视觉信号

| 信号 | 计算方法 | 实现 | 成本/张 |
|------|---------|------|--------|
| `sharpness` | Laplacian 方差 | `cv2.Laplacian(gray, cv2.CV_64F).var()` | ~1ms |
| `edge_ratio` | Canny 边缘占比 | `cv2.Canny(50,150)` → `count_nonzero/total` | ~2ms |
| `colorfulness` | Hasler-Susstrunk M | RG-YG 空间标准差 (见论文公式) | ~0.5ms |
| `entropy` | 灰度直方图香农熵 | `skimage.measure.shannon_entropy` 或 numpy | ~0.5ms |
| `brightness_mean` | 灰度均值 | `gray.mean()` | ~0.2ms |
| `brightness_std` | 灰度标准差 | `gray.std()` | ~0.2ms |
| `aspect_ratio`, `min_side` | 宽高比 + 最小边长 | shape 直接计算 | ~0.1ms |
| **合计** | | | **~4-5ms/张** |

### 2.2 硬拒绝规则 (0.3 节的工程实现)

对每张图片按顺序检查 (短路: 一旦触发就跳过后续检查):

```python
def reject_image(signals) -> str | None:  # 返回 None=通过, 字符串=拒绝原因
    if signals.min_side < 64:
        return f"min_side={signals.min_side}<64"
    if signals.aspect_ratio > 10.0 or signals.aspect_ratio < 0.1:
        return f"aspect_ratio={signals.aspect_ratio:.2f} not in [0.1, 10.0]"
    if signals.sharpness < 5.0:
        return f"sharpness={signals.sharpness:.1f}<5"
    if signals.edge_ratio < 0.005 and signals.colorfulness < 5.0:
        return f"edge_ratio={signals.edge_ratio:.4f}<0.005 & colorfulness={signals.colorfulness:.1f}<5"
    if signals.edge_ratio > 0.4 and signals.entropy < 4.0:
        return f"edge_ratio={signals.edge_ratio:.3f}>0.4 & entropy={signals.entropy:.2f}<4"
    if signals.brightness_mean > 230 or signals.brightness_mean < 20:
        return f"brightness={signals.brightness_mean:.0f} extreme"
    return None
```

### 2.3 自适应阈值 (Per-dataset 百分比, 不硬编码)

对每个数据集:
1. 计算该数据集所有图片的 7 信号
2. 获取各信号的 P10, P25, P50, P75, P90
3. 自适应阈值:

```python
sharpness_th     = max(5.0,   P25_sharpness * 0.6)
edge_ratio_th    = max(0.005, P25_edge * 0.5)
colorfulness_th  = max(5.0,   P25_colorfulness * 0.5)
entropy_th       = max(2.0,   P25_entropy * 0.7)
```

**宽松模式** (对 truro_school 等模糊数据集):
- 检测条件: `P25_sharpness < global_P25_sharpness * 0.5`
- 触发后: 全部阈值乘以 `0.7` 因子

### 2.4 truro_school 特殊处理

| 问题 | 方案 |
|------|------|
| 占总量 59% (~36,500 张) | 宽松阈值 (×0.7) 避免过度淘汰 |
| 可能大量模糊/低质图 | 自适应: 如果 P25_sharpness < 10, 自动放宽松 |
| 连拍/重复场景 | 预留 30% 淘汰余量给 Stage C 去重 |
| 不需要特殊代码路径 | 通用自适应阈值即可覆盖 |

### 2.5 并发策略

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

# 按数据集分组后, 每个 worker 处理一个数据集的子集
# 每 batch = 500 张 → 减少进程通信开销
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_batch, batch): batch_id for batch in batches}
    for future in as_completed(futures):
        batch_result = future.result()
        # 写入共享队列/文件
```

**为什么不是 ThreadPool?** OpenCV 的 GIL 释放有限, ProcessPool 可获得真实并行。

### 2.6 缓存策略

- **中间结果**: `cache/stageA_signals_{dataset}.parquet` — 每张图片的 7 信号
- **已计算图片**: 记录 SHA-256 文件名列表, 断点续跑时跳过
- **阈值**: `config/thresholds_{dataset}.json` — 校准后缓存

### 2.7 Stage A 输出

```json
{
  "survivors": {
    "truro_school": [
      {"filepath": "C:/pics/truro_school/IMG_001.jpg",
       "signals": {"sharpness": 124.3, "edge_ratio": 0.032, ...}}
    ]
  },
  "rejected": {
    "truro_school": [
      {"filepath": "C:/pics/truro_school/blurry.jpg",
       "reason": "sharpness=2.1<5.0; edge_ratio=0.002<0.005"}
    ]
  },
  "thresholds": {
    "truro_school": {"sharpness_th": 12.5, "edge_ratio_th": 0.008, ...}
  }
}
```

---

## 3. Stage B: 轻量神经网络分类 (仅对 ~18-25k 幸存者)

### 3.1 模型选择

| 模型 | 参数 | ONNX CPU 推理/张 | 18,000 张耗时 | <100M? | 推荐? |
|------|------|----------------|-------------|--------|------|
| MobileNetV3-Small | 2.5M | ~12ms | ~3.6 min | ✅ | ✅ **首选** |
| MobileNetV3-Large | 5.4M | ~20ms | ~6 min | ✅ | 可选 |
| EfficientNet-B0 | 5.3M | ~25ms | ~7.5 min | ✅ | 备选 |
| ResNet-50 | 25.6M | ~50ms | ~15 min | ✅ | 过重 |
| MobileCLIP-S0 | ~12M | ~80ms | ~24 min | ✅ | CGI 困难集可用 |

**选择: MobileNetV3-Small 2.5M 参数, ONNX Runtime CPU 推理**

### 3.2 推理加速策略

1. **ONNX Runtime**: 比 PyTorch CPU 推理快 2-3x
2. **Session 复用**: 只加载一次, 复用全局 session
3. **Batch 推理**: `batch_size=32` 可减少 30% 总推理时间
4. **Int8 量化**: ONNX Runtime 的动态量化 (可选, 可降 ~50% 推理时间, 代价是约 1-2% 精度损失)

```
pip install onnxruntime onnx
# 模型下载: torchvision MobileNetV3-Small → export ONNX
```

### 3.3 分类任务设计 (共享 backbone, 两个头)

**头 1: 实景可信度 (5 级)**
```
输入: 576-dim feature (MobileNetV3 avgpool 输出)
线性层: 576 → 5 (softmax)
```

**头 2: 质量评分 (4 级)**
```
输入: 576-dim feature
线性层: 576 → 4 (softmax)
```

**免训练方案 k-means 聚类 (推荐初始版本):**
1. 用 MobileNetV3-Small (ImageNet 预训练) 提取 576-dim 特征
2. 对所有特征做 PCA(32) 降维
3. k-means(k=5) 聚类 → 每个簇对应一个实景可信度级别
4. 人工标注 5-10 张/簇 确定簇语义
5. 对新图片: 最近簇标签即为实景可信度

### 3.4 使用库和安装

```bash
pip install opencv-python numpy onnxruntime scikit-learn scikit-image
# ONNX 模型: 从 torchvision 导出或下载预转换版本
# 推荐: https://github.com/onnx/models/tree/main/vision/classification/mobilenet
```

### 3.5 Fallback: 如果模型推理太慢或不可用

**模式 B: 纯启发式评分函数 (完全不用模型)**

```python
def heuristic_realism_score(signals):
    """返回 0.0~1.0 的实景可信度分数"""
    edge_norm = min(signals.edge_ratio / 0.15, 1.0)       # 0.15 为理想边缘密度
    color_norm = min(signals.colorfulness / 40.0, 1.0)     # 40 为理想色彩度
    entropy_norm = min(signals.entropy / 7.0, 1.0)         # 7.0 为理想熵
    sharp_norm = min(signals.sharpness / 200.0, 1.0)       # 200 为理想清晰度
    
    score = 0.30*edge_norm + 0.30*color_norm + 0.20*entropy_norm + 0.20*sharp_norm
    return score

def heuristic_quality_score(signals):
    """返回 0.0~1.0 的质量分数"""
    sharp_norm = min(signals.sharpness / 300.0, 1.0)
    color_norm = 1.0 - abs(signals.colorfulness - 35.0) / 35.0  # 适中最好
    entropy_norm = min(signals.entropy / 7.5, 1.0)
    contrast_norm = min(signals.brightness_std / 60.0, 1.0)
    
    score = 0.25*sharp_norm + 0.25*color_norm + 0.25*entropy_norm + 0.25*contrast_norm
    return max(0.0, min(1.0, score))
```

**警告**: 此模式无法区分高质感 CGI 渲染与真实照片, 对 digital_domain 等高危数据集无效。

### 3.6 Stage B 输出

```json
{
  "truro_school": [
    {
      "filepath": "C:/pics/truro_school/IMG_4521.jpg",
      "features_576": [0.12, -0.34, ...],  // 用于 Stage C 聚类
      "realism": "REAL",                    // 5 级
      "realism_conf": 0.87,
      "quality": "EXCELLENT",               // 4 级
      "quality_conf": 0.72
    }
  ]
}
```

---

## 4. Stage C: 去重 + 多样性 + top-100 排序

### 4.1 dhash 去重

```python
def dhash(image, hash_size=8):
    """计算 64-bit dhash (difference hash)"""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# 去重: Hamming distance < 8 → 重复对
# 重复时保留 quality_conf 更高的那张
```

**性能**: ~1ms/张, 用 dict 做 exact dedup: O(n)

**truro_school 特殊**: 预期可去重 20-40% (连拍、重复场景)

### 4.2 多样性: DBSCAN 聚类

```
输入: Stage B 576-dim 特征向量 (去重后)
1. PCA 降维: 576 → 32 (加速距离计算, sklearn.decomposition.PCA)
2. DBSCAN(eps=0.5, min_samples=3, metric='euclidean', n_jobs=-1)
3. 输出: cluster_id (-1 = 噪声 = 独特场景)
```

**备选 (sklearn 不可用)**: Tiered 贪心策略
```python
# 按 quality 降序 → 贪心选取 → 移除 L2 邻居 → 重复
selected = []
candidates = all_images.copy()
while len(selected) < 100:
    best = candidates.pop(0)  # 最高 quality
    selected.append(best)
    # 移除特征空间 L2 < 0.5 的邻居
    candidates = [c for c in candidates 
                  if np.linalg.norm(np.array(best.features) - np.array(c.features)) > 0.5]
```

### 4.3 反人脸偏见

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                       minNeighbors=5, minSize=(30, 30))
has_face = len(faces) > 0
# 如果有正面人脸: final_score *= 0.9 (软惩罚, 不硬过滤)
```

**可靠性守卫**: 仅 detectMultiScale 返回 faces>0 且 minSize>=30 时触发。侧脸/遮挡/小脸不触发。

### 4.4 最终排序公式

```python
realism_weight = {'REAL': 1.0, 'PROBABLY_REAL': 0.8, 'AMBIGUOUS': 0.5,
                  'PROBABLY_NON_REAL': 0.3, 'NON_REAL': 0.1}
quality_weight = {'EXCELLENT': 1.0, 'GOOD': 0.7, 'FAIR': 0.4, 'POOR': 0.1}

final_score = (
    realism_weight[realism] * 
    quality_weight[quality] * 
    (0.9 if has_face else 1.0)
)
```

### 4.5 配额排序 (确保多样性)

```python
# 1. 按 final_score 降序排列
sorted_images = sorted(candidates, key=lambda x: -x.final_score)

# 2. 配额: 每簇最多 15 张 (噪声簇 -1 不受限制)
selected, cluster_counts = [], {}
for img in sorted_images:
    cid = img.cluster_id
    if cid == -1 or cluster_counts.get(cid, 0) < 15:
        selected.append(img)
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    if len(selected) >= 100:
        break

# 3. 如果不够 100 张: 从复核池按分数补选 (不受配额限制)
```

### 4.6 truro_school 特殊处理

| 措施 | 实现 |
|------|------|
| KL 散度偏差检查 | top-100 中某簇 >30% → 强制降采样至该簇 ≤20 张 |
| 分层配额 | 按 DBSCAN 簇大小分配席位: `max(1, round(cluster_size/total*100))` |
| 宽松阈值 | Stage A 已经应用宽松 (×0.7) |

---

## 5. 脚本模块划分与实现细节

### 5.1 文件组织

```
pipeline/
├── run_pipeline.py              # 主入口: A→B→C 串行, 支持断点续跑
├── config/
│   └── pipeline_config.yaml     # 全局配置: 路径、阈值、模型
├── stage_a/
│   └── heuristic_filter.py      # Stage A: 信号计算 + 硬拒绝 + 自适应阈值
├── stage_b/
│   ├── feature_extractor.py     # MobileNetV3-Small ONNX 推理
│   ├── classifier.py            # 5 级实景 + 4 级质量分类
│   └── heuristic_fallback.py    # 模式 B: 纯启发式评分
├── stage_c/
│   ├── dedup.py                 # dhash 去重
│   ├── diversity.py             # DBSCAN 聚类 + Tiered 备选
│   ├── face_penalty.py          # Haar Cascade 人脸检测
│   └── ranking.py               # 配额排序 + top-100 输出
├── utils/
│   ├── image_signals.py         # 7 信号计算函数
│   ├── io_utils.py              # 文件读写、路径统一处理(UTF-8)
│   ├── logging_utils.py         # 统一日志
│   └── state.py                 # 断点续跑状态管理
├── scripts/
│   ├── calibrate_thresholds.py  # 阈值校准 (可独立运行)
│   ├── quick_test.py            # 100 张/数据集快速验证
│   └── generate_report.py       # 从结果生成 HTML gallery
└── requirements.txt             # opencv-python, numpy, onnxruntime, scikit-learn, pyyaml
```

### 5.2 主入口逻辑

```python
# run_pipeline.py 伪代码
def main():
    config = load_config("config/pipeline_config.yaml")
    state = load_state("output/pipeline_state.json")
    
    # Stage A
    if state.get("stage_a_done") != True:
        # A1: 扫描全量数据集 → 计算 7 信号 (8 workers)
        all_signals = stage_a.compute_all_signals(config.input_dir)
        # A2: 自适应阈值校准
        thresholds = stage_a.calibrate_thresholds(all_signals)
        # A3: 应用硬拒绝规则
        survivors, rejected = stage_a.apply_filters(all_signals, thresholds)
        state["stage_a_done"] = True
        save_state(state)
    
    # Stage B
    if state.get("stage_b_done") != True:
        # B1: 加载 survivor 列表
        # B2: MobileNetV3-Small 特征提取 (batch_size=32)
        features = stage_b.extract_features(survivors)
        # B3: 分类 (5 级实景 + 4 级质量)
        results = stage_b.classify(features)
        state["stage_b_done"] = True
        save_state(state)
    
    # Stage C
    if state.get("stage_c_done") != True:
        # C1: dhash 去重
        deduped = stage_c.dedup(results)
        # C2: DBSCAN 聚类
        clustered = stage_c.cluster(deduped)
        # C3: 人脸惩罚
        scored = stage_c.apply_face_penalty(clustered)
        # C4: 配额排序
        top100_per_dataset = stage_c.rank_and_select(scored)
        # C5: 输出
        stage_c.write_outputs(top100_per_dataset)
        state["stage_c_done"] = True
        save_state(state)
    
    # 汇总
    generate_summary_report()
```

### 5.3 断点续跑状态文件

```json
{
  "stage_a_done": true,
  "stage_b_done": false,
  "stage_c_done": false,
  "datasets_processed": ["truro_school", "digital_domain"],
  "total_processed": 61958,
  "total_errors": 3,
  "elapsed_seconds": 14500,
  "last_update": "2025-07-16T15:30:00"
}
```

### 5.4 日志设计

| 级别 | 输出 | 内容 |
|------|------|------|
| ERROR | stderr + log | 无法读取文件、模型加载失败、异常 |
| WARN | log | 通过率异常 (<5% 或 >80%)、低置信度分类 |
| INFO | console + log | 阶段完成、计数、进度条 |
| DEBUG | log (默认关) | 每张图片的信号值 |

---

## 6. 输出目录组织

```
output/
├── logs/
│   ├── pipeline_run.log          # 完整运行日志
│   ├── stageA_bad_files.txt      # 损坏/无法读取文件列表 (绝不静默)
│   └── calibration_report.json   # 阈值校准报告
├── stageA_results.json           # Stage A 中间结果
├── stageB_results.json           # Stage B 中间结果
├── per_dataset/
│   ├── truro_school/
│   │   ├── top100_list.txt       # rank | filepath | score | realism | quality | cluster_id | has_face
│   │   ├── review_pool_list.txt  # 需要人工复核的边界样本列表
│   │   ├── rejected_examples.txt # 淘汰样本抽样 (2%) 供复核淘汰合理性
│   │   └── gallery.html          # 缩略图网格 (可选, 由 generate_report.py 生成)
│   ├── digital_domain/
│   │   └── ...
│   └── ... (共 11 个数据集)
├── aggregate_stats.json          # 全局汇总
└── pipeline_state.json           # 断点续跑状态
```

### 输出文件格式示例

**top100_list.txt:**
```
rank    filepath                                final_score realism        quality     cluster_id  has_face
1       C:\pics\truro_school\IMG_4521.jpg       0.8500      REAL           EXCELLENT   3           false
2       C:\pics\truro_school\IMG_1123.jpg       0.8200      REAL           GOOD        7           true
...
```

**review_pool_list.txt:**
```
filepath                                realism       realism_conf quality     quality_conf  reason
C:\pics\digital_domain\render_033.png   AMBIGUOUS     0.52         GOOD        0.68          AMBIGUOUS+GOOD → 复核
C:\pics\truro_school\IMG_998.jpg        PROBABLY_REAL 0.71         FAIR        0.44          PROBABLY_REAL+FAIR → 复核
...
```

---

## 7. 抽样复核与质量控制

### 7.1 复核采样

| 复核对象 | 抽样率 | 目的 |
|---------|--------|------|
| Stage A 淘汰样本 | 2% 随机 | 确认淘汰是否过度 |
| AMBIGUOUS 分类 | 10% 随机 | 确认实景/非实景分界线 |
| top-100 入选 | 100% 输出 | 最终人工确认 |
| review pool | 全部 (上限 200/数据集) | 边界样本全部供审查 |

### 7.2 损坏文件处理

```python
# 绝不静默跳过
try:
    img = cv2.imread(str(filepath))
    if img is None:
        raise ValueError(f"cv2.imread returned None: {filepath}")
except Exception as e:
    bad_files.append({"filepath": str(filepath), "error": str(e)})
    logging.error(f"无法读取: {filepath} -> {e}")
    # 继续处理下一张
```

- 写入 `logs/stageA_bad_files.txt` (含完整路径 + 错误类型)
- 计入 `pipeline_state.json` 的 `total_errors`
- 脚本退出码为非零

---

## 8. 风险登记与缓解

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|------|------|---------|
| R1 | truro_school 通过率 <10% | 中 | 高 | 宽松阈值 ×0.7; 自适应校准; 人工审查 200 张样本 |
| R2 | CGI 渲染被误判为 REAL | 高 | 中 | digital_domain 标记高危; AMBIGUOUS 全部进复核池; Stage C 不自动补足 100 |
| R3 | 某数据集被 Stage A 全淘汰 | 低 | 高 | 校准检查: 通过率 <5% → WARN + 人工介入 |
| R4 | DBSCAN 在 576-dim 太慢 | 低 | 中 | PCA 32-dim; 备选 Tiered 贪心 |
| R5 | ONNX Runtime Windows 加载慢 | 中 | 低 | session 复用 + 预热; batch 推理 |
| R6 | 全量处理 >8 小时 | 中 | 高 | 断点续跑; 支持指定单数据集运行 |
| R7 | 中文路径/编码问题 | 中 | 中 | pathlib + UTF-8 统一处理 |
| R8 | Haar Cascade 侧脸误检 | 低 | 低 | 仅软惩罚 0.9, 不硬过滤 |
| R9 | 边界图片被错误淘汰 | 中 | 中 | 保守策略: 边界进复核池; 2% 淘汰样本抽检 |

---

## 9. Executor 执行检查清单

### 9.1 环境准备
- [ ] `pip install opencv-python numpy onnxruntime scikit-learn scikit-image pyyaml`
- [ ] 下载 MobileNetV3-Small ONNX 模型到 `pipeline/models/`
- [ ] 确认 `C:\pics` 下 11 个数据集目录可读

### 9.2 校准阶段
- [ ] 运行 `python scripts/calibrate_thresholds.py` → 生成 `calibration_report.json`
- [ ] 人工审查校准报告: 确认各数据集 P25 值、建议阈值、预计通过率
- [ ] 如有异常 (通过率 <5% 或 >80%), 手动调整 `pipeline_config.yaml`

### 9.3 快速验证
- [ ] 运行 `python scripts/quick_test.py` (100 张/数据集)
- [ ] 检查输出: 各阶段通过率、分类分布、top-100 合理性
- [ ] 如果快速验证通过, 进入全量运行

### 9.4 全量运行
- [ ] 运行 `python pipeline/run_pipeline.py` (预计 4-6 小时)
- [ ] 中途检查 `logs/pipeline_run.log` — 进度、通过率、错误数

### 9.5 结果交付
- [ ] 检查 `logs/stageA_bad_files.txt` — 损坏文件列表
- [ ] 检查 `per_dataset/*/top100_list.txt` — 各数据集结果
- [ ] 检查 `per_dataset/*/review_pool_list.txt` — 启动人工复核
- [ ] 运行 `python scripts/generate_report.py` → 生成 gallery HTML
- [ ] 最终交付: 11 个 top-100 gallery + 复核池 + `aggregate_stats.json`

---

## 10. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否全量跑模型 | 否 | Stage A 淘汰 ~65%, 剩余 ~30% 进模型 |
| 模型选择 | MobileNetV3-Small 2.5M ONNX | CPU 推理最快 (~12ms/张), <100M 约束 |
| 实景判别方式 | 5 级分类 + B 轴联合决策 | 避免硬切边界样本 |
| 去重算法 | dhash 64-bit | CPU 友好 (~1ms/张), 缩放鲁棒 |
| 多样性聚类 | DBSCAN + PCA 32-dim | 无需预知簇数, 噪声点独特场景优先 |
| 反人脸偏见 | Haar Cascade 软惩罚 ×0.9 | 轻量, 不硬过滤 |
| 阈值校准 | Per-dataset 自适应百分比 | 跨数据集泛化, 不硬编码 |
| 边界样本处理 | 进复核池, 不补入 top-100 | 保守优先 |
| 断点续跑 | pipeline_state.json | 长任务可靠性 (>4h) |
| 损坏文件 | 日志 + 非零退出, 绝不静默 | 透明可靠 |
