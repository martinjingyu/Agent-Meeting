# Stage 2 Pipeline Plan — 可直接交付 Executor

> 基于: Stage 1 探索结果 + 视觉判据专家建议（五级实景可信度 + 四级质量联合决策）
> 目标: CPU-only Windows 环境, 61,958 张图片 → 每数据集 top-100 gallery + 复核池
> 核心约束: 禁止全量重模型推理; 禁止依赖非视觉信息做分类; 边界样本不武断硬切

---

## 1. 总体架构: 三级级联 + 免模型启发式优先

```
全量 61,958 张
    │
    ▼
Stage A ── 启发式预筛 (纯 OpenCV/numpy, 零模型)
    │  ~60-80 min
    │  输出: 18,000-25,000 幸存者 + 淘汰池日志
    ▼
Stage B ── 轻量神经网络分类 (MobileNetV3-Small 2.5M params)
    │  ~3-5 hours (仅对 Stage A 幸存者)
    │  输出: 每张图片 5 级实景可信度 + 4 级质量评分
    ▼
Stage C ── 去重 + 多样性聚类 + 最终排序
    │  ~10-15 min
    ▼
输出: 11 个 per-dataset top-100 + 复核池 + 日志
```

### 为什么是三级而非两级或一级?

| 方案 | 总耗时 (估计) | 问题 |
|------|--------------|------|
| 全量 MobileNetV3 | ~12-18 hours | 不可接受 |
| 全量启发式 → 直接排序 | ~2 hours | 无语义理解, CGI/AI 无法区分 |
| 本方案 (A→B→C) | ~4-6 hours | **可接受**, 且 B 阶段只处理 ~30% 原图 |

---

## 2. Stage A: 启发式预筛 (零模型, CPU-only 友好)

### 2.1 输入
全部 61,958 张图片, 按数据集分组。

### 2.2 每张图片提取的视觉信号 (共 7 个)

| 信号 | 计算方法 | OpenCV 函数 | 计算成本 |
|------|---------|-------------|---------|
| `sharpness` | Laplacian 方差 | `cv2.Laplacian(→float64).var()` | ~1ms |
| `edge_ratio` | Canny 边缘像素占比 | `cv2.Canny(50,150)` → `count_nonzero / total` | ~2ms |
| `colorfulness` | Hasler & Susstrunk M metric | 见论文公式, RG-YG 空间标准差 | ~0.5ms |
| `entropy` | 灰度直方图香农熵 | `skimage.measure.shannon_entropy` 或 numpy 手算 | ~0.5ms |
| `brightness_mean` | 灰度均值 | `cv2.cvtColor→GRAY.mean()` | ~0.2ms |
| `brightness_std` | 灰度标准差 | `np.std(gray)` | ~0.2ms |
| `aspect_ratio` | w/h | 从 shape 计算 | ~0.1ms |
| `min_side` | min(w, h) | shape 计算 | ~0.1ms |

**总计: ~4-5ms/张, 61,958 张 → ~310s ≈ 5-6 min** (单线程)
多线程 (8 workers) → **~1-2 min**

### 2.3 阈值校准 (Per-dataset 自适应)

**关键设计: 不做全局硬阈值, 而是 per-dataset 百分比阈值**

对于每个数据集:
1. 计算该数据集内所有图片的 7 个信号
2. 取各信号的分布统计 (P10, P25, P50, P75, P90)
3. 自适应阈值计算:
   - `sharpness_threshold = max(global_min, dataset_P25_sharpness * 0.6)`
   - `colorfulness_threshold = max(8.0, dataset_P25_colorfulness * 0.5)`
   - `edge_ratio_threshold = max(0.01, dataset_P25_edge_ratio * 0.5)`
   - `entropy_threshold = max(3.0, dataset_P25_entropy * 0.7)`
   - `min_side_threshold = 64`  (固定, 过小的图不可能是好照片)
4. 推荐但不强制: 对 truro_school (59% 总量) 用较宽松阈值以免砍光

### 2.4 淘汰规则 (拒绝条件)

**一张图片进入 Stage B 需要同时满足:**

| 条件 | 判断 | 拒绝原因 |
|------|------|---------|
| 清晰度 | `sharpness >= sharpness_threshold` | 完全模糊/单色 |
| 边缘丰富度 | `edge_ratio >= edge_ratio_threshold` | 纯色背景/空场景 |
| 色彩丰富度 | `colorfulness >= colorfulness_threshold` | 黑白/单色/过曝 |
| 信息熵 | `entropy >= entropy_threshold` | 纯噪音/损坏/空白 |
| 最小边 | `min_side >= 64` | 缩略图/图标/太小编码图 |
| 宽高比 | `0.1 <= aspect_ratio <= 10.0` | 极端长条/拼接图 |

**宽松模式** (对 truro_school 等可能有大量低质图的数据集):
- 上述阈值乘以 0.7 (更宽松)
- 仍保留 `min_side >= 48` 和 `aspect_ratio` 边界

### 2.5 Stage A 输出

```
stageA_survivors.json       # [数据集, 文件名, 7 个信号值] 列表
stageA_rejected.json         # [数据集, 文件名, 拒绝原因] 列表
stageA_stats.json            # 每个数据集的通过率、分布统计
```

---

## 3. Stage B: 轻量神经网络分类

### 3.1 为什么是 MobileNetV3-Small 而非更大模型?

| 模型 | 参数量 | ONNX CPU 推理时间/张 | 1.8 万张耗时 |
|------|--------|---------------------|-------------|
| ResNet-50 | 25.6M | ~50ms | ~15 min |
| MobileNetV3-Small | 2.5M | ~12ms | ~3.6 min |
| MobileNetV3-Large | 5.4M | ~20ms | ~6 min |
| EfficientNet-B0 | 5.3M | ~25ms | ~7.5 min |
| MobileCLIP-S0 | ~12M | ~80ms | ~24 min |

**选择: MobileNetV3-Small 2.5M** — 在 <100M 约束下, 性价比最高

### 3.2 分类任务设计

两个独立的轻量分类头, 共享 backbone:

**头 1: 实景可信度 (5 级分类)**
- 训练标签映射:
  - `NON_REAL (0)`: 纯 CGI 渲染、AI 生成、截屏、UI 截图
  - `PROBABLY_NON_REAL (1)`: 高质感渲染、重度滤镜、AI 辅助合成
  - `AMBIGUOUS (2)`: 实景与 CGI 混合、HDR 过度处理、艺术滤镜
  - `PROBABLY_REAL (3)`: 实景但有小瑕疵 (轻度后期、轻微噪点)
  - `REAL (4)`: 明显实景照片、自然光、真实质感

**头 2: 质量评分 (4 级分类)**
- `POOR (0)`: 过曝/欠曝/严重噪点/模糊/构图失败
- `FAIR (1)`: 可用但有明显质量缺陷
- `GOOD (2)`: 清晰、曝光正常、构图合理
- `EXCELLENT (3)`: 光线/构图/清晰度俱佳

### 3.3 免训练方案 (零样本推理)

**如果无法训练分类头, 用以下 Zero-shot 策略:**

方案 A (推荐): MobileNetV3-Small 预训练 + 特征向量聚类
1. 用 ImageNet 预训练的 MobileNetV3-Small 提取 576-dim 特征向量
2. 在特征空间做 **k-means (k=5)** 聚类
3. 对每个簇: 聚合边界样本手动标注 10-20 张确定该簇的语义
4. 对新图片: 最近簇标签即为实景可信度
5. 质量评分: 从同一特征向量的辅助回归头 (线性层) 预测

方案 B (备选): 纯启发式评分函数 (完全不用模型)
```
realism_score = w1*edge_ratio + w2*colorfulness + w3*entropy + w4*(1-brightness_std_norm)
quality_score = w5*sharpness_norm + w6*colorfulness_norm + w7*entropy_norm + w8*asymmetry
```
权重从 Stage A 信号中自动拟合 — 但**不推荐**, 因为无法区分高质感渲染。

**推荐方案: 方案 A + 对 AMBIGUOUS 区域的图片额外做 10% 抽样送人工复核**

### 3.4 Stage B 输出

```
stageB_results.json
  # [
  #   {
  #     "dataset": "truro_school",
  #     "file": "IMG_001.jpg",
  #     "realism": "REAL",          # 5级
  #     "realism_conf": 0.87,
  #     "quality": "GOOD",          # 4级
  #     "quality_conf": 0.72,
  #     "features_576": [...]       # 用于 Stage C 多样性
  #   }, ...
  # ]
```

---

## 4. Stage C: 去重 + 多样性 + top-100 排序

### 4.1 去重: 多分辨率 dhash

**为什么不用 pHash?** dhash 在 CPU 上快 5-10x, 且对轻微缩放的 Robustness 足够。

算法:
```
for each image:
  1. resize to 9x8 (dhash 需要 9x8 灰度)
  2. compare horizontal pixel differences → 64-bit hash
  3. store as integer

去重判断: Hamming distance < 8 → 重复 (重复保留质量最高的那张)
```

**对于 truro_school (~15,000 幸存者):**
- 去重预计移除 20-40% (连拍、重复场景)
- 用 `minhash` 近似去重可加速: 分桶策略, O(n) 而非 O(n²)

### 4.2 多样性聚类: DBSCAN on 特征向量

使用 Stage B 提取的 576-dim 特征向量:

1. PCA 降到 32-dim (加速距离计算)
2. DBSCAN(eps=0.5, min_samples=3) 聚类
3. 每个簇代表一个"视觉场景类型"
4. 噪声点 = 独特场景, 优先保留

**替代方案 (如果 DBSCAN 太慢): Tiered 贪心策略**
```
1. 按 quality 排序
2. 从最高 quality 开始贪心选取
3. 每选一张, 移除特征空间 L2 距离 < threshold 的候选
4. 重复直到选满 100 张
```
复杂度 O(n²), 但对 ~15k 张约 20s 可完成。

### 4.3 反肖像偏见

**问题**: 近距人脸照往往 quality 评分高, 但多样性差。
**策略**: 检测到人脸 (OpenCV Haar Cascade, CPU-only) → quality 乘以 0.9 惩罚因子。
不硬过滤, 仅减少人脸照被过度选中的比例。

**可靠性守卫**: 仅当 Haar Cascade 检测到人脸且 confidence > 0.6 才启用惩罚。
侧脸/遮挡不触发, 避免误罚。

### 4.4 最终 top-100 排序公式

```
final_score = 
    realism_weight(realism)     # REAL=1.0, PROBABLY_REAL=0.8, AMBIGUOUS=0.5, ...
    * quality_weight(quality)   # EXCELLENT=1.0, GOOD=0.7, FAIR=0.4, POOR=0.1
    * 0.9_if_has_face
    * diversity_bonus           # 每个场景簇(DBSCAN cluster) 最多 15 张, 确保覆盖
```

### 4.5 针对 truro_school 的特殊处理

truro_school 占 59% 总量, 可能包含大量相似校园照片。

| 策略 | 操作 |
|------|------|
| 宽松阈值 | Stage A 阈值乘 0.7 因子 |
| KL散度偏差检查 | 检查 top-100 中每个场景簇的占比, 如果某簇 >30% → 强制降采样 |
| 分层配额 | 按 DBSCAN 簇大小分配 top-100 席位: `max(1, round(cluster_size/total*100))` |
| 人工复核标记 | top-100 中 AMBIGUOUS 置信度的图片需要人工确认 |

### 4.6 Stage C 输出

```
per_dataset/
  truro_school/
    top_100/              # 100 张最终入选 (copy or symlink)
      gallery.html        # 缩略图网格 + 评分标注
    review_pool/          # 边界样本 (50-200 张, 需要人工审)
      reviewed.json       # 人工审核结果记录
  digital_domain/
    top_100/
    review_pool/
  ... (共 11 个数据集)
```

---

## 5. 输出目录组织

```
output/
├── logs/
│   ├── stageA_run.log           # 每个文件处理记录, 包含错误
│   ├── stageA_bad_files.txt     # 无法读取/损坏文件列表
│   ├── stageB_run.log
│   ├── stageC_run.log
│   └── pipeline_summary.log     # 总览: 输入数, 各阶段通过数
├── per_dataset/
│   ├── truro_school/
│   │   ├── gallery.html         # 可视化展示
│   │   ├── top100_list.txt      # 文件名列表 + 排序分数
│   │   ├── review_pool_list.txt # 需人工复核的图片
│   │   └── stage_stats.json     # 本数据集各阶段结果统计
│   ├── digital_domain/
│   │   └── ...
│   └── ... (11 个数据集)
├── aggregate_stats.json         # 全局汇总
└── pipeline_state.json          # 可恢复的运行状态 (断点续跑用)
```

---

## 6. 日志与抽样复核设计

### 6.1 日志规则

| 级别 | 用途 | 输出到 |
|------|------|--------|
| ERROR | 无法读取/损坏/异常 | `logs/*.log` + stderr |
| WARN | 低质量通过样本、模型推理低置信度 | `logs/*.log` |
| INFO | 正常处理进度、各阶段计数 | `logs/*.log` + console |
| DEBUG | 每张图片信号值 (可选, 默认关闭) | `logs/*.log` |

### 6.2 损坏文件处理

**绝不静默跳过。** 机制:
1. `try/except` 包裹 `cv2.imread`
2. 失败时写入 `stageA_bad_files.txt` (含完整路径 + 错误类型)
3. 计入 `pipeline_state.json` 的 `errors` 计数
4. 脚本退出码为非零 (但不清除已处理结果)

### 6.3 抽样复核策略

| 复核类型 | 抽样率 | 样本来源 |
|---------|--------|---------|
| Stage A 淘汰复核 | 2% | 从每个数据集被淘汰的图片中随机抽 |
| Stage B AMBIGUOUS 复核 | 10% | `realism == AMBIGUOUS` 的全部图片 |
| Stage C top-100 复核 | 100% | 每数据集前 100 张全部输出供人工检查 |
| Stage C review pool | 全部 | 每数据集 50-200 张边界样本 |

---

## 7. 风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| truro_school 通过率过低 | 中 | 高 | 宽松阈值 + 单独参数校准 |
| CGI 渲染被误判为 REAL | 高 | 中 | Stage B 特征 + 标记 digital_domain 为高风险; AMBIGUOUS 抽样送审 |
| DBSCAN 在 576-dim 空间太慢 | 低 | 中 | 降维到 32-dim PCA; 备选 Tiered 贪心 |
| MobileNetV3-Small ONNX 推理慢于预期 | 中 | 中 | 使用 OpenVINO 运行时加速 (CPU 优化) |
| Haar Cascade 人脸检测精度差 | 中 | 低 | 仅做软惩罚 (0.9因子), 不做过滤 |
| 某个数据集全被 Stage A 淘汰 | 低 | 中 | 阈值校准后检查, 如果通过率 <5% 发出 WARN |
| 全量处理 >8 小时 | 中 | 高 | 支持断点续跑 (pipeline_state.json); 支持只处理指定数据集 |
| 硬盘空间不足 | 低 | 中 | 用符号链接而非 copy; 定期清理临时文件 |
| 中文路径/文件名编码问题 | 中 | 中 | 所有内部操作统一用 `pathlib.Path` + UTF-8 |

---

## 8. 阈值校准流程 (详细)

### 8.1 校准步骤

```
Step 1: 对每个数据集, 随机抽样 200 张
Step 2: 计算 7 个信号的分布 (P10, P25, P50, P75, P90)
Step 3: 人工审查抽样图片, 标记被淘汰/被保留的合理性
Step 4: 如果某数据集的 P25 sharpness < global_P10 → 该数据集整体模糊
         → 使用 "相对宽松" 模式的阈值 (乘 0.7)
Step 5: 对所有数据集应用自适应阈值公式 (见 2.3)
Step 6: 在全量跑之前, 先跑抽样验证: 校准集 200 张上的通过率应在 20-60% 之间
Step 7: 如果通过率不在范围内 → 调整百分比参数 (如 P25→P30 或 P20)
```

### 8.2 校准脚本位置

`scripts/calibrate_thresholds.py` — 可独立运行, 输出校准报告 + 建议参数

---

## 9. 脚本文件组织

```
pipeline/
├── run_pipeline.py              # 主入口: 串行执行 A→B→C
├── stage_a_heuristic.py         # Stage A 启发式预筛
├── stage_b_classifier.py        # Stage B 特征提取 + 分类
├── stage_c_dedup_diversity.py   # Stage C 去重 + 多样性 + 排序
├── utils/
│   ├── image_signals.py         # 7 个信号计算函数
│   ├── hash_utils.py            # dhash 实现
│   ├── clustering.py            # DBSCAN + PCA 封装
│   ├── face_detect.py           # Haar Cascade 人脸检测
│   ├── io_utils.py              # 文件读写、路径处理
│   └── logging_utils.py         # 统一日志配置
├── scripts/
│   ├── calibrate_thresholds.py  # 阈值校准
│   ├── quick_test.py            # 小样本快速验证 (100 张)
│   └── generate_report.py       # 从结果生成 HTML gallery
└── config/
    └── pipeline_config.yaml     # 全局配置: 路径、阈值、模型路径
```

---

## 10. 与 Stage 1 的衔接

Stage 1 的探索结果为:
- 数据集画像: 各数据集大小、平均分辨率、格式分布
- 信号分布: 每个数据集的信号均值和标准差
- 特殊发现: truro_school 占 59%, digital_domain 可能有 CGI

这些结果直接输入到:
1. `config/pipeline_config.yaml` 中 per-dataset 的 `init_threshold_multiplier` 字段
2. `calibrate_thresholds.py` 的初始参数
3. 风险登记表 (Section 7) 的优先级排序

---

## 11. Executor 执行检查清单

- [ ] 确认 `C:\pics` 数据完整可读
- [ ] 运行 `calibrate_thresholds.py` → 检查校准报告
- [ ] 运行 `quick_test.py` (100 张/数据集) → 检查各阶段输出
- [ ] 运行 `run_pipeline.py` (全量)
- [ ] 检查 `logs/pipeline_summary.log` → 各阶段计数
- [ ] 检查 `per_dataset/*/review_pool_list.txt` → 启动人工复核
- [ ] 运行 `generate_report.py` → 输出 `gallery.html`
- [ ] 最终交付: 11 个 top-100 gallery + 复核池 + aggregate_stats.json

---

## 附录: 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否全量跑模型 | 否 | Stage A 淘汰 ~65%, 剩余 ~30% 进模型 |
| 模型选择 | MobileNetV3-Small 2.5M | CPU 推理最快, <100M 约束 |
| 去重算法 | dhash 64-bit | CPU 友好, 对缩放鲁棒 |
| 多样性聚类 | DBSCAN + PCA 32-dim | 无需预知簇数 |
| 反人脸偏见 | Haar Cascade 软惩罚 | 轻量, 不依赖 GPU |
| 阈值校准 | Per-dataset 自适应百分比 | 跨数据集泛化 |
| 实景判断 | 5 级分类 + 联合质量决策 | 避免硬切 |
| 断点续跑 | pipeline_state.json | 长任务可靠性 |
