# 视觉定义框架：实景 vs 非实景 & Gallery 适配性

> 基于 Stage 1 跨 11 数据集、220+ 样本的实证验证，完全依赖 CPU-only 可计算的视觉判据

---

## 1. 实景图 (Real-Scene Photograph) 的视觉定义

### 1.1 核心定义

**实景图** = 由真实相机在物理世界中拍摄的数字影像，其视觉特征受物理光学、传感器噪声、自然光照分布和三维几何的约束。

### 1.2 可计算的视觉信号（全部 CPU-only，每图 ~0.05-0.09s）

| 信号 | 计算公式 | 实景典型范围 | 非实景典型范围 | 分离能力 |
|------|----------|-------------|----------------|----------|
| **Edge Ratio** | Canny 边缘像素数 / 总像素数 | 0.05 - 0.25 | 0.15 - 0.50 | **最强的单一分离器** |
| **Colorfulness** | Hasler-Susstrunk 指标 | 8 - 80 | 5 - 140 | **单独使用会误导** |
| **Brightness Std** | 像素亮度(0-255)的帧内标准差 | 40 - 80 | 10 - 40（文档）或 60-120（图形） | 辅助但不充分 |
| **Image Entropy** | 信息熵（灰度直方图） | 6.0 - 7.8 | 3.0 - 7.5（文档低，UI 可高） | 低熵时有用 |
| **Aspect Ratio** | 宽/高 | 0.5 - 3.0 | <0.25 或 >4.0 | 极值时强信号 |
| **Sharpness** | Sobel 梯度均值 (@640px) | 3.0 - 100+（室内可低至3.0） | 2.0 - 150+ | **无法单独分离** |

### 1.3 实证发现：为什么不能靠单一阈值

**实证数据（来自 heuristic_validation.json，220+ 样本）：**

| 数据集 | 中位 Sharpness | 中位 Colorfulness | 内容属性 |
|--------|---------------|-------------------|----------|
| m_immobilier | 6.9 | 29.2 | 全部实景（室内房产）|
| kpmg_forensic | 8.1 | 60.0 | 全部截图/UI |
| maior_capital | 62.0+ | 50.0+ | 全部实景（高端房产）|
| roland_berger | 30-150 | 20-125 | ~50% 信息图 |
| boston_university | 12-50 | 26-63 | 混合实景+UI |

**关键发现：**
1. **Sharpness 无分离力**：kpmg（截图）中位 sharpness=8.1，m_immobilier（实景）中位=6.9 — 两者几乎重叠
2. **Colorfulness 反向误导**：截图（kpmg 中位=60.0）比实景（m_immobilier 中位=29.2）更鲜艳
3. **Edge Ratio 是最佳信号**：截图/UI 的 edge_ratio 通常 >0.30，实景通常 <0.25
4. **跨数据集差异大**：m_immobilier p5 sharpness=3.0（真实室内照片可能很模糊），但 maior_capital 最低 sharpness=22.9

**结论：必须用多信号组合，不能用单一阈值。**

---

## 2. 实景/非实景分类体系（5 级）

### Level 1: NON_REAL（确定非实景）
**视觉证据：**
- `edge_ratio > 0.35 AND colorfulness > 70` → 彩色 UI 截图/信息图
- `aspect_ratio < 0.25 OR aspect_ratio > 4.0` → 极端横幅/广告
- `entropy < 3.5 AND brightness_std < 15` → 纯色占位/空白图
- `sharpness_640 < 2.0` → 极度模糊，无法分类
- `colorfulness < 4 AND edge_ratio > 0.20` → 文档/白底文字截图

### Level 2: PROBABLY_NON_REAL（可能非实景）
**视觉证据：** 不满足 Level 1 硬阈值，但复合软分 < 0.25
- 软分公式 = `0.25×colorfulness_score + 0.25×entropy_score + 0.25×brightness_std_score + 0.25×(1 - min(edge_ratio/0.3, 1.0))`

### Level 3: AMBIGUOUS（不确定）
**视觉证据：** 复合软分 0.25 - 0.40（或 0.40-0.55 但某个信号异常）
- 例如：真实背景 + 文字覆盖；高质感 3D 渲染；严重滤镜照片
- 全部进入人工复核池

### Level 4: PROBABLY_REAL（可能是实景）
**视觉证据：** 复合软分 0.40 - 0.55，所有信号在正常范围内
- 例如：轻度编辑的真实照片；低光室内场景；裁剪后的照片
- 进入 gallery 候选但标注需复核

### Level 5: REAL（确定实景）
**视觉证据：** 复合软分 ≥ 0.55，同时满足：
- colorfulness ≥ 8（排除文档）
- edge_ratio < 0.35（排除 UI）
- entropy ≥ 4.5（排除纯色）
- sharpness_640 ≥ max(2.0, dataset_p2)（排除极度模糊）

---

## 3. Gallery 展示适配性定义

### 3.1 四层质量体系

**EXCELLENT（可直接展示）：**
- Sharpness > 30.0（@640px）：细节清晰
- Colorfulness ≥ 20：色彩自然丰富
- Brightness 60-220：曝光正常
- 最长边 ≥ 800px：足够大
- 无明显压缩伪影或噪点

**GOOD（可展示但有轻微瑕疵）：**
- Sharpness 15-30：略微柔和但可接受
- Colorfulness 8-20：饱和度略低
- Brightness 40-60 或 220-240：轻微欠曝/过曝
- 最长边 500-800px

**FAIR（边缘 → 复核池）：**
- Sharpness 8-15：明显模糊但内容可辨
- Colorfulness 3-8：接近黑白
- Brightness < 30 或 > 240：严重曝光问题
- 最长边 300-500px
- 中等压缩伪影

**POOR（淘汰）：**
- Sharpness < max(2.0, dataset_p2)：无法接受
- 最长边 < 200px
- 亮度 < 20 或 > 250
- 严重压缩/损毁

### 3.2 最终适配性判定矩阵

|  | REAL | PROBABLY_REAL | AMBIGUOUS | PROBABLY_NON_REAL | NON_REAL |
|--|------|---------------|-----------|-------------------|----------|
| **EXCELLENT** | ✅ GALLERY | ✅ GALLERY | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **GOOD** | ✅ GALLERY | ✅ GALLERY | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **FAIR** | ⚠️ REVIEW | ⚠️ REVIEW | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT |
| **POOR** | ⚠️ REVIEW | ❌ REJECT | ❌ REJECT | ❌ REJECT | ❌ REJECT |

### 3.3 Gallery 展示的额外偏好规则（排序不淘汰）
- **场景优先**：风景/建筑/室内环境 > 单人肖像
- **群体优先**：多人场景 > 单人
- **自然光照优先**：brightness_std > 30 > 平光
- **多样性优先**：同一 dhash 簇只保留质量最高的

---

## 4. 边界样本处理规则（每个都有明确的可审计依据）

### B1: 高质感 3D 渲染（如 architectural viz）
- **视觉证据**：完美 sharpness、丰富色彩、自然曝光 — 但缺乏传感器噪点、边缘过度完美
- **操作**：无法通过启发式单独区分。标记 `digital_domain` 为高风险。GALLERY_READY 候选自动加注 "CGI risk" 标志
- **为什么**：这是 CPU-only 启发式的已知盲区

### B2: 严重滤镜/HDR 照片
- **视觉证据**：colorfulness > 100（超出自然范围）AND edge_ratio < 0.10（过度平滑）
- **操作**：降级一级（EXCELLENT→GOOD→FAIR→REVIEW）
- **为什么**：colorfulness > 100 在实景中极为罕见（kpmg 截图可达 140，但实景很少超过 80）

### B3: AI 生成图像
- **视觉证据**：完美 sharpness、超现实光照、不可能几何、过度平滑纹理
- **操作**：无法可靠检测。所有指标 > 90th 百分位的图像标记为 "potential AI"
- **为什么**：这是启发式+小模型的公认盲区

### B4: 屏幕截图中的照片
- **视觉证据**：整体 edge_ratio 高（来自 UI 边框），内部有照片区域
- **操作**：edge_ratio > 0.30 → NON_REAL（整个图像是截图，不是照片）
- **为什么**：任务要求图像本身是照片，不是包含照片的图像

### B5: 文档/信息图扫描
- **视觉证据**：colorfulness < 12 AND edge_ratio > 0.20 → 白底文字/图表
- **操作**：NON_REAL
- **为什么**：文档边缘密度高但颜色单一，这是高确定性信号

### B6: 小尺寸缩略图/头像
- **视觉证据**：最短边 < 150px
- **操作**：NON_REAL（gallery 不适用）
- **为什么**：物理尺寸太小，放大后像素化

### B7: 近似重复（dhash 簇）
- **检测**：dhash（8×9→64bit），Hamming 距离 ≤ 4
- **操作**：每簇只保留质量最高的
- **为什么**：gallery 不能出现 5 张一模一样的不同角度照片

### B8: 单人肖像
- **视觉证据**：aspect_ratio 接近 1:1 或 3:4，中央区域高 sharpness
- **操作**：排序降权，最终 100 张中 ≤ 10 张
- **为什么**：企业官网 gallery 应优先展示场景和活动

### B9: 损毁/无法读取文件
- **视觉证据**：Pillow 打开报错
- **操作**：记录到 `read_errors.log`，不静默跳过
- **为什么**：任务明确要求日志记录

### B10: 低熵占位图
- **视觉证据**：entropy < 3.0，brightness_std < 10
- **操作**：NON_REAL
- **为什么**：没有有意义的视觉内容

---

## 5. 为什么不能只靠单一阈值：实证证明

### 5.1 Sharpness 的欺骗性

| 样本 | Sharpness | 人眼判断 | 单阈值(>8)判断 | 正确判断 |
|------|-----------|---------|--------------|---------|
| m_immobilier 室内照 | 3.0 | 实景 | ❌ 淘汰（误判） | ✅ REAL |
| kpmg 截图 | 8.1 | 非实景 | ✅ 通过（误判） | ✅ NON_REAL |
| maior_capital 外景 | 73.9 | 实景 | ✅ 通过 | ✅ REAL |
| digital_domain CGI | 49.9 | 非实景但像实景 | ✅ 通过（误判） | ⚠️ 需要人工 |

**结论**：Sharpness 既不能单独区分实景/非实景，也不能单独判断质量。室内照片可以很模糊（sharpness=3.0）却是实景；截图可以很清晰（sharpness=103.3）却是非实景。

### 5.2 Colorfulness 的欺骗性

| 样本 | Colorfulness | 人眼判断 | 单阈值(<5=非实景)判断 | 正确判断 |
|------|-------------|---------|---------------------|---------|
| m_immobilier 室内 | 5.9 | 实景（白墙） | ❌ 淘汰（误判） | ✅ REAL |
| kpmg 截图 | 116.4 | 非实景 | ✅ 通过（误判） | ✅ NON_REAL |
| kpmg 信息图 | 136.9 | 非实景 | ✅ 通过（误判） | ✅ NON_REAL |

**结论**：Colorfulness 单独使用时**反向误导** — 非实景内容往往比实景更鲜艳。正确的组合是 `edge_ratio > 0.35 AND colorfulness > 70`。

### 5.3 跨数据集分布漂移

| 指标 | m_immobilier（实景）p5 | kpmg（截图）p5 | maior_capital（实景）p5 |
|------|----------------------|----------------|----------------------|
| Sharpness | 3.0 | 2.7 | 22.9 |
| Colorfulness | 7.7 | 24.8 | 31.1 |

同一指标在不同数据集间的分布差异可达 **7×以上**。全局固定阈值必然导致某些数据集大量误判。

---

## 6. 保守偏置策略

### 6.1 以下情况必须进入复核池，不得自动入选
1. **AMBIGUOUS** 级别的所有图像（不论质量）
2. **PROBABLY_REAL + FAIR** 质量的图像
3. **REAL/EXCELLENT** 但在 `digital_domain` 数据集中（CGI 风险）
4. 所有指标的 90th 百分位以上的图像（可疑的"完美"）

### 6.2 以下情况直接淘汰
1. NON_REAL / PROBABLY_NON_REAL（任何质量）
2. POOR 质量（任何实景等级）
3. PROBABLY_REAL + POOR
4. 最短边 < 200px

### 6.3 期望产出量（基于 Stage 1 实证估算）

| 数据集 | 总量 | 期望实景通过率 | 期望 gallery top-100 |
|--------|------|--------------|-------------------|
| truro_school | 36,266 | ~90% | 100 ✅ 轻松 |
| m_immobilier | 5,000 | ~95% | 100 ✅ |
| maior_capital | 5,000 | ~85%（含平面图） | 100 ✅ |
| tara_guerard | 4,971 | ~80%（含旧照片） | 70-100 |
| boston_university | 2,722 | ~60% | 50-80 |
| roland_berger | 2,997 | ~40%（大量信息图） | 20-40 |
| digital_domain | 1,669 | ~30%（CGI 风险） | 10-30 + 全量人工 |
| tuv_rheinland | 1,465 | ~30%（大量图表） | 10-30 |
| ul_solutions | 1,515 | ~50% | 20-50 |
| thema-med | 273 | ~40% | 10-30 |
| kpmg_forensic | 80 | ~5% | 0-5 |

> **注意**：当数据集无法产出 100 张合格图片时，输出真实数量，不填充低质量替代品。这是保守偏置的直接体现。

---

## 7. 总结

**实景 vs 非实景的视觉判断核心**：不是 sharpness，不是 colorfulness，而是 **edge_ratio + colorfulness 的组合**。实景照片边缘密度低（0.05-0.25），颜色来自自然光照；非实景内容要么边缘密度高（截图/文档），要么颜色异常鲜艳（UI），要么两者兼有。

**Gallery 适配性的视觉判断核心**：三层筛选 — (1) 必须是实景（REAL或PROBABLY_REAL），(2) 必须达到 GOOD 以上质量，(3) 必须通过多样性去重和场景类型配额。

**不能只靠单一阈值的原因**：实证数据清晰显示了跨数据集分布漂移（sharpness p5 从 3.0 到 22.9）、误导性相关性（colorfulness 在非实景中更高）、以及多个指标间的互补性。正确方法是数据集自适应百分位阈值 + 多信号组合判断 + 明确的边界条件处理规则。
