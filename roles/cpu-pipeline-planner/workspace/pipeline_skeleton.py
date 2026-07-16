#!/usr/bin/env python3
"""
pipeline_skeleton.py — CPU-only 三级级联流水线骨架
Stage A: 启发式预筛 (零模型, 7个视觉信号 + per-dataset自适应阈值)
Stage B: MobileNetV3-Small 特征提取 + 5级实景/4级质量分类
Stage C: dhash 去重 + DBSCAN多样性聚类 + 反肖像偏见 + top-100排序

用法:
    python pipeline_skeleton.py --input C:\\pics --output output/
    python pipeline_skeleton.py --datasets truro_school  # 单数据集调试
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_SIDE_THRESHOLD = 64


# ===================== 数据结构 =====================

@dataclass
class ImageSignals:
    sharpness: float = 0.0
    edge_ratio: float = 0.0
    colorfulness: float = 0.0
    entropy: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    aspect_ratio: float = 1.0
    min_side: int = 0


@dataclass
class StageBResult:
    dataset: str = ""
    filepath: str = ""
    realism_label: str = "AMBIGUOUS"
    realism_conf: float = 0.0
    quality_label: str = "FAIR"
    quality_conf: float = 0.0
    features_576: List[float] = field(default_factory=list)


@dataclass
class StageCItem:
    dataset: str = ""
    filepath: str = ""
    final_score: float = 0.0
    cluster_id: int = -1
    has_face: bool = False
    realism_label: str = "AMBIGUOUS"
    quality_label: str = "FAIR"


# ===================== Stage A =====================

def compute_signals(img_path: str) -> Optional[ImageSignals]:
    """计算单张图片的 7 个视觉信号。返回 None 表示文件损坏。"""
    try:
        import cv2
        import numpy as np

        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sig = ImageSignals()
        sig.min_side = min(h, w)
        sig.aspect_ratio = w / h if h > 0 else 1.0
        sig.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        edges = cv2.Canny(gray, 50, 150)
        sig.edge_ratio = float(np.count_nonzero(edges) / (h * w))

        b, g, r = cv2.split(img.astype(np.float32))
        rg, yb = r - g, 0.5 * (r + g) - b
        sig.colorfulness = float(np.sqrt(rg.var()**2 + yb.var()**2) / 0.3)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0]
        hn = hist / hist.sum()
        sig.entropy = float(-np.sum(hn * np.log2(hn)))
        sig.brightness_mean = float(gray.mean())
        sig.brightness_std = float(gray.std())
        return sig
    except Exception as e:
        logging.warning(f"信号计算失败 [{img_path}]: {e}")
        return None


def calibrate_thresholds(all_signals: Dict[str, List[ImageSignals]]) -> Dict[str, dict]:
    """Per-dataset 自适应阈值校准。"""
    import numpy as np

    flat = [s for sigs in all_signals.values() for s in sigs]
    global_p25_sharp = np.percentile([s.sharpness for s in flat], 25) if flat else 100.0
    thresholds = {}

    for ds, sigs in all_signals.items():
        if not sigs:
            thresholds[ds] = {"reject_all": True}
            continue
        p25_sharp = float(np.percentile([s.sharpness for s in sigs], 25))
        p25_edge = float(np.percentile([s.edge_ratio for s in sigs], 25))
        p25_color = float(np.percentile([s.colorfulness for s in sigs], 25))
        p25_ent = float(np.percentile([s.entropy for s in sigs], 25))
        mult = 0.7 if p25_sharp < global_p25_sharp * 0.5 else 1.0
        thresholds[ds] = {
            "sharpness_th": max(5.0, p25_sharp * 0.6 * mult),
            "edge_ratio_th": max(0.005, p25_edge * 0.5 * mult),
            "colorfulness_th": max(5.0, p25_color * 0.5 * mult),
            "entropy_th": max(2.0, p25_ent * 0.7 * mult),
            "min_side_th": MIN_SIDE_THRESHOLD,
        }
    return thresholds


def stage_a_run(
    dataset_files: Dict[str, List[str]], num_workers: int = 8
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, dict]]:
    """Stage A 主入口: 信号计算 → 阈值校准 → 过滤。"""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logging.info(f"=== Stage A: {sum(len(v) for v in dataset_files.values())} 张 ===")

    all_results: List[tuple] = []  # (dataset, filepath, signals_or_None)
    bad_files: List[tuple] = []    # (dataset, filepath, error)

    def worker(ds: str, fp: str):
        sig = compute_signals(fp)
        if sig is None:
            return (ds, fp, None, "unreadable_or_corrupt")
        return (ds, fp, sig, "")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        fut_map = {pool.submit(worker, ds, fp): (ds, fp)
                   for ds, files in dataset_files.items() for fp in files}
        for fut in as_completed(fut_map):
            ds, fp = fut_map[fut]
            try:
                ds2, fp2, sig, err = fut.result()
                if sig is None:
                    bad_files.append((ds2, fp2, err))
                else:
                    all_results.append((ds2, fp2, sig))
            except Exception as e:
                bad_files.append((ds, fp, str(e)))

    if bad_files:
        Path("output/logs").mkdir(parents=True, exist_ok=True)
        with open("output/logs/stageA_bad_files.txt", "w", encoding="utf-8") as f:
            for ds, fp, err in bad_files:
                f.write(f"[{ds}] {fp} -> {err}\n")
        logging.warning(f"损坏/不可读文件 {len(bad_files)} 个, 已记录")

    per_ds_sigs: Dict[str, List[ImageSignals]] = defaultdict(list)
    for ds, fp, sig in all_results:
        per_ds_sigs[ds].append(sig)

    thresholds = calibrate_thresholds(dict(per_ds_sigs))

    survivors: Dict[str, List[str]] = defaultdict(list)
    rejected: Dict[str, List[str]] = defaultdict(list)

    for ds, fp, sig in all_results:
        th = thresholds.get(ds, {})
        reasons = []
        if sig.min_side < th.get("min_side_th", 64):
            reasons.append(f"min_side={sig.min_side}<{th.get('min_side_th')}")
        if not (0.1 <= sig.aspect_ratio <= 10.0):
            reasons.append(f"aspect={sig.aspect_ratio:.2f}")
        if sig.sharpness < th.get("sharpness_th", 5.0):
            reasons.append(f"sharp={sig.sharpness:.1f}<{th.get('sharpness_th'):.1f}")
        if sig.edge_ratio < th.get("edge_ratio_th", 0.005):
            reasons.append(f"edge={sig.edge_ratio:.4f}<{th.get('edge_ratio_th'):.4f}")
        if sig.colorfulness < th.get("colorfulness_th", 5.0):
            reasons.append(f"color={sig.colorfulness:.1f}<{th.get('colorfulness_th'):.1f}")
        if sig.entropy < th.get("entropy_th", 2.0):
            reasons.append(f"entropy={sig.entropy:.2f}<{th.get('entropy_th'):.2f}")

        if reasons:
            rejected[ds].append(f"{fp}\t{'; '.join(reasons)}")
        else:
            survivors[ds].append(fp)

    for ds in dataset_files:
        tot = len(dataset_files[ds])
        sv = len(survivors.get(ds, []))
        logging.info(f"  [{ds}] {sv}/{tot} 通过 ({sv/max(tot,1)*100:.1f}%)")

    return survivors, rejected, thresholds


# ===================== Stage B =====================

def _heuristic_classify(sig: ImageSignals) -> tuple:
    """纯启发式: (realism_label, quality_label, realism_score, quality_score)"""
    s_n = min(sig.sharpness / 500.0, 1.0)
    e_n = min(sig.edge_ratio / 0.5, 1.0)
    c_n = min(sig.colorfulness / 80.0, 1.0)
    ent_n = min(sig.entropy / 7.5, 1.0)
    r_score = 0.3 * e_n + 0.3 * c_n + 0.2 * ent_n + 0.2 * s_n
    q_score = 0.25 * s_n + 0.25 * c_n + 0.25 * ent_n + 0.25 * e_n

    if r_score < 0.3:
        rl = "NON_REAL"
    elif r_score < 0.45:
        rl = "PROBABLY_NON_REAL"
    elif r_score < 0.6:
        rl = "AMBIGUOUS"
    elif r_score < 0.8:
        rl = "PROBABLY_REAL"
    else:
        rl = "REAL"

    if q_score < 0.25:
        ql = "POOR"
    elif q_score < 0.5:
        ql = "FAIR"
    elif q_score < 0.75:
        ql = "GOOD"
    else:
        ql = "EXCELLENT"

    return rl, ql, r_score, q_score


def stage_b_run(
    survivors: Dict[str, List[str]],
    survivors_signals: Optional[Dict[str, List[ImageSignals]]] = None,
    model_path: Optional[str] = None,
) -> Dict[str, List[StageBResult]]:
    """Stage B: 特征提取 + 分类。支持三种模式。"""
    logging.info(f"=== Stage B: {sum(len(v) for v in survivors.values())} 张 ===")

    results: Dict[str, List[StageBResult]] = defaultdict(list)
    use_model = model_path and os.path.exists(model_path)

    for ds, files in survivors.items():
        for i, fp in enumerate(files):
            br = StageBResult(dataset=ds, filepath=fp)

            if use_model:
                # 模式 A: ONNX 推理 (占位, 实际需调用 session.run)
                br.features_576 = [0.0] * 576
                br.realism_label = "AMBIGUOUS"
                br.realism_conf = 0.5
                br.quality_label = "FAIR"
                br.quality_conf = 0.5
            elif survivors_signals and ds in survivors_signals and i < len(survivors_signals[ds]):
                # 模式 B: 启发式评分
                sig = survivors_signals[ds][i]
                rl, ql, rc, qc = _heuristic_classify(sig)
                br.realism_label = rl
                br.realism_conf = rc
                br.quality_label = ql
                br.quality_conf = qc
                # 模拟特征
                br.features_576 = [sig.sharpness / 500.0, sig.edge_ratio / 0.5,
                                   sig.colorfulness / 80.0, sig.entropy / 7.5] * 144
            else:
                br.realism_label = "AMBIGUOUS"
                br.quality_label = "FAIR"

            results[ds].append(br)

    # 统计
    r_cnt, q_cnt = defaultdict(int), defaultdict(int)
    for rs in results.values():
        for r in rs:
            r_cnt[r.realism_label] += 1
            q_cnt[r.quality_label] += 1
    logging.info(f"Stage B 完成. 实景: {dict(r_cnt)}, 质量: {dict(q_cnt)}")
    return dict(results)


# ===================== Stage C =====================

def dhash(img_path: str, hash_size: int = 8) -> int:
    """64-bit dhash."""
    import cv2
    import numpy as np
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for i in range(hash_size):
        for j in range(hash_size):
            h |= int(diff[i, j]) << (i * hash_size + j)
    return h


def hamming_dist(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count() if hasattr(int, "bit_count") else bin(h1 ^ h2).count("1")


def dbscan_cluster(features_list: List[List[float]], eps: float = 0.5, min_samples: int = 3) -> List[int]:
    """DBSCAN 聚类, 回退到单簇。"""
    import numpy as np
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        X = np.array(features_list, dtype=np.float32)
        if X.shape[1] > 32:
            n = min(32, X.shape[0], X.shape[1])
            X = PCA(n_components=n).fit_transform(X)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1).fit_predict(X)
        return [int(l) for l in labels]
    except ImportError:
        return [0] * len(features_list)


def detect_face(img_path: str) -> bool:
    """Haar Cascade 人脸检测 (CPU-only)。"""
    try:
        import cv2
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return False
        return len(cc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))) > 0
    except Exception:
        return False


def stage_c_run(stage_b_results: Dict[str, List[StageBResult]], top_k: int = 100) -> Dict[str, List[StageCItem]]:
    """Stage C: dhash 去重 → DBSCAN 聚类 → 人脸检测 → 配额排序 → top-100。"""
    logging.info(f"=== Stage C: {sum(len(v) for v in stage_b_results.values())} 张 ===")
    output: Dict[str, List[StageCItem]] = {}

    for dataset, items in stage_b_results.items():
        # 1. dhash 去重
        hash_best: Dict[int, StageBResult] = {}
        for item in items:
            h = dhash(item.filepath)
            if h == 0:
                continue
            dup_key = None
            for ek in hash_best:
                if hamming_dist(h, ek) < 8:
                    dup_key = ek
                    break
            if dup_key is None:
                hash_best[h] = item
            elif item.quality_conf > hash_best[dup_key].quality_conf:
                hash_best[dup_key] = item

        deduped = list(hash_best.values())
        logging.info(f"  [{dataset}] 去重: {len(items)} -> {len(deduped)}")

        # 2. 特征聚类
        feats = [it.features_576[:576] for it in deduped if it.features_576]
        if feats:
            max_d = max(len(f) for f in feats)
            feats_pad = [f + [0.0] * (max_d - len(f)) for f in feats]
        else:
            feats_pad = []
        cids = dbscan_cluster(feats_pad) if feats_pad else []

        # 3. final score
        rw = {"NON_REAL": 0.1, "PROBABLY_NON_REAL": 0.3, "AMBIGUOUS": 0.5,
              "PROBABLY_REAL": 0.8, "REAL": 1.0}
        qw = {"POOR": 0.1, "FAIR": 0.4, "GOOD": 0.7, "EXCELLENT": 1.0}
        pool: List[StageCItem] = []
        for idx, item in enumerate(deduped):
            score = rw.get(item.realism_label, 0.5) * qw.get(item.quality_label, 0.4)
            has_face = detect_face(item.filepath)
            if has_face:
                score *= 0.9
            pool.append(StageCItem(
                dataset=dataset, filepath=item.filepath,
                final_score=score, cluster_id=cids[idx] if idx < len(cids) else -1,
                has_face=has_face, realism_label=item.realism_label,
                quality_label=item.quality_label,
            ))

        # 4. 配额排序 (每簇 ≤15)
        pool.sort(key=lambda x: x.final_score, reverse=True)
        cc: Dict[int, int] = {}
        selected: List[StageCItem] = []
        for it in pool:
            if cc.get(it.cluster_id, 0) >= 15:
                continue
            selected.append(it)
            cc[it.cluster_id] = cc.get(it.cluster_id, 0) + 1
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for it in pool:
                if it not in selected:
                    selected.append(it)
                    if len(selected) >= top_k:
                        break

        noise = sum(1 for x in selected[:top_k] if x.cluster_id == -1)
        clusters = len(set(x.cluster_id for x in selected[:top_k]))
        output[dataset] = selected[:top_k]
        logging.info(f"  [{dataset}] top-{top_k}: {clusters} 个簇, {noise} 噪声点")

    return output


# ===================== 主入口 =====================

def run_pipeline(
    input_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    skip_stage_b: bool = False,
    model_path: Optional[str] = None,
    num_workers: int = 8,
):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_dir / "pipeline_run.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"流水线启动: input={input_dir}, output={output_dir}")
    start = time.time()

    # ---- 扫描文件 ----
    dataset_files: Dict[str, List[str]] = defaultdict(list)
    inp = Path(input_dir)
    for d in inp.iterdir():
        if d.is_dir():
            dn = d.name
            if datasets and dn not in datasets:
                continue
            for ext in SUPPORTED_EXTENSIONS:
                for f in d.rglob(f"*{ext}"):
                    dataset_files[dn].append(str(f))
            logging.info(f"  发现 [{dn}]: {len(dataset_files[dn])} 张")

    total = sum(len(v) for v in dataset_files.values())
    logging.info(f"总计: {total} 张, {len(dataset_files)} 个数据集")
    if total == 0:
        logging.error("未找到图片, 退出")
        sys.exit(1)

    # ---- Stage A ----
    survivors, rejected, thresholds = stage_a_run(dataset_files, num_workers)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "stageA_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "survivors": dict(survivors),
            "rejected_count": {ds: len(v) for ds, v in rejected.items()},
            "thresholds": {ds: {k: float(v) if not isinstance(v, bool) else v
                                for k, v in th.items()} for ds, th in thresholds.items()},
        }, f, ensure_ascii=False, indent=2)
    logging.info(f"Stage A 结果 -> {output_dir}/stageA_results.json")

    # ---- Stage B ----
    if skip_stage_b:
        logging.warning("跳过 Stage B")
        stage_b_results = {
            ds: [StageBResult(dataset=ds, filepath=fp) for fp in files]
            for ds, files in survivors.items()
        }
    else:
        stage_b_results = stage_b_run(survivors, model_path=model_path)

    with open(Path(output_dir) / "stageB_results.json", "w", encoding="utf-8") as f:
        json.dump({
            ds: [{"filepath": r.filepath, "realism": r.realism_label,
                  "realism_conf": r.realism_conf, "quality": r.quality_label,
                  "quality_conf": r.quality_conf} for r in rs]
            for ds, rs in stage_b_results.items()
        }, f, ensure_ascii=False, indent=2)

    # ---- Stage C ----
    top100 = stage_c_run(stage_b_results)
    per_ds_dir = Path(output_dir) / "per_dataset"
    per_ds_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {}

    for dataset, items in top100.items():
        ds_dir = per_ds_dir / dataset
        ds_dir.mkdir(exist_ok=True)
        with open(ds_dir / "top100_list.txt", "w", encoding="utf-8") as f:
            f.write("rank\tfilepath\tfinal_score\trealism\tquality\tcluster_id\thas_face\n")
            for rank, it in enumerate(items, 1):
                f.write(f"{rank}\t{it.filepath}\t{it.final_score:.4f}\t"
                        f"{it.realism_label}\t{it.quality_label}\t"
                        f"{it.cluster_id}\t{it.has_face}\n")

        review = [
            r for rs in stage_b_results.get(dataset, [])
            for r in [rs] if r.realism_label == "AMBIGUOUS" or r.quality_label in ("POOR", "FAIR")
        ]
        with open(ds_dir / "review_pool_list.txt", "w", encoding="utf-8") as f:
            for r in review[:200]:
                f.write(f"{r.filepath}\t{r.realism_label}\t{r.realism_conf:.2f}\t"
                        f"{r.quality_label}\t{r.quality_conf:.2f}\n")

        aggregate[dataset] = {
            "total_input": len(dataset_files.get(dataset, [])),
            "stage_a_survived": len(survivors.get(dataset, [])),
            "stage_b_processed": len(stage_b_results.get(dataset, [])),
            "top100_delivered": len(items),
            "unique_clusters": len(set(it.cluster_id for it in items)),
        }

    elapsed = time.time() - start
    aggregate["_meta"] = {
        "total_input_images": total,
        "total_datasets": len(dataset_files),
        "pipeline_elapsed_minutes": round(elapsed / 60, 1),
        "skip_stage_b": skip_stage_b,
    }
    with open(Path(output_dir) / "aggregate_stats.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    state = {"status": "completed", "completed_stages": ["A", "B", "C"],
             "output_dir": output_dir, "elapsed_seconds": round(elapsed, 1)}
    with open(Path(output_dir) / "pipeline_state.json", "w") as f:
        json.dump(state, f, indent=2)

    logging.info(f"=== 完成! {elapsed/60:.1f} 分钟 ===")
    logging.info(f"输出: {output_dir}/per_dataset/<ds>/top100_list.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only 三级流水线")
    parser.add_argument("--input", default=r"C:\pics")
    parser.add_argument("--output", default="workspace/output")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input, output_dir=args.output,
        datasets=args.datasets, skip_stage_b=args.skip_stage_b,
        model_path=args.model, num_workers=args.workers,
    )
