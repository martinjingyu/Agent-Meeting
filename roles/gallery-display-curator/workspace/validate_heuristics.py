"""
Small-scale validation script: test heuristic thresholds and MobileCLIP zero-shot
on a handful of sample images from each dataset to verify design decisions.

This is NOT the full pipeline — just sanity-checking assumptions.
"""

import os, sys, json, time, warnings
from pathlib import Path
from PIL import Image, ImageFilter, ImageStat
import numpy as np

warnings.filterwarnings("ignore")

DATA_ROOT = Path("C:/pics")
DS_NAMES = [
    "boston_university", "digital_domain", "kpmg_forensic",
    "m_immobilier", "maior_capital", "roland_berger",
    "tara_guerard", "thema-med", "truro_school",
    "tuv_rheinland", "ul_solutions"
]

# ── heuristic helpers ────────────────────────────────────────────────

def load_rgb(path, max_px=1024):
    """Load image as RGB numpy array, optionally downsizing to max_px on longest side."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_px / max(w, h) if max(w, h) > max_px else 1.0
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.uint8), img.size

def sobel_sharpness(arr):
    """Mean Sobel gradient magnitude as sharpness proxy."""
    gray = np.mean(arr.astype(np.float32), axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0,0),(0,1)), mode='edge')
    gy = np.pad(gy, ((0,1),(0,0)), mode='edge')
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def brightness(arr):
    return float(np.mean(arr))

def colorfulness(arr):
    """Hasler & Susstrunk colorfulness metric."""
    r, g, b = arr[:,:,0].astype(np.float32), arr[:,:,1].astype(np.float32), arr[:,:,2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def aspect_ratio(img):
    w, h = img.size
    return w / h if h > 0 else 0

def dhash(img, hash_size=8):
    """Difference hash (64-bit)."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum(2**i for i, b in enumerate(diff.flatten()))

def hamming_dist(h1, h2):
    return bin(h1 ^ h2).count("1")

# ── Sample validation ───────────────────────────────────────────────

print("=" * 70)
print("HEURISTIC VALIDATION ON SAMPLE IMAGES")
print("=" * 70)

results = {}

for ds in DS_NAMES:
    ds_path = DATA_ROOT / ds
    if not ds_path.is_dir():
        print(f"\n[SKIP] {ds} — not found")
        continue
    
    # Get up to 10 random images
    all_files = []
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]:
        all_files.extend(list(ds_path.glob(f"*{ext}")))
        all_files.extend(list(ds_path.glob(f"*{ext.upper()}")))
    
    np.random.shuffle(all_files)
    samples = all_files[:10]
    
    ds_results = []
    for fpath in samples:
        try:
            arr, pil_size = load_rgb(str(fpath), max_px=1024)
            sharp = sobel_sharpness(arr)
            bright = brightness(arr)
            color = colorfulness(arr)
            ar = aspect_ratio(Image.open(str(fpath)))
            h = dhash(Image.open(str(fpath)))
            ds_results.append({
                "file": fpath.name,
                "size": f"{pil_size[0]}x{pil_size[1]}",
                "sharpness": round(sharp, 2),
                "brightness": round(bright, 2),
                "colorfulness": round(color, 2),
                "aspect_ratio": round(ar, 3),
                "hash": h
            })
        except Exception as e:
            print(f"  ERROR {fpath.name}: {e}")
    
    results[ds] = ds_results
    
    # Summary stats
    if ds_results:
        sharp_vals = [r["sharpness"] for r in ds_results]
        color_vals = [r["colorfulness"] for r in ds_results]
        print(f"\n{ds} ({len(ds_results)} samples):")
        print(f"  Sharpness:   mean={np.mean(sharp_vals):.1f}, min={np.min(sharp_vals):.1f}, max={np.max(sharp_vals):.1f}")
        print(f"  Colorfulness: mean={np.mean(color_vals):.1f}, min={np.min(color_vals):.1f}, max={np.max(color_vals):.1f}")
        
        # Count how many would pass/fail heuristic filters
        blurry = sum(1 for r in ds_results if r["sharpness"] < 20)
        low_color = sum(1 for r in ds_results if r["colorfulness"] < 5)
        extreme_ar = sum(1 for r in ds_results if r["aspect_ratio"] < 0.3 or r["aspect_ratio"] > 3.5)
        print(f"  Would be rejected as blurry (sharp<20): {blurry}/{len(ds_results)}")
        print(f"  Would be rejected as low-color (col<5): {low_color}/{len(ds_results)}")
        print(f"  Would be rejected as extreme AR: {extreme_ar}/{len(ds_results)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
