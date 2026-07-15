"""Calibrate sharpness thresholds properly using real data samples."""
from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = Path("C:/pics")
DS_NAMES = [
    "boston_university", "digital_domain", "kpmg_forensic",
    "m_immobilier", "maior_capital", "roland_berger",
    "tara_guerard", "thema-med", "truro_school",
    "tuv_rheinland", "ul_solutions"
]

def sobel_sharpness(arr):
    gray = np.mean(arr.astype(np.float32), axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0,0),(0,1)), mode='edge')
    gy = np.pad(gy, ((0,1),(0,0)), mode='edge')
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def load_rgb(path, max_px=640):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_px / max(w, h) if max(w, h) > max_px else 1.0
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.uint8), img.size

print("=" * 70)
print("SHARPNESS THRESHOLD CALIBRATION (640px scale)")
print("=" * 70)

for ds in DS_NAMES:
    ds_path = DATA_ROOT / ds
    files = sorted(ds_path.glob("*.jpg"))[:20]
    if not files:
        files = sorted(ds_path.glob("*.png"))[:20]
    
    sharp_vals = []
    for f in files:
        try:
            arr, sz = load_rgb(str(f), max_px=640)
            sharp_vals.append(sobel_sharpness(arr))
        except:
            pass
    
    if sharp_vals:
        p5 = np.percentile(sharp_vals, 5)
        p10 = np.percentile(sharp_vals, 10)
        p25 = np.percentile(sharp_vals, 25)
        p50 = np.percentile(sharp_vals, 50)
        print(f"\n{ds} ({len(sharp_vals)} samples):")
        print(f"  Sharpness: min={min(sharp_vals):.1f}  p5={p5:.1f}  p10={p10:.1f}  p25={p25:.1f}  p50={p50:.1f}  max={max(sharp_vals):.1f}")
        print(f"  If threshold=5: rejected={sum(1 for s in sharp_vals if s < 5)}/{len(sharp_vals)}")
        print(f"  If threshold=3: rejected={sum(1 for s in sharp_vals if s < 3)}/{len(sharp_vals)}")
        print(f"  If threshold=2: rejected={sum(1 for s in sharp_vals if s < 2)}/{len(sharp_vals)}")
