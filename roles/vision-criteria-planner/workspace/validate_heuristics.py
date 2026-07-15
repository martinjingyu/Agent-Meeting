"""
Small-scale validation script for heuristic thresholds.
Tests a handful of images from each dataset to verify that
our proposed thresholds (sharpness, colorfulness, aspect ratio)
make sense before committing to them in the full pipeline.
"""
import os, sys, json
from PIL import Image
import numpy as np

PICS_ROOT = r"C:\pics"
DATASETS = [
    "boston_university", "digital_domain", "kpmg_forensic",
    "m_immobilier", "maior_capital", "roland_berger",
    "tara_guerard", "thema-med", "truro_school",
    "tuv_rheinland", "ul_solutions"
]

def sobel_sharpness(img_gray):
    """Mean Sobel gradient magnitude as sharpness proxy."""
    from scipy.ndimage import sobel
    sx = sobel(img_gray, axis=1)
    sy = sobel(img_gray, axis=0)
    mag = np.sqrt(sx.astype(np.float64)**2 + sy.astype(np.float64)**2)
    return float(np.mean(mag))

def colorfulness(img_rgb):
    """Hasler & Susstrunk colorfulness metric."""
    R, G, B = img_rgb[:,:,0].astype(np.float64), img_rgb[:,:,1].astype(np.float64), img_rgb[:,:,2].astype(np.float64)
    rg = R - G
    yb = 0.5*(R+G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2))

def brightness(img_rgb):
    return float(np.mean(img_rgb))

def aspect_ratio(w, h):
    return w/h if h > 0 else 1.0

def dhash(img_gray, hash_size=8):
    """Difference hash: returns 64-bit integer."""
    from PIL.Image import Resampling
    img = Image.fromarray(img_gray).resize((hash_size+1, hash_size), Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    return sum((1 << i) for i, b in enumerate(bits[:64]) if b)

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

results = {}

for ds_name in DATASETS:
    ds_path = os.path.join(PICS_ROOT, ds_name)
    if not os.path.isdir(ds_path):
        print(f"SKIP {ds_name}: path not found")
        continue
    
    # Get image files
    exts = {'.jpg','.jpeg','.png','.webp','.bmp','.gif'}
    all_files = [f for f in os.listdir(ds_path) 
                 if os.path.isfile(os.path.join(ds_path, f))
                 and os.path.splitext(f)[1].lower() in exts]
    
    # Take up to 20 samples
    np.random.seed(42)
    if len(all_files) > 20:
        samples = list(np.random.choice(all_files, 20, replace=False))
    else:
        samples = all_files[:]
    
    ds_samples = []
    for fname in samples:
        fpath = os.path.join(ds_path, fname)
        try:
            with Image.open(fpath) as img:
                img.load()
                w, h = img.size
                # Convert to RGB
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGBA').convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_rgb = np.array(img, dtype=np.float64)
                img_gray = np.mean(img_rgb, axis=2)
                
                # Compute metrics
                s = sobel_sharpness(img_gray)
                c = colorfulness(img_rgb)
                b = brightness(img_rgb)
                ar = aspect_ratio(w, h)
                dh = dhash(img_gray)
                
                ds_samples.append({
                    "file": fname,
                    "size": f"{w}x{h}",
                    "sharpness": round(s, 1),
                    "colorfulness": round(c, 1),
                    "brightness": round(b, 1),
                    "aspect_ratio": round(ar, 3),
                    "dhash": dh
                })
        except Exception as e:
            ds_samples.append({"file": fname, "error": str(e)})
    
    results[ds_name] = {
        "total_files": len(all_files),
        "sampled": len(ds_samples),
        "samples": ds_samples
    }
    print(f"{ds_name}: sampled {len(ds_samples)}/{len(all_files)}")

# Output summary
print("\n=== SUMMARY BY DATASET ===")
for ds, data in results.items():
    vals = [s for s in data["samples"] if "sharpness" in s]
    if not vals:
        print(f"{ds}: no valid samples")
        continue
    sharp = [v["sharpness"] for v in vals]
    color = [v["colorfulness"] for v in vals]
    bright = [v["brightness"] for v in vals]
    print(f"{ds}: sharpness=[{np.min(sharp):.1f}, {np.median(sharp):.1f}, {np.max(sharp):.1f}] "
          f"colorfulness=[{np.min(color):.1f}, {np.median(color):.1f}, {np.max(color):.1f}] "
          f"brightness=[{np.min(bright):.1f}, {np.median(bright):.1f}, {np.max(bright):.1f}]")

# Save full results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heuristic_validation.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nFull results saved to {out_path}")
