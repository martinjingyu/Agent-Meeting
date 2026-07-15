"""
Validate sharpness at different scales to calibrate thresholds.
Also test MobileCLIP zero-shot classification on a few samples.
"""
import os, sys, warnings, time
from pathlib import Path
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

DATA_ROOT = Path("C:/pics")
DS_NAMES = [
    "boston_university", "digital_domain", "kpmg_forensic",
    "m_immobilier", "maior_capital", "roland_berger",
    "tara_guerard", "thema-med", "truro_school",
    "tuv_rheinland", "ul_solutions"
]

def load_rgb(path, max_px=None):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max_px:
        scale = max_px / max(w, h) if max(w, h) > max_px else 1.0
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.uint8), img.size

def sobel_sharpness(arr):
    gray = np.mean(arr.astype(np.float32), axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0,0),(0,1)), mode='edge')
    gy = np.pad(gy, ((0,1),(0,0)), mode='edge')
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

def colorfulness(arr):
    r, g, b = arr[:,:,0].astype(np.float32), arr[:,:,1].astype(np.float32), arr[:,:,2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def entropy(arr):
    """Image entropy as a measure of complexity (higher = more detail)."""
    gray = np.mean(arr.astype(np.float32), axis=2).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

# Test sharpness at multiple scales
print("=" * 70)
print("SHARPNESS SCALE CALIBRATION")
print("=" * 70)

test_files = []
for ds in DS_NAMES[:4]:  # First 4 datasets
    ds_path = DATA_ROOT / ds
    files = sorted(ds_path.glob("*.jpg"))[:3]
    test_files.extend([(ds, f) for f in files])

for ds, fpath in test_files:
    # Full res
    arr_full, size_full = load_rgb(str(fpath), max_px=None)
    # Downscaled
    arr_640, size_640 = load_rgb(str(fpath), max_px=640)
    arr_1024, size_1024 = load_rgb(str(fpath), max_px=1024)
    
    sharp_full = sobel_sharpness(arr_full)
    sharp_640 = sobel_sharpness(arr_640)
    sharp_1024 = sobel_sharpness(arr_1024)
    col = colorfulness(arr_1024)
    ent = entropy(arr_1024)
    
    print(f"\n{ds}/{fpath.name}")
    print(f"  Original size: {size_full[0]}x{size_full[1]}")
    print(f"  Sharpness @full: {sharp_full:.1f}  @1024: {sharp_1024:.1f}  @640: {sharp_640:.1f}")
    print(f"  Colorfulness: {col:.1f}  Entropy: {ent:.2f}")

# Now try MobileCLIP on 2-3 samples to see if it works
print("\n" + "=" * 70)
print("MobileCLIP ZERO-SHOT TEST (on 3 samples)")
print("=" * 70)

try:
    import torch
    import open_clip
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Try MobileCLIP-S0
    model_name = "MobileCLIP-S0"
    pretrained = "datacompdr_lt_256"  # Common pretrained tag for S0
    
    print(f"Loading {model_name}...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"  Model loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in model.parameters()):,}")
    assert sum(p.numel() for p in model.parameters()) < 100_000_000, "Model exceeds 100M params!"
    
    # Test prompts for real-scene vs non-real
    real_prompts = [
        "a real photograph of an outdoor landscape",
        "a real photograph of a building or cityscape",
        "a real photograph of an indoor room or interior space",
        "a real photograph of people in a group activity",
        "a real photograph of a natural scene",
    ]
    non_real_prompts = [
        "a computer generated 3D render",
        "a screenshot of a website or document",
        "a graphic design or illustration",
        "a chart, diagram or infographic",
        "a drawing or painting",
    ]
    
    # Pick 3 samples from different datasets
    test_samples = []
    for ds in ["m_immobilier", "truro_school", "roland_berger"]:
        ds_path = DATA_ROOT / ds
        files = sorted(ds_path.glob("*.jpg"))[:2]
        test_samples.extend([(ds, f) for f in files])
    
    with torch.no_grad():
        real_tokens = tokenizer(real_prompts)
        non_real_tokens = tokenizer(non_real_prompts)
        real_features = model.encode_text(real_tokens)
        real_features = real_features / real_features.norm(dim=-1, keepdim=True)
        non_real_features = model.encode_text(non_real_tokens)
        non_real_features = non_real_features / non_real_features.norm(dim=-1, keepdim=True)
        # Average prompt embeddings
        real_avg = real_features.mean(dim=0, keepdim=True)
        real_avg = real_avg / real_avg.norm(dim=-1, keepdim=True)
        non_real_avg = non_real_features.mean(dim=0, keepdim=True)
        non_real_avg = non_real_avg / non_real_avg.norm(dim=-1, keepdim=True)
    
    for ds, fpath in test_samples:
        img = Image.open(str(fpath)).convert("RGB")
        img_input = preprocess(img).unsqueeze(0)
        
        t1 = time.time()
        with torch.no_grad():
            image_features = model.encode_image(img_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Zero-shot classification
        real_sim = (image_features @ real_avg.T).item()
        non_real_sim = (image_features @ non_real_avg.T).item()
        softmax_real = np.exp(real_sim) / (np.exp(real_sim) + np.exp(non_real_sim))
        
        print(f"\n{ds}/{fpath.name}  (inference: {time.time()-t1:.2f}s)")
        print(f"  Real similarity: {real_sim:.4f}")
        print(f"  Non-real similarity: {non_real_sim:.4f}")
        print(f"  P(real) ≈ {softmax_real:.3f}")
        
        # Also check per-prompt
        with torch.no_grad():
            all_text_feat = torch.cat([real_features, non_real_features], dim=0)
            all_text_feat = all_text_feat / all_text_feat.norm(dim=-1, keepdim=True)
        sims = (image_features @ all_text_feat.T).squeeze(0)
        print(f"  Per-prompt similarities:")
        for i, p in enumerate(real_prompts + non_real_prompts):
            print(f"    [{i}] {p}: {sims[i].item():.4f}")

except ImportError as e:
    print(f"ImportError: {e}")
    print("Model test skipped.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
