"""
Small-scale validation: test MobileCLIP-S0 on CPU with few sample images.
Measures real inference time and checks classification quality.
"""
import os, sys, time, json
import numpy as np
from PIL import Image

PICS_ROOT = r"C:\pics"

# Take 5 images from a few different datasets for testing
test_images = []
for ds in ["m_immobilier", "roland_berger", "truro_school", "tuv_rheinland", "kpmg_forensic"]:
    ds_path = os.path.join(PICS_ROOT, ds)
    if not os.path.isdir(ds_path):
        continue
    exts = {'.jpg','.jpeg','.png'}
    files = [f for f in os.listdir(ds_path) 
             if os.path.isfile(os.path.join(ds_path, f))
             and os.path.splitext(f)[1].lower() in exts]
    if files:
        # take first 2
        for f in files[:2]:
            test_images.append(os.path.join(ds_path, f))

print(f"Testing MobileCLIP on {len(test_images)} images...")

try:
    import open_clip
    print("open_clip imported OK")
except ImportError:
    print("open_clip not installed")
    sys.exit(1)

# Load MobileCLIP-S0 (smallest variant)
print("Loading MobileCLIP-S0 model...")
t0 = time.time()
model_name = "MobileCLIP-S0"
# open_clip expects: model, _, preprocess = open_clip.create_model_and_transforms(...)
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, 
    pretrained='datacompdr-gpu',  # typical small variant
    device='cpu'
)
tokenizer = open_clip.get_tokenizer(model_name)
load_time = time.time() - t0
print(f"Model load time: {load_time:.2f}s")

# Define prompts for real-photo vs non-photo classification
photo_prompts = [
    "a photograph taken with a camera of a real outdoor landscape",
    "a photograph taken with a camera of a real building or architecture",
    "a photograph taken with a camera of real people in a real scene",
    "a photograph taken with a camera of a real indoor room or space",
    "a photograph taken with a camera of a real object or product",
    "a real photograph of a natural scene",
    "a real photograph of an urban environment",
    "a real photograph of a residential interior",
]

non_photo_prompts = [
    "a screenshot of a computer interface or website",
    "a digital illustration or drawing",
    "a chart, graph, or infographic",
    "a text document or presentation slide",
    "a computer generated 3D render or CGI",
    "a diagram or technical schematic",
    "a logo or graphic design element",
    "an AI generated artificial image",
]

# Tokenize prompts once
photo_tokens = tokenizer(photo_prompts)
non_photo_tokens = tokenizer(non_photo_prompts)

results = []
for img_path in test_images:
    fname = os.path.basename(img_path)
    ds_name = os.path.basename(os.path.dirname(img_path))
    try:
        with Image.open(img_path) as img:
            img.load()
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Preprocess
            t1 = time.time()
            img_tensor = preprocess(img).unsqueeze(0)  # [1,3,H,W]
            
            # Encode
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Text features
                text_features_photo = model.encode_text(photo_tokens)
                text_features_photo /= text_features_photo.norm(dim=-1, keepdim=True)
                text_features_non = model.encode_text(non_photo_tokens)
                text_features_non /= text_features_non.norm(dim=-1, keepdim=True)
                
                # Similarities
                sim_photo = (image_features @ text_features_photo.T).mean().item()
                sim_non = (image_features @ text_features_non.T).mean().item()
                
                # Softmax-like
                photo_score = np.exp(sim_photo) / (np.exp(sim_photo) + np.exp(sim_non))
            
            t2 = time.time()
            
            results.append({
                "dataset": ds_name,
                "file": fname,
                "size": f"{img.size[0]}x{img.size[1]}",
                "photo_similarity": round(sim_photo, 3),
                "non_photo_similarity": round(sim_non, 3),
                "photo_probability": round(photo_score, 3),
                "inference_time_s": round(t2-t1, 2),
                "total_time_s": round(t2-t1, 2),
            })
            print(f"  [{ds_name}] {fname}: photo_prob={photo_score:.3f} (infer={t2-t1:.2f}s)")
    except Exception as e:
        results.append({
            "dataset": ds_name,
            "file": fname,
            "error": str(e)
        })
        print(f"  [{ds_name}] {fname}: ERROR {e}")

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobileclip_validation.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Compute avg time
times = [r["inference_time_s"] for r in results if "inference_time_s" in r]
if times:
    print(f"\nAvg inference time: {np.mean(times):.2f}s per image (n={len(times)})")
    print(f"Min: {np.min(times):.2f}s, Max: {np.max(times):.2f}s")

print(f"\nResults saved to {out_path}")
