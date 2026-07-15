import torch, open_clip, time, sys, os

sys.stdout = open(r'C:\Users\LX034\Code\Agent-Meeting\roles\skeptic-reviewer\workspace\bench_clip_out.txt', 'w')

# Load model with tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

# Test real vs non-real prompts
real_prompts = [
    "a real photograph of an outdoor landscape, cityscape, or building",
    "a real photograph of people in a natural scene or event",
    "a real photograph of an interior room or indoor space",
    "a real photograph of a building, architecture, or urban scene",
    "a real camera photo showing a realistic scene"
]
nonreal_prompts = [
    "a screenshot, illustration, or digital graphic",
    "a CGI rendering or computer generated image",
    "a diagram, chart, infographic, or text document",
    "a cartoon, anime, or stylized digital art",
    "a synthetic or AI-generated image"
]

all_prompts = real_prompts + nonreal_prompts
print(f"Number of prompts: {len(all_prompts)}")

t0 = time.time()
text_tokens = tokenizer(all_prompts)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(f"Text feature extraction: {time.time() - t0:.3f}s")
print(f"Text feature shape: {text_features.shape}")

# Test zero-shot classification
rand_input = torch.randn(1, 3, 224, 224)
t0 = time.time()
with torch.no_grad():
    image_features = model.encode_image(rand_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1)
    real_prob = probs[0, :5].sum().item()
    nonreal_prob = probs[0, 5:].sum().item()
    print(f"Zero-shot P(real): {real_prob:.4f}, P(non-real): {nonreal_prob:.4f}")
print(f"Zero-shot classification time: {time.time() - t0:.3f}s")

# Now benchmark on a real image
from PIL import Image

# Find a jpg file from truro_school
truro_dir = r'C:\pics\truro_school'
if os.path.exists(truro_dir):
    jpgs = [f for f in os.listdir(truro_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    if jpgs:
        sample = os.path.join(truro_dir, jpgs[0])
        img = Image.open(sample).convert('RGB')
        print(f"\nReal image: {jpgs[0]}, size={img.size}")
        
        # Warm caches first
        for _ in range(3):
            _ = model.encode_image(preprocess(img).unsqueeze(0))
        
        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.time()
            inp = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(inp)
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        print(f"Real image avg inference (10 runs): {avg:.4f}s")
        print(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s")

sys.stdout.close()
