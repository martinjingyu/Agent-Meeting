import torch, time, sys, os
sys.stdout = open(r'C:\Users\LX034\Code\Agent-Meeting\roles\skeptic-reviewer\workspace\bench_mobilenet_out.txt', 'w')

import timm
from PIL import Image
from torchvision import transforms

# Load MobileNetV3-Small
t0 = time.time()
model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True, num_classes=0)  # feature extractor
model.eval()
print(f"Model load time: {time.time()-t0:.1f}s")
p = sum(p.numel() for p in model.parameters())
print(f"MobileNetV3-Small params: {p:,}")

# Transform
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

# Cold inference
rand = torch.randn(1, 3, 224, 224)
t0 = time.time()
with torch.no_grad():
    _ = model(rand)
print(f"Cold inference (random tensor): {time.time()-t0:.3f}s")

# Warm inference
times = []
for _ in range(10):
    t0 = time.time()
    with torch.no_grad():
        _ = model(rand)
    times.append(time.time()-t0)
print(f"Warm avg random (10 runs): {sum(times)/len(times):.4f}s")

# Real image benchmark
truro_dir = r'C:\pics\truro_school'
if os.path.exists(truro_dir):
    jpgs = [f for f in os.listdir(truro_dir) if f.lower().endswith(('.jpg','.jpeg'))]
    if jpgs:
        img = Image.open(os.path.join(truro_dir, jpgs[0])).convert('RGB')
        inp = transform(img).unsqueeze(0)
        # warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(inp)
        times = []
        for _ in range(10):
            t0 = time.time()
            with torch.no_grad():
                feat = model(inp)
            times.append(time.time()-t0)
        print(f"Real image warm avg (10 runs): {sum(times)/len(times):.4f}s")
        print(f"Feature shape: {feat.shape}")

sys.stdout.close()
