import torch, open_clip, time, sys

sys.stdout = open(r'C:\Users\LX034\Code\Agent-Meeting\roles\skeptic-reviewer\workspace\bench_output.txt', 'w')

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')
load_time = time.time() - t0
p = sum(p.numel() for p in model.parameters())
vp = sum(p.numel() for p in model.visual.parameters())
print(f'MobileCLIP2-S0 total params: {p:,}')
print(f'Visual tower params: {vp:,}')
print(f'Under 100M: {p < 100_000_000}')
print(f'Model load time: {load_time:.1f}s')

# First inference (cold, includes torch setup)
t0 = time.time()
rand_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model.encode_image(rand_input)
inf_time = time.time() - t0
print(f'First inference (cold, includes randn): {inf_time:.3f}s')

# Warm
t0 = time.time()
for _ in range(5):
    rand_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model.encode_image(rand_input)
avg_inf = (time.time() - t0) / 5
print(f'Average warm inference (5 runs): {avg_inf:.3f}s')
print(f'Output shape: {out.shape}')

sys.stdout.close()
