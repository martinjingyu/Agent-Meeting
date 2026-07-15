import timm, torch, time, numpy as np
m = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
m.eval()
x = torch.randn(1,3,224,224)
# warm up
with torch.no_grad():
    m(x)
# time 10 iterations
start = time.time()
for _ in range(10):
    with torch.no_grad():
        m(x)
elapsed = time.time() - start
print(f'10 inferences: {elapsed:.3f}s, per inference: {elapsed/10:.3f}s')
