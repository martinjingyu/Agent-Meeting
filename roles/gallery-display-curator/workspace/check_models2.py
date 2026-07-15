"""Check actual MobileCLIP model availability with correct pretrained tags."""
import open_clip
import torch

# MobileCLIP-S1 / S2 use 'datacompdr'
# MobileCLIP-B uses 'datacompdr' or 'datacompdr_lt'
# MobileCLIP2-* use 'dfndr2b'

checks = [
    ("MobileCLIP-S1", "datacompdr"),
    ("MobileCLIP-S2", "datacompdr"),
    ("MobileCLIP-B", "datacompdr_lt"),
    ("MobileCLIP2-S0", "dfndr2b"),
    ("MobileCLIP2-S2", "dfndr2b"),
]

for name, tag in checks:
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=tag)
        p = sum(p.numel() for p in model.parameters())
        print(f"{name} ({tag}): {p/1e6:.1f}M params  (under 100M: {'YES' if p < 100_000_000 else 'NO'})")
        # Quick timing test
        dummy = torch.randn(1, 3, 224, 224)
        import time
        t0 = time.time()
        with torch.no_grad():
            _ = model.encode_image(dummy)
        print(f"  Single forward: {time.time()-t0:.3f}s")
        del model
    except Exception as e:
        print(f"{name} ({tag}): ERROR - {e}")
