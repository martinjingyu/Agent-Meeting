"""Check MobileCLIP model availability and parameter counts."""
import open_clip

models_to_check = [
    "MobileCLIP2-S0",
    "MobileCLIP2-S2",
    "MobileCLIP-S1",
    "MobileCLIP-S2",
    "MobileCLIP-B",
]

for m in models_to_check:
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            m, pretrained="datacompdr_lt_256"
        )
        p = sum(p.numel() for p in model.parameters())
        print(f"{m}: {p/1e6:.1f}M params  (under 100M: {'YES' if p < 100_000_000 else 'NO'})")
        del model
    except Exception as e:
        print(f"{m}: ERROR loading - {e}")
