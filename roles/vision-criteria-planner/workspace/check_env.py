"""Quick environment validation script."""
import os, sys
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

# Check C:\pics
pics_path = r"C:\pics"
if os.path.isdir(pics_path):
    datasets = [d for d in os.listdir(pics_path) if os.path.isdir(os.path.join(pics_path, d))]
    print(f"C:\\pics datasets ({len(datasets)}): {datasets}")
    # Count files in first few
    for ds in datasets[:5]:
        files = [f for f in os.listdir(os.path.join(pics_path, ds)) 
                 if os.path.isfile(os.path.join(pics_path, ds, f))]
        print(f"  {ds}: {len(files)} files, sample: {files[:3]}")
else:
    print("C:\\pics does not exist!")

# Check available packages
pkgs = ["PIL", "numpy", "torch", "open_clip_torch", "transformers", "timm", "cv2"]
for pkg in pkgs:
    try:
        __import__(pkg.replace("_", "-").split(".")[0] if "-" in pkg else pkg)
        print(f"{pkg}: available")
    except ImportError:
        print(f"{pkg}: NOT available")

print("Done.")
