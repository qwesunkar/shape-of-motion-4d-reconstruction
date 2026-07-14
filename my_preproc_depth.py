import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

img_dir = "data/my_head_video/JPEGImages/480p"
out_dir = "data/my_head_video/aligned_depth_anything/480p"
os.makedirs(out_dir, exist_ok=True)

print("Loading Depth Anything model...")
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1,
)

img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Found {len(img_paths)} images")

for i, img_path in enumerate(img_paths):
    img = Image.open(img_path).convert("RGB")
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    # Normalize disparity to a reasonable range
    depth = depth / (depth.max() + 1e-8)

    fname = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{fname}.npy")
    np.save(out_path, depth)

    if i % 20 == 0:
        print(f"[{i+1}/{len(img_paths)}] {fname} done")

print("Depth estimation done!")
