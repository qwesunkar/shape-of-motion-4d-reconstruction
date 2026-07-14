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

# First pass: compute raw disparity maps and track global min/max
raw_depths = []
for i, img_path in enumerate(img_paths):
    img = Image.open(img_path).convert("RGB")
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    raw_depths.append(depth)
    if i % 20 == 0:
        print(f"[pass1 {i+1}/{len(img_paths)}] done")

raw_depths = np.stack(raw_depths, axis=0)
global_min = raw_depths.min()
global_max = raw_depths.max()
print(f"Global disparity range: [{global_min}, {global_max}]")

# Depth Anything outputs DISPARITY (inverse depth). Convert to depth,
# then normalize with a single global scale so all frames are consistent.
eps = 1e-6
disp_norm = (raw_depths - global_min) / (global_max - global_min + eps)
disp_norm = np.clip(disp_norm, 0.05, 1.0)  # avoid div by ~0
depth_metric = 1.0 / disp_norm  # relative depth, consistent scale across frames
# Normalize overall scale to a reasonable range (median ~1.0)
depth_metric = depth_metric / np.median(depth_metric)

for i, img_path in enumerate(img_paths):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{fname}.npy")
    np.save(out_path, depth_metric[i])

print(f"Saved {len(img_paths)} depth maps with GLOBAL consistent scale")
print(f"depth_metric range: [{depth_metric.min():.3f}, {depth_metric.max():.3f}]")
