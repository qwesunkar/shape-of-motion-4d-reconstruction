import os
import glob
import numpy as np
from PIL import Image

img_dir = "data/my_head_video/JPEGImages/480p"
mask_out_dir = "data/my_head_video/Annotations/480p"
os.makedirs(mask_out_dir, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Found {len(img_paths)} images")

# Create simple full-frame masks (foreground = everything, since head fills most of frame)
for img_path in img_paths:
    img = Image.open(img_path)
    W, H = img.size
    mask = np.ones((H, W), dtype=np.uint8) * 255
    fname = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(mask_out_dir, f"{fname}.png")
    Image.fromarray(mask).save(out_path)

print(f"Saved {len(img_paths)} masks")

# Create static camera poses (identity) since camera did not move
img = Image.open(img_paths[0])
W, H = img.size
N = len(img_paths)

# reasonable focal length guess (iPhone-like FOV ~60deg)
focal = 1.2 * max(W, H)
K = np.array([
    [focal, 0, W / 2.0],
    [0, focal, H / 2.0],
    [0, 0, 1.0]
], dtype=np.float32)

Ks = np.tile(K[None], (N, 1, 1))
w2cs = np.tile(np.eye(4, dtype=np.float32)[None], (N, 1, 1))

data_name = "my_head_video"
out_path = f"data/my_head_video/{data_name}.npz"
np.savez(out_path, w2cs=w2cs, Ks=Ks)
print(f"Saved camera params to {out_path}")
print(f"W={W}, H={H}, focal={focal}")
