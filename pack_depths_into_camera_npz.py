import numpy as np
import glob
import os

depth_dir = "data/my_head_video/aligned_depth_anything/480p"
depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
print(f"Found {len(depth_paths)} depth files")

depths = np.stack([np.load(p) for p in depth_paths], axis=0)
print(f"depths.shape={depths.shape}")

cam_path = "data/my_head_video/my_head_video.npz"
data = dict(np.load(cam_path))
data["depths"] = depths

np.savez(cam_path, **data)
print("Saved depths into camera npz")
print("keys:", list(np.load(cam_path).keys()))
