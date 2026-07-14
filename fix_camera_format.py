import numpy as np

data = np.load("data/my_head_video/my_head_video.npz")
w2cs = data["w2cs"]
Ks = data["Ks"]

c2ws = np.linalg.inv(w2cs)
K = Ks[0]  # single shared intrinsic

np.savez(
    "data/my_head_video/my_head_video.npz",
    cam_c2w=c2ws,
    intrinsic=K,
)
print("Fixed camera format")
print(f"cam_c2w.shape={c2ws.shape}, intrinsic.shape={K.shape}")
