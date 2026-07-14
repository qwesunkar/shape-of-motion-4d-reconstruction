import os
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
from flow3d.scene_model import SceneModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "output/my_head_video/checkpoints/last.ckpt"
out_dir = "output/my_head_video/results"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False)
state_dict = ckpt["model"]
model = SceneModel.init_from_state_dict(state_dict)
model = model.to(device)
model.training = False
print(f"num gs: {model.fg.num_gaussians}")
print(f"global_step: {ckpt.get('global_step', 0)}")

# Load real camera params
cam_data = np.load("data/my_head_video/my_head_video.npz")
c2ws = cam_data["cam_c2w"]
K = cam_data["intrinsic"]

W, H = 480, 854
K_t = torch.tensor(K, device=device, dtype=torch.float32)

img_dir = "data/my_head_video/JPEGImages/480p"
img_paths = sorted(os.listdir(img_dir))

sample_ts = list(range(0, len(img_paths), len(img_paths) // 8))[:8]

for t in sample_ts:
    c2w = torch.tensor(c2ws[t], device=device, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)

    with torch.inference_mode():
        out = model.render(t, w2c[None], K_t[None], (W, H))
        img = out["img"][0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)

    out_path = os.path.join(out_dir, f"recon_t{t:03d}.png")
    iio.imwrite(out_path, img)

    gt_path_src = os.path.join(img_dir, img_paths[t])
    gt_img = Image.open(gt_path_src).resize((W, H))
    gt_out_path = os.path.join(out_dir, f"gt_t{t:03d}.png")
    gt_img.save(gt_out_path)

    print(f"Saved {out_path} and {gt_out_path}")

print("Done!")
