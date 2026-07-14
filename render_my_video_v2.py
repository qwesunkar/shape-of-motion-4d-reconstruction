import os
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
from flow3d.scene_model import SceneModel
from flow3d.data.casual_dataset import CasualDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "output/my_head_video/checkpoints/last.ckpt"
out_dir = "output/my_head_video/results_v2"
os.makedirs(out_dir, exist_ok=True)

print("Loading dataset (with proper scene normalization)...")
dataset = CasualDataset(
    data_dir="data/my_head_video",
    image_type="JPEGImages",
    mask_type="Annotations",
    res="480p",
    camera_type="megasam",
)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False)
state_dict = ckpt["model"]
model = SceneModel.init_from_state_dict(state_dict)
model = model.to(device)
model.training = False
print(f"num gs: {model.fg.num_gaussians}")
print(f"global_step: {ckpt.get('global_step', 0)}")

W, H = dataset.get_img_wh()
Ks = dataset.get_Ks().to(device)
w2cs = dataset.get_w2cs().to(device)
print(f"img_wh={W},{H}  num cameras={w2cs.shape[0]}")

sample_ts = list(range(0, dataset.num_frames, dataset.num_frames // 8))[:8]

for t in sample_ts:
    K = Ks[t]
    w2c = w2cs[t]
    with torch.inference_mode():
        out = model.render(t, w2c[None], K[None], (W, H))
        img = out["img"][0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)

    out_path = os.path.join(out_dir, f"recon_t{t:03d}.png")
    iio.imwrite(out_path, img)

    gt_img = (dataset.get_image(t).numpy() * 255).astype(np.uint8)
    gt_path = os.path.join(out_dir, f"gt_t{t:03d}.png")
    iio.imwrite(gt_path, gt_img)

    print(f"Saved {out_path} and {gt_path}")

print("Done!")
