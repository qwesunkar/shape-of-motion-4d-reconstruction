import os
import numpy as np
import torch
import imageio.v3 as iio
from flow3d.scene_model import SceneModel
from flow3d.data.iphone_dataset import iPhoneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "output/paper-windmill/checkpoints/last.ckpt"
out_dir = "output/paper-windmill/dataset_views"
os.makedirs(out_dir, exist_ok=True)

print("Loading dataset for real camera params...")
dataset = iPhoneDataset(
    data_dir="data/paper-windmill",
    split="train",
    load_from_cache=True,
)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False)
state_dict = ckpt["model"]
model = SceneModel.init_from_state_dict(state_dict)
model = model.to(device)
model.training = False
print(f"num gs: {model.fg.num_gaussians + model.bg.num_gaussians}")

W, H = dataset.get_img_wh()
Ks = dataset.get_Ks().to(device)
w2cs = dataset.get_w2cs().to(device)

print(f"img_wh={W},{H}  num cameras={w2cs.shape[0]}")

# Render at several real training timesteps, using their real camera pose
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

    # Also save the ground truth for comparison
    gt_img = (dataset.imgs[t].numpy() * 255).astype(np.uint8)
    gt_path = os.path.join(out_dir, f"gt_t{t:03d}.png")
    iio.imwrite(gt_path, gt_img)

    print(f"Saved {out_path} and {gt_path}")

print("Done!")
