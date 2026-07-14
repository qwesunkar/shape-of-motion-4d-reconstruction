import os
import numpy as np
import torch
import imageio.v3 as iio
from flow3d.scene_model import SceneModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "output/paper-windmill/checkpoints/last.ckpt"
out_dir = "output/paper-windmill/novel_views"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False)
state_dict = ckpt["model"]
model = SceneModel.init_from_state_dict(state_dict)
model = model.to(device)
model.training = False
print(f"num gs: {model.fg.num_gaussians + model.bg.num_gaussians}")
print(f"global_step: {ckpt.get('global_step', 0)}")

# Base camera intrinsics (typical iPhone-like)
W, H = 720, 960
focal = 0.9 * H
K = torch.tensor(
    [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
    device=device, dtype=torch.float32,
)

# Base camera pose - looking at scene center from a reasonable distance
base_c2w = torch.eye(4, device=device, dtype=torch.float32)
base_c2w[2, 3] = 2.5  # move back along z

# Pick a mid-training timestep to render
t = int(model.num_frames // 2) if hasattr(model, "num_frames") else 0

print(f"Rendering novel views at t={t}")

# Render a circular camera path around the scene (novel views)
n_views = 12
radius = 2.5
for i in range(n_views):
    angle = 2 * np.pi * i / n_views
    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[0, 3] = radius * np.sin(angle)
    c2w[2, 3] = radius * np.cos(angle)
    # simple look-at rotation towards origin
    z_axis = -c2w[:3, 3] / c2w[:3, 3].norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    x_axis = torch.linalg.cross(up, z_axis)
    x_axis = x_axis / x_axis.norm()
    y_axis = torch.linalg.cross(z_axis, x_axis)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis

    w2c = torch.linalg.inv(c2w)

    with torch.inference_mode():
        out = model.render(t, w2c[None], K[None], (W, H))
        img = out["img"][0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)

    out_path = os.path.join(out_dir, f"view_{i:03d}.png")
    iio.imwrite(out_path, img)
    print(f"Saved {out_path}")

print("Done! Novel views saved to", out_dir)
