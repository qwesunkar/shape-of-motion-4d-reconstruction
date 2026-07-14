# 4D Novel View Synthesis from Monocular Video (Shape of Motion)

This project applies [**Shape of Motion**](https://github.com/vye16/shape-of-motion) (ICCV 2025) — a method for
4D reconstruction and novel-view synthesis from **casual monocular video** — to:

1. A benchmark scene (`paper-windmill`, from the official iPhone dataset)
2. A **custom video I captured myself** on an iPhone (a static camera filming a person's head turning)

The goal was to reconstruct a dynamic 3D scene from a single, uncalibrated video and render it from novel
viewpoints / at novel points in time — going through the full pipeline personally, including debugging
a from-scratch custom-data preprocessing pipeline.

## Results

### Experiment 1 — Benchmark (paper-windmill)
Trained on the official iPhone dataset sequence. ~71k Gaussians, reconstruction closely matches ground truth
including fine details (windmill blades, background wood fence texture).

| Ground Truth | Reconstruction |
|---|---|
| `output/paper-windmill/dataset_views/gt_t136.png` | `output/paper-windmill/dataset_views/recon_t136.png` |

### Experiment 2 — Custom video (own head-turn video)
Captured a 9.5s / 285-frame 1080p video on an iPhone with the camera held static, head turning side to side.
Built a full custom preprocessing pipeline (see below) since the raw video isn't in the format the model expects.

| Ground Truth | Reconstruction |
|---|---|
| `output/my_head_video/results_v2/gt_t036.png` | `output/my_head_video/results_v2/recon_t036.png` |

Result is recognizable (head shape, hair, face, dark t-shirt, room/chandelier in background) but noticeably
blurrier than the benchmark — discussed in Limitations below.

## Custom Data Preprocessing Pipeline

The official repo expects heavy preprocessing (DROID-SLAM for camera poses, UniDepth/COLMAP, SAM+XMem for masks,
BootsTAPIR for 2D tracks) that is difficult to install reliably (CUDA-version-locked, several submodules require
compiling custom C++/CUDA extensions). Given a **static camera**, I built a lighter custom pipeline:

1. **Frame extraction** — `ffmpeg`, downscaled to 10fps / 480x854 (`my_preproc_masks_cams.py`)
2. **Monocular depth** — [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
   (`my_preproc_depth_v2.py`) — critically, depth had to be normalized with a **single global scale across all
   frames**, not per-frame, or the triangulated 3D points are inconsistent between frames (see Limitations)
3. **Masks** — simplified to full-frame foreground masks (head fills most of the frame)
4. **Camera poses** — identity extrinsics for every frame (camera was physically static), single shared intrinsic
   matrix (`my_preproc_masks_cams.py`, `fix_camera_format.py`)
5. **2D point tracks** — BootsTAPIR, PyTorch implementation from the repo's `tapnet_torch` submodule
   (avoids the Jax dependency chain), grid_size=8 (`preproc/compute_tracks_torch.py`)
6. Packed depth into the camera `.npz` in the format `flow3d`'s `CasualDataset` (`camera_type="megasam"`) expects
   (`pack_depths_into_camera_npz.py`)

## Rendering

`render_from_dataset.py` / `render_my_video_v2.py` load a trained checkpoint and render frames using the
dataset's **normalized** camera poses (`dataset.get_w2cs()` / `get_Ks()`) — an early version of the render
script used raw un-normalized camera extrinsics directly and produced meaningless blurry output, since the
Gaussians live in the model's internally rescaled/re-centered coordinate frame, not the raw input frame
(see Problems & Fixes).

## Problems & Fixes (the interesting part)

- **Blackwell GPU (RTX 5060, sm_120) vs. stable PyTorch**: no stable PyTorch build supports sm_120 yet →
  used PyTorch nightly (`cu128`/`cu130`) throughout.
- **`gsplat` failed to compile**: multiple layered issues — `NVCC_PREPEND_FLAGS` env var injected by conda
  containing a literal `UNSET` token that nvcc parsed as a source file; Ubuntu 26.04 system headers conflicting
  with CUDA math headers (`rsqrt` redefinition); needed to force the CUDA-toolkit-provided g++ as `-ccbin`.
  Root-caused via manually replaying the exact `nvcc` invocation from the ninja build file outside of ninja.
- **`gsplat` 1.5.x API drift vs. Shape of Motion's `trainer.py`**: `_prepare_control_step` assumed old
  `_current_radii`/`_current_xys` tensor shapes (per-camera vs. flattened); patched the adaptive-density-control
  indexing logic to handle the new flattened `(C*G,)` shapes.
- **WSL2 instability under memory pressure**: WSL crashed outright (`Wsl/Service/E_UNEXPECTED`) during
  densification when Gaussian count spiked; fixed by setting a `.wslconfig` memory/swap cap and disabling
  adaptive density control after an early step for a stable, predictable memory profile.
- **`torch.load` default `weights_only=True`** (PyTorch 2.6+) broke checkpoint loading (numpy globals in the
  pickled state); explicitly passed `weights_only=False`.
- **Per-frame depth normalization** produced geometrically inconsistent 3D points across frames (same disparity
  value ≠ same real depth in different frames) → completely degenerate reconstruction on the custom video.
  Fixed by computing depth normalization statistics globally, across the whole video, once.
- **Render script using raw camera extrinsics**: produced meaningless blurry blobs even after training
  converged with a low loss, because the trained Gaussians live in the dataset's internally normalized
  coordinate frame (scale + re-centering transform computed in `CasualDataset.__init__`). Fixed by rendering
  through `dataset.get_w2cs()`/`get_Ks()` rather than the raw input `.npz`.

## Limitations

- No true camera-pose estimation (DROID-SLAM/COLMAP) was used for the custom video — this only works because
  the physical camera was static; it would not generalize to a moving camera.
- Only 500 optimization steps / ~4.8k foreground Gaussians on the custom video (vs. the benchmark's ~71k) to
  keep training memory-stable on an 8GB laptop GPU under WSL2 — visibly limits sharpness, especially at
  motion boundaries (hair, silhouette edges).
- Simplified full-frame foreground masks (no SAM/XMem segmentation) — no explicit background model was learned.

## Acknowledgements

Built on top of [Shape of Motion](https://github.com/vye16/shape-of-motion) (Wang et al., ICCV 2025) and its
dependencies: [gsplat](https://github.com/nerfstudio-project/gsplat),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[TAPIR/BootsTAPIR](https://github.com/google-deepmind/tapnet).
