# Learning Spherical Occupancy Profiles for Multi-View 3D Reconstruction and Generation

Code and trained model checkpoints for the paper (arXiv:2608.23206).

We study **spherical occupancy profiles (SOPs)** — the ray-wise occupancy probability
profiles $$P(r) = T(r)\,o(r)$$ distilled from multi-view 3D Gaussian reconstructions — as a
unified intermediate representation for both *discriminative* and *generative* 3D
reconstruction from images.

- **Discriminative**: a per-ray decoder injects global view-averaged and ray-specific image
  evidence into a FiLM-conditioned profile head (models **D8**, **M3b**), reaching median
  soft-depth error $$0.035$$ on an independent 90-object test split of Google Scanned Objects.
- **Generative**: a profile VAE (**M1**) plus a latent diffusion model (**M2**) supports
  unconditional sampling on the reconstruction manifold, and an image-conditioned variant
  (**M3a**) performs multi-solution reconstruction whose spread is quantifiable and tunable via
  classifier-free guidance.

## Repository layout

```
sop-release/
├── README.md
├── LICENSE
├── code/                  # 157 Python scripts (training / inference / evaluation / data pipeline)
│   ├── d8_train_full.py   #   D8  training (global-condition decoder)
│   ├── m1_train_vae.py    #   M1  profile VAE
│   ├── m2_train.py        #   M2  latent diffusion
│   ├── m3a_train_cond.py  #   M3a image-conditioned latent diffusion
│   ├── m3b_train_sg.py    #   M3b learned-sharpening head training (--gamma)
│   ├── m3b_train_preray.py#   M3b v1 per-ray pathway
│   ├── d8_infer.py        #   D8 / M3b inference (writes per-object .npy artifacts)
│   ├── m1_infer.py / m2_gen.py / m3a_gen_cond.py / m3b_infer.py
│   ├── d7_eval_fixed.py / d6_eval_v3.py   # metrics used in the paper
│   └── ... (data pipeline: render_turntable.py, build_opacity_grid.py, ...)
└── weights/               # final checkpoints (see table below)
```

## Model checkpoints

| file | model | paper results (val / test) |
|---|---|---|
| `d8_mean_pool_model.pt` | D8, global mean-pool condition | dmed_s 0.0338 / 0.0384 |
| `m3b_v1_head.pt` | M3b v1, per-ray L1 | dmed_s 0.0364 / 0.0356 |
| `m3b_v2_gamma1.5_head.pt` | M3b v2, learned sharpening γ=1.5 | dmed_s 0.0354 / 0.0351 |
| `m3b_v2_gamma2_head.pt` | M3b v2, learned sharpening γ=2 | dmed_s 0.0348 / 0.0356 |
| `m3b_v2_rdim512_head.pt` | M3b v2, ray-channel width 512 (ablation) | dmed_s 0.0354 / 0.0347 |
| `m1_vae_dg1024_b1e-4_model.pt` | M1, profile VAE (main) | val dmed_s 0.0419 |
| `m1_vae_dg1024_b1e-4_aux0.1_model.pt` | M1, auxiliary-loss variant (ablation) | — |
| `m1_vae_dg2048_b1e-4_model.pt` | M1, decoder global dim 2048 (ablation) | — |
| `m2_latent_diff_denoiser.pt` | M2, unconditioned latent diffusion | — |
| `m3a_cond_dd0_denoiser.pt` | M3a, image-conditioned diffusion (main, cond-drop 0.1) | val dmed_s ≈ 0.0498 |
| `m3a_cond_denoiser.pt` | M3a, earlier run (comparison) | — |

The `head.pt` files contain only the ray-conditioned head (2.2M params); the shared encoder and
global branch come from `d8_mean_pool_model.pt`, which the per-ray training initializes from.

## Requirements

- Python 3.10, conda
- torch 2.1.2+cu121, torchvision, numpy, Pillow, scipy, matplotlib
- For the full data-distillation pipeline only: gaussian-opacity-fields (GOF), 3DGS, COLMAP

The exact environment used for the paper is `/root/private_data/miniconda3/envs/e0`.

## Data preparation

The corpus is a 999-object subset of [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Google-Scanned-Objects)
with 48 turntable views per object (819/90/90 train/val/test split, seed 42).

1. **Render** turntable views with `render_turntable.py` (8 input views at evaluation: frames
   0, 6, …, 42; 48 views for the volume stage).
2. **Volume stage** — train a GOF field per object and distill per-ray occupancy profiles
   (see `build_opacity_grid.py` and the `gso_d4` layout below).
3. Training scripts load per-object GT profiles from
   `output/gso_d4/<object>/profiles.npy`, `coverage.npy`, `depth_peak.npy`
   (`load_object` in `m1_train_vae.py`).

> **Path note.** `ROOT` is hardcoded to `/root/e0lab/e0` in the training/inference scripts.
> When using this repository elsewhere, either update `ROOT` in each script or create a
> symlink: `ln -s <your-dir> /root/e0lab/e0`.

## Reproduction

All scripts print per-epoch / per-object metrics to stdout. The paper numbers are medians over
objects of the validation / held-out test splits.

**Discriminative branch.**
```
python d8_train_full.py            # train D8 → d8_mean_pool/model.pt
python d8_infer.py --fusion mean_pool   # write val/test .npy predictions
# then evaluate with the d6/d7 metric scripts (soft-depth, Chamfer)
```

**Per-ray refinement (M3b).**
```
python m3b_train_preray.py         # v1 per-ray pathway, init from D8
python m3b_train_sg.py --gamma 2   # v2 learned-sharpening target ŝ^γ
python m3b_infer.py                # → m3b_preray_<cfg>_{val,test}/summary.json
```

**Generative branch.**
```
python m1_train_vae.py             # profile VAE → m1_vae_dg1024_b1e-4/model.pt
python m1_infer.py                 # → val/test profile-field predictions
python m2_train.py                 # latent diffusion on frozen VAE latents
python m2_gen.py                   # unconditional sampling (DDIM 50 steps)
python m3a_train_cond.py           # image-conditioned diffusion (cond-drop 0.1)
python m3a_gen_cond.py             # conditional sampling / best-of-N / CFG sweep
```

Detailed argument options are available via each script's `--help`. Hyper-parameters follow the
paper's *Setup* section (D8: 40 epochs, Adam lr 3e-4, batch 8, StepLR ×0.3; M3b head: 30 epochs,
lr 3e-4, batch 16, cosine to 5%, grad-clip 1.0; M1: 60 epochs, β=1e-4, 5-epoch KL warm-up; M2/M3a:
400 epochs, v-prediction, linear noise β 1e-4→2e-2, T=1000).

## Results summary (reproducible with the provided weights)

| method | val dmed_s | test dmed_s | notes |
|---|---|---|---|
| mean profile | 0.084 | — | no image condition |
| D8 | 0.0338 | 0.0384 | global condition |
| M3b v1 | 0.0364 | 0.0356 | per-ray L1 |
| M3b v2 γ=1.5 | 0.0354 | 0.0351 | FWHM 1.33 |
| M3b v2 γ=2 | 0.0348 | 0.0356 | FWHM 1.00 |
| M1 (VAE recon.) | 0.0419 | — | latent faithful |
| M3a best-of-8 (CFG 8) | 0.0337 | — | matches D8 |

## Additional experiments (Pattern Recognition review revision)

Scripts added for the review revision — center robustness, inference efficiency, and an
external ray-wise baseline with view scaling, all on the same 819/90/90 split and the same
soft-depth protocol:

| script | experiment | paper section |
|---|---|---|
| `center_robust.py` | D8 robustness to object-center mis-estimation, $\Delta x = 0/2/4/9$ px ($\approx 0$–$18.7\%$ of object radius) | center-robustness table |
| `runtime_prof.py` | single-object latency + peak GPU memory (NVIDIA L20, fp32) | inference-efficiency table |
| `exts_baseline.py` | trains D8 with $K{=}1/2$ views and a RayDF-style ray-wise distance field with $K{=}8/1$ views, same encoder / fusion / optimizer / metric | "External baseline and view scaling" |
| `run_exts.sh` | serial driver for the four `exts_baseline.py` runs | — |

Key numbers:

| model | views $K$ | val dmed_s | test dmed_s |
|---|---|---|---|
| D8 (profile) | 8 | 0.0338 | 0.0384 |
| D8 (profile) | 2 | 0.0430 | 0.0419 |
| D8 (profile) | 1 | 0.0380 | 0.0442 |
| Ray-wise (RayDF-style) | 8 | 0.0396 | 0.0369 |
| Ray-wise (RayDF-style) | 1 | 0.0455 | 0.0422 |

Center robustness: $\mathrm{dmed_s}$ stays within $0.0338\to0.0373$ up to $\Delta x = 9$ px
($18.7\%$ radius). Inference: D8 $2.2$ ms / $165$ MB; M3b $4.8$ ms / $474$ MB; M1 (VAE recon)
$1.5$ ms; M2 (50-step DDIM sample) $19.5$ ms.

> **Checkpoints for the view-scaling / ray-wise models** (`d8_mean_pool_v1/model.pt`,
> `d8_mean_pool_v2/model.pt`, `rw_dist_v8/model.pt`, `rw_dist_v1/model.pt`) live on the compute
> server used for the revision and will be uploaded to GitHub Releases when that server is next
> powered on. The code and the numbers above are final.

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@misc{tsai2026learning,
  title  = {Learning Spherical Occupancy Profiles for Multi-View 3D Reconstruction and Generation},
  author = {Tsai, YiHsuan},
  year   = {2026},
  eprint = {2608.23206},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
