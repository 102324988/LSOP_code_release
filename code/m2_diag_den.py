#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2 denoiser diagnostic: how well does the denoiser predict eps per t?

The DDIM sampler exploded (|z|_dim ~40 vs expected ~0.8). Hypothesis: the
denoiser fails at extreme timesteps — near t=0 eps_true is pure noise
(optimal pred ~0, loss floor ~1) and near t=T x_t ~ eps (optimal pred ~ x_t,
loss floor ~0). If the model didn't learn the t=T trivial solution, DDIM's
x0 = (x - eps)/sqrt(a_bar) amplifies its error ~1/sqrt(a_bar) and blows up.

Checks per t-bucket: eps-prediction MSE, and cosine(eps_pred, x_t) near t=T
(should approach 1 if it learned "predict the noise itself").
"""
import json
import os

import numpy as np
import torch

from m1_train_vae import VAE, load_object
from m2_train import TimeMLP, get_beta_schedule

ROOT = "/root/e0lab/e0"
M2 = os.path.join(ROOT, "m2_latent_diff")
N_BINS = 96


meta = json.load(open(os.path.join(M2, "meta.json")))
T = meta["T"]
dim_g = meta["dim_g"]
_, _, alpha_bar = get_beta_schedule(T)

den = TimeMLP(dim_g)
den.load_state_dict(torch.load(os.path.join(M2, "denoiser.pt"), map_location="cpu"))
den.eval()

Zw = np.load(os.path.join(M2, "z_mean.npy"))  # just reuse meta; Zw not saved
# reconstruct whitened data: we have mu_all; approx p(z) sample set
mu_all = np.load(os.path.join(M2, "mu_all.npy")).astype(np.float32)
z_mean = np.load(os.path.join(M2, "z_mean.npy")).astype(np.float32)
z_std = np.load(os.path.join(M2, "z_std.npy")).astype(np.float32)
# approximate whitened training data as standard normal samples perturbed
rng = np.random.RandomState(0)
Zw = (mu_all - z_mean) / z_std + 1.02 * rng.randn(*mu_all.shape)
Zt = torch.from_numpy(Zw.astype(np.float32))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
den = den.to(device)
Zt = Zt.to(device)
alpha_bar = alpha_bar.to(device)

buckets = [(0.02, "t~0"), (0.3, "t~0.3"), (0.5, "t~0.5"),
           (0.8, "t~0.8"), (0.95, "t~0.95"), (0.999, "t~1")]
with torch.no_grad():
    for frac, label in buckets:
        t = torch.full((len(Zt),), frac, device=device)
        eps = torch.randn_like(Zt)
        a_bar = alpha_bar[(t * (T - 1)).long()]
        x_t = torch.sqrt(a_bar[:, None]) * Zt + torch.sqrt(1.0 - a_bar[:, None]) * eps
        eps_pred = den(x_t, t)
        mse = ((eps_pred - eps) ** 2).mean().item()
        # what the model effectively predicts about x0 (per-dim std of x0_pred)
        x0 = (x_t - torch.sqrt(1.0 - a_bar[:, None]) * eps_pred) / torch.sqrt(a_bar[:, None])
        cos_noise = torch.nn.functional.cosine_similarity(
            eps_pred.flatten(1).mean(0), x_t.flatten(1).mean(0), dim=0).item()
        print("%-6s a_bar=%.1e  eps_mse=%.4f  |x0_pred|_dim=%.2f  cos(eps_pred,x_t)=%.3f"
              % (label, a_bar[0].item(), mse,
                 float(x0.flatten(1).abs().mean()), cos_noise), flush=True)

# direct DDIM probe: run 3 steps from the top and watch |x| grow
alpha_bar_cpu = alpha_bar.cpu()
x = torch.randn(4, dim_g, device=device)
print("\nDDIM |x|_dim walk (steps=50, first 6 iters):")
ts = np.linspace(T - 1, 0, 50).astype(np.int64)
for i, t in enumerate(ts[:6]):
    t_prev = 0 if i == 49 else int(ts[i + 1])
    a_t = alpha_bar_cpu[t].item()
    a_tp = alpha_bar_cpu[t_prev].item()
    t_frac = torch.full((4,), t / (T - 1), dtype=torch.float32, device=device)
    eps = den(x, t_frac)
    x0 = (x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
    x = np.sqrt(a_tp) * x0 + np.sqrt(1.0 - a_tp) * eps
    print("iter %d t=%d  a_bar=%.2e  |x|_dim=%.2f" % (i, t, a_t, float(x.abs().mean())))
