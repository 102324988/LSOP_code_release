#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2 diagnostic: where does generation degradation come from?

Decode 30 val objects under three z strategies and compare profile FWHM:
  (a) mu       - deterministic posterior mean (M1 recon, FWHM ratio 2.25)
  (b) reparam  - posterior sample z = mu + sigma*eps  (what decoder saw in train)
  (c) prior    - z ~ N(0,1)  (M1 gen probe, FWHM 13)
If (c) >> (a) but (b) ~ (a), degradation comes from z distribution mismatch
(p(z) vs N(0,1)) -> latent diffusion is the right fix.
If (b) ~ (c) >> (a), decoder is sensitive to any z != mu -> p(z) diffusion
cannot fix it; different strategy needed.
Also reports per-dim stats of mu / reparam z / N(0,1) z.
"""
import json
import os

import numpy as np
import torch

from m1_train_vae import VAE, load_object, ray_encodings

ROOT = "/root/e0lab/e0"
M1 = "m1_vae_dg1024_b1e-4"
N_BINS = 96


def fwhm(prof):
    pk = prof.argmax(-1)
    hm = prof.max(-1) * 0.5
    w = np.zeros(len(prof), np.float32)
    for i in range(len(prof)):
        p = prof[i]
        lo, hi = pk[i], pk[i]
        while lo > 0 and p[lo - 1] >= hm[i]:
            lo -= 1
        while hi < N_BINS - 1 and p[hi + 1] >= hm[i]:
            hi += 1
        w[i] = hi - lo
    return w


meta = json.load(open(os.path.join(ROOT, M1, "meta.json")))
model = VAE(dim_g=meta["dim_g"])
model.load_state_dict(torch.load(os.path.join(ROOT, M1, "model.pt"),
                                 map_location="cpu"))
model.eval().cuda()
ray_pe, _ = ray_encodings()
ray_pe = ray_pe.cuda()

names = meta["val_names"][:30]
mu_fw, rep_fw, pri_fw, mus = [], [], [], []
with torch.no_grad():
    for n in names:
        sh, cov, peak, rmax = load_object(n)
        sh = sh[None].cuda()
        mu, lv = model.encode(sh)
        m = cov.numpy() >= 0.02
        zs = {"mu": mu, "reparam": mu + torch.exp(0.5 * lv) * torch.randn_like(mu),
              "prior": torch.randn_like(mu)}
        for tag, z in zs.items():
            pred = torch.sigmoid(model.decode(z, ray_pe))[0].cpu().numpy()
            fw = fwhm(pred[m])
            if tag == "mu":
                mu_fw.append(fw)
                mus.append(mu[0].cpu().numpy())
            elif tag == "reparam":
                rep_fw.append(fw)
            else:
                pri_fw.append(fw)
mu_fw = np.concatenate(mu_fw)
rep_fw = np.concatenate(rep_fw)
pri_fw = np.concatenate(pri_fw)
mus = np.stack(mus)
z_rep = mus + 1.02 * np.random.randn(*mus.shape)
z_pri = np.random.randn(*mus.shape)
print("FWHM med:   mu=%.2f  reparam=%.2f  prior=%.2f"
      % (np.median(mu_fw), np.median(rep_fw), np.median(pri_fw)))
print("FWHM p75:   mu=%.2f  reparam=%.2f  prior=%.2f"
      % (np.percentile(mu_fw, 75), np.percentile(rep_fw, 75),
         np.percentile(pri_fw, 75)))
print("z per-dim std: mu=%.4f  reparam=%.4f  prior=1.0000"
      % (mus.std(0).mean(), z_rep.std(0).mean()))
print("|z| (dim mean): mu=%.4f  reparam=%.4f  prior=%.4f"
      % (np.linalg.norm(mus.mean(0)), np.linalg.norm(z_rep.mean(0)),
         np.linalg.norm(z_pri.mean(0))))
print("per-dim |mu|: mean=%.4f  max=%.4f"
      % (np.abs(mus).mean(), np.abs(mus).max()))
