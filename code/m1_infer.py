#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 VAE inference: unconditional reconstruction (encode -> mu -> decode).

Produces d7_eval_fixed-compatible artifacts (hard depth bin, soft-argmax,
pred profile, coverage, pred peak) for every object in a split, so the
authoritative evaluator scores M1 exactly like D6/D7/D8. Uses the posterior
MEAN (deterministic) as the readout latent.

Usage:
  python m1_infer.py --split val|test [--outtag NAME]
  # reads m1_vae_dg<dim>_b<beta>/model.pt+meta.json + d8_mean_pool/meta.json (test names)
"""
import argparse
import json
import os

import numpy as np
import torch

from m1_train_vae import (D4, N_PHI, N_THETA, N_BINS, VAE, ray_encodings,
                          load_object)

ROOT = "/root/e0lab/e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", default="m1_vae_dg1024_b1e-3")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ROOT, args.ckpt, "meta.json")))
    dim_g = meta["dim_g"]
    ray_pe, _ = ray_encodings()
    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, args.ckpt, "model.pt")))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ray_pe = ray_pe.to(device)

    d8 = json.load(open(os.path.join(ROOT, "d8_mean_pool", "meta.json")))
    names = meta["val_names"] if args.split == "val" else d8["test_names"]
    outdir = os.path.join(ROOT, args.ckpt + "_" + args.split)
    os.makedirs(outdir, exist_ok=True)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        data = dict(pool.map(lambda n: (n, load_object(n)), names, chunksize=4))

    axis = torch.arange(N_BINS)
    with torch.no_grad():
        for i, n in enumerate(names):
            sh, cov, _, _ = data[n]
            mu, _ = model.encode(sh[None].to(device))
            pred_t = torch.sigmoid(model.decode(mu, ray_pe))[0].cpu()
            soft_t = ((pred_t * axis[None, :]).sum(-1) /
                      (pred_t.sum(-1) + 1e-6))
            pred = pred_t.numpy()
            soft = soft_t.numpy().astype(np.float32)
            hard = pred.argmax(-1).astype(np.float32)
            np.save(os.path.join(outdir, n + ".npy"), hard)
            np.save(os.path.join(outdir, n + "_soft.npy"), soft)
            np.save(os.path.join(outdir, n + "_prof.npy"), pred.astype(np.float32))
            np.save(os.path.join(outdir, n + "_cov.npy"), cov.numpy())
            np.save(os.path.join(outdir, n + "_predpeak.npy"),
                    pred.max(-1).astype(np.float32))
            if (i + 1) % 30 == 0:
                print("%d/%d %s" % (i + 1, len(names), n), flush=True)
    print("done -> %s/ (%d objects)" % (outdir, len(names)))


if __name__ == "__main__":
    main()
