#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3a: multi-solution diversity of the conditional generative model.

For each of the first N_VAL val objects, sample S=8 solutions z~p(z|c_j)
at CFG weights w in {0,1,8} (w=0 = unconditional branch baseline), decode
through the frozen M1 decoder, and measure:

  - INTRA-object diversity: pairwise Chamfer among the S solutions of one
    object. Large = the conditional posterior proposes genuinely distinct
    plausible reconstructions (multi-hypothesis claim); ~0 = collapse to a
    single solution (deterministic, condition fully pins the shape).
  - INTER-object diversity: pairwise Chamfer among the N_VAL average-profile
    clouds (cross-object spread; M2-protocol reference). Compares against
    the unconditional spread to see whether conditioning narrows the space.
  - z-spread: pairwise L2 among the whitened z samples (posterior width in
    latent space, decoder-free).
  - depth tie-in: best_of_N / avg_profile dmed per object (links diversity
    to the known cfg numbers: best 0.0337/0.038, avg 0.055/0.056).

References computed in-script with the SAME protocol (m2_gen): GT
cross-object pairwise Chamfer among the same N_VAL objects.

Usage: python m3a_diversity.py --out m3a_div_val
"""
import argparse
import json
import os

import numpy as np
import torch

from m1_train_vae import D4, N_BINS, VAE, load_object, ray_encodings
from m2_train import get_beta_schedule
from m2_gen import dirs_grid, pairwise_chamfer
from m3a_train_cond import CondTimeMLP, encode_cond, load_views
from m3a_cfg import ddim_sample_cfg, dmed_of

ROOT = "/root/e0lab/e0"
N_BINS = 96
N_VAL = 24          # objects (matches M2 GT-reference scale, more robust)
S = 8               # solutions per object
WS = [0.0, 1.0, 8.0]
DDIM_STEPS = 50


def profile_cloud(pred, dirs, rmax_ref, pmax=0.3):
    """Single (R,96) profile -> soft-depth point cloud (m2_gen protocol)."""
    axis = np.arange(N_BINS)
    soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
    r = soft / (N_BINS - 1) * rmax_ref
    keep = pred.max(-1) > pmax
    if keep.sum() < 100:
        keep = np.ones(len(pred), dtype=bool)
    return (r[keep, None] * dirs[keep]).astype(np.float32)


def gt_cloud(sh, cov, rmax_ref, dirs):
    """GT normalized profile -> soft-depth cloud on covered rays (m2_gen)."""
    axis = np.arange(N_BINS)
    m = cov >= 0.02
    soft = (sh[m] * axis[None, :]).sum(-1) / (sh[m].sum(-1) + 1e-6)
    r = soft / (N_BINS - 1) * rmax_ref
    return (r[:, None] * dirs[m]).astype(np.float32)


def z_pairwise_l2(z):
    """Median pairwise L2 distance among rows of z (whitened space)."""
    n = z.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = float(np.linalg.norm(z[i] - z[j]))
    return float(np.median(d[d > 0])) if (d > 0).any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3a", default="m3a_cond")
    ap.add_argument("--out", default="m3a_div_val")
    ap.add_argument("--nv", type=int, default=N_VAL)
    ap.add_argument("--s", type=int, default=S)
    ap.add_argument("--ws", default="0,1,8")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    ws = [float(x) for x in args.ws.split(",")]

    m3a = json.load(open(os.path.join(ROOT, args.m3a, "meta.json")))
    m1 = json.load(open(os.path.join(ROOT, m3a["m1_ckpt"], "meta.json")))
    dim_g, T = m3a["dim_g"], m3a["T"]
    names = m1["val_names"][:args.nv]
    print("val objects n=%d  S=%d  ws=%s" % (len(names), args.s, ws),
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, dirs_t = ray_encodings()
    dirs = dirs_grid()                     # (8192,3) m2_gen ordering
    ray_pe = ray_pe.to(device)

    # rmax_ref: median over train-object metas (same as m2_gen/m1_gen)
    rmaxs = []
    for n in m3a["train_names"][::40]:
        rm = json.load(open(os.path.join(D4, n, "profiles", "meta.json")))
        rmaxs.append(rm["rmax"])
    rmax_ref = float(np.median(rmaxs))
    print("rmax_ref=%.3f (median over train)" % rmax_ref, flush=True)

    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, m3a["m1_ckpt"], "model.pt"), map_location="cpu"))
    model.eval().to(device)

    from d8_train_full import Model as D8Model
    d8 = D8Model(fusion="mean_pool")
    d8.load_state_dict(torch.load(
        os.path.join(ROOT, m3a["d8_ckpt"], "model.pt"), map_location="cpu"))
    d8.eval().to(device)

    _, _, alpha_bar = get_beta_schedule(T)
    den = CondTimeMLP(dim_g, m3a["cond_dim"])
    den.load_state_dict(torch.load(os.path.join(ROOT, args.m3a,
                                                "denoiser.pt"),
                                   map_location="cpu"))
    den.to(device)
    z_mean = np.load(os.path.join(ROOT, args.m3a, "z_mean.npy")).astype(np.float32)
    z_std = np.load(os.path.join(ROOT, args.m3a, "z_std.npy")).astype(np.float32)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        profs = dict(pool.map(lambda n: (n, load_object(n)), names, chunksize=4))
        views = dict(pool.map(lambda n: (n, load_views(n)), names, chunksize=4))

    # GT cross-object reference (same N_VAL objects, same rmax_ref)
    gt_pcs = [gt_cloud(profs[n][0].numpy(), profs[n][1].numpy(), rmax_ref,
                       dirs) for n in names]
    Dg = pairwise_chamfer(gt_pcs)
    go = Dg[np.triu_indices(len(names), 1)]

    torch.manual_seed(args.seed)
    out = {}
    for w in ws:
        intra_all, zsp_all, best_all, avg_all = [], [], [], []
        avg_clouds = []
        with torch.no_grad():
            for n in names:
                imgs, feats = views[n]
                c = encode_cond(d8, imgs[None].to(device),
                                feats[None].to(device))          # (1,512)
                C = c.repeat(args.s, 1)
                z_white = ddim_sample_cfg(den, alpha_bar, C, dim_g, w,
                                          DDIM_STEPS, device)
                z = (z_white.cpu().numpy() * z_std + z_mean).astype(np.float32)
                pred = torch.sigmoid(model.decode(
                    torch.from_numpy(z).to(device), ray_pe)).cpu().numpy()
                sh, cov, peak, rmax = profs[n]
                clouds = [profile_cloud(pred[k], dirs, rmax_ref)
                          for k in range(args.s)]
                D = pairwise_chamfer(clouds)
                intra_all.append(float(np.median(D[np.triu_indices(args.s, 1)])))
                zsp_all.append(z_pairwise_l2(z_white.cpu().numpy()))
                dj = [dmed_of(pred[k], cov, peak, rmax) for k in range(args.s)]
                best_all.append(float(np.min(dj)))
                avg_all.append(dmed_of(pred.mean(0), cov, peak, rmax))
                avg_clouds.append(profile_cloud(pred.mean(0), dirs, rmax_ref))
        Di = pairwise_chamfer(avg_clouds)         # inter-object (mean solutions)
        kws = "w=%.0f" % w
        out[kws] = {
            "intra_solution_chamfer_med": float(np.median(intra_all)),
            "intra_solution_chamfer_p25": float(np.percentile(intra_all, 25)),
            "intra_solution_chamfer_p75": float(np.percentile(intra_all, 75)),
            "intra_solution_chamfer_mean": float(np.mean(intra_all)),
            "inter_object_chamfer_med": float(
                np.median(Di[np.triu_indices(len(names), 1)])),
            "z_spread_med": float(np.median(zsp_all)),
            "best_of_S_med": float(np.median(best_all)),
            "avg_profile_med": float(np.median(avg_all)),
            "avg_profile_mean": float(np.mean(avg_all)),
        }
        print("%s  intra_sol=%.4f [%.3f-%.3f]  inter_obj=%.4f  z_spread=%.3f  "
              "best=%.4f  avg=%.4f" %
              (kws, out[kws]["intra_solution_chamfer_med"],
               out[kws]["intra_solution_chamfer_p25"],
               out[kws]["intra_solution_chamfer_p75"],
               out[kws]["inter_object_chamfer_med"], out[kws]["z_spread_med"],
               out[kws]["best_of_S_med"], out[kws]["avg_profile_med"]),
              flush=True)

    out["_refs"] = {
        "gt_cross_object_chamfer_med": float(np.median(go)),
        "gt_cross_object_chamfer_min": float(go.min()),
        "m2_uncond_cross_object": 0.034,          # stored m2_gen summary
        "n_objects": len(names), "s_per_object": args.s, "ws": ws,
        "rmax_ref": rmax_ref,
    }
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, args.out, "summary.json"), "w"),
              indent=1)
    print("saved -> %s/summary.json" % args.out, flush=True)
    print("GT cross-object chamfer med=%.4f (reference)" % out["_refs"][
        "gt_cross_object_chamfer_med"], flush=True)


if __name__ == "__main__":
    main()
