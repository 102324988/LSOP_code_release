#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q5 runtime profiling (reviewer): end-to-end inference latency + peak VRAM
for each reconstruction path in the paper, with a stage breakdown.

  D8  (discriminative, mean_pool)  : 8 imgs -> ResNet18 -> per-ray FiLM decoder
                                     -> 8192 96-bin profiles
  M3b (discriminative + per-ray)   : + deterministic ray->pixel sampling
                                     (grid_sample over 8-view multi-scale feats)
  M1  (generative VAE recon)       : prof -> latent -> prof (one reconstruction)
  M2  (generative latent DDIM)     : z ~ p(z) via 50-step DDIM -> decode

Reports median wall-clock (GPU) per object, stage breakdown, and peak VRAM.
Usage: python runtime_prof.py
"""
import json
import os
import time

import numpy as np
import torch

from d8_train_full import Model as D8Model
from d8_train_full import ray_encodings, IMG_T
from m3b_train_preray import Model as M3Model
from m3b_train_preray import compute_grid, load_views, load_from_d8
from m1_train_vae import VAE, N_BINS, load_object
from m2_train import TimeMLP, get_beta_schedule

ROOT = "/root/e0lab/e0"
D8_CKPT = os.path.join(ROOT, "d8_mean_pool")
M3B_CKPT = os.path.join(ROOT, "m3b_preray_sg2")
M1_CKPT = os.path.join(ROOT, "m1_vae_dg1024_b1e-4")
M2_DIR = os.path.join(ROOT, "m2_latent_diff")
CHUNK = 2048
DDIM_STEPS = 50


def ddim_sample(den, alpha_bar, n, dim_g, steps=DDIM_STEPS, target="v",
                eta=0.0, device="cpu"):
    """DDIM over a linear timestep grid (identical to m2_gen.ddim_sample;
    inlined here because m2_gen imports scipy, absent in this env)."""
    den.eval()
    T = len(alpha_bar)
    ts = np.linspace(T - 1, 0, steps).astype(np.int64)
    x = torch.randn(n, dim_g, device=device)
    for i, t in enumerate(ts):
        t_prev = 0 if i == steps - 1 else int(ts[i + 1])
        a_t = alpha_bar[t].item()
        a_tp = alpha_bar[t_prev].item()
        t_frac = torch.full((n,), t / (T - 1), dtype=torch.float32, device=device)
        out = den(x, t_frac)
        if target == "x0":
            x0 = out.clamp(-6.0, 6.0)
            eps = (x - np.sqrt(a_t) * x0) / np.sqrt(1.0 - a_t)
        elif target == "eps":
            eps = out
            x0 = ((x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)).clamp(-6.0, 6.0)
        else:  # v
            v = out
            x0 = (np.sqrt(a_t) * x - np.sqrt(1.0 - a_t) * v).clamp(-6.0, 6.0)
            eps = np.sqrt(1.0 - a_t) * x + np.sqrt(a_t) * v
        if i < steps - 1:
            sigma = eta * np.sqrt((1.0 - a_tp) / (1.0 - a_t)) * \
                    np.sqrt(max(0.0, 1.0 - a_t / a_tp))
            x = np.sqrt(a_tp) * x0 + np.sqrt(1.0 - a_tp - sigma ** 2) * eps \
                + sigma * torch.randn_like(x)
        else:
            x = x0
    return x


def med_time(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))  # ms


def peak_mem_mb(fn, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    device = torch.device("cuda")
    ray_pe, dirs = ray_encodings()
    ray_pe = ray_pe.to(device)
    names = json.load(open(os.path.join(D8_CKPT, "meta.json")))["val_names"][:8]

    out = {"device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "ray_res": "%dx%d" % (dirs.shape[0], N_BINS),
           "n_views": 8, "img": "224x224", "models": {}}

    # ---------- data ----------
    imgs1, feats1 = load_views(names[0])          # (8,3,224,224),(8,4)
    imgs1, feats1 = imgs1[None].to(device), feats1[None].to(device)
    grd1 = torch.from_numpy(compute_grid(names[0], dirs.numpy()))[None].to(device)
    # batch of 8 objects
    vs = [load_views(n) for n in names[:8]]
    imgs8 = torch.stack([v[0] for v in vs])[None].squeeze(0).to(device)
    feats8 = torch.stack([v[1] for v in vs]).to(device)
    grd8 = torch.stack([torch.from_numpy(compute_grid(n, dirs.numpy()))
                        for n in names[:8]]).to(device)
    # VAE input profiles (one object); load_object returns torch tensors
    prof1 = load_object(names[0])[0][None].to(device)   # (1,8192,96)

    print("data ready: imgs1%s feats1%s grd1%s imgs8%s grd8%s prof1%s"
          % (tuple(imgs1.shape), tuple(feats1.shape), tuple(grd1.shape),
             tuple(imgs8.shape), tuple(grd8.shape), tuple(prof1.shape)),
          flush=True)

    # ---------- D8 (mean_pool) ----------
    d8 = D8Model(fusion="mean_pool").to(device)
    d8.load_state_dict(torch.load(os.path.join(D8_CKPT, "model.pt"),
                                  map_location=device))
    d8.eval()

    def d8_timed(imgs, feats):
        B, K = imgs.shape[0], imgs.shape[1]
        st = torch.cuda.Event(True); mid = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        with torch.no_grad():
            st.record()
            x = d8.backbone(imgs.flatten(0, 1)).flatten(1).view(B, K, -1)
            f = torch.cat([x, feats], -1)
            g = d8.proj(f).mean(dim=1)
            mid.record()
            e = d8.pe_proj(ray_pe).unsqueeze(0).expand(B, -1, -1)
            outs = []
            for r0 in range(0, e.shape[1], CHUNK):
                h = e[:, r0:r0 + CHUNK]
                for blk in d8.decoder:
                    h = blk(h, g)
                outs.append(d8.head(h))
            torch.cat(outs, 1)
            en.record()
            torch.cuda.synchronize()
        return st.elapsed_time(mid), mid.elapsed_time(en)

    t_enc, t_dec = zip(*[d8_timed(imgs1, feats1) for _ in range(20)])
    out["models"]["d8_b1"] = {"t_enc_ms": float(np.median(t_enc)),
                              "t_dec_ms": float(np.median(t_dec)),
                              "t_total_ms": float(np.median(t_enc) + np.median(t_dec))}
    out["models"]["d8_b1"]["peak_mem_mb"] = peak_mem_mb(
        lambda: d8_timed(imgs1, feats1))
    t_enc, t_dec = zip(*[d8_timed(imgs8, feats8) for _ in range(20)])
    out["models"]["d8_b8"] = {"t_enc_ms": float(np.median(t_enc)),
                              "t_dec_ms": float(np.median(t_dec)),
                              "t_total_ms": float(np.median(t_enc) + np.median(t_dec)),
                              "per_obj_ms": float(np.median(t_enc) + np.median(t_dec)) / 8.0}
    out["models"]["d8_b8"]["peak_mem_mb"] = peak_mem_mb(
        lambda: d8_timed(imgs8, feats8))
    print("D8 done", flush=True)

    # ---------- M3b v2 (per-ray) ----------
    m3 = M3Model(rdim=128).to(device)
    load_from_d8(m3, os.path.join(D8_CKPT, "model.pt"))
    m3.load_state_dict(torch.load(os.path.join(M3B_CKPT, "head.pt"),
                                  map_location="cpu"), strict=False)
    m3.eval()

    def m3b_timed(imgs, feats, grd):
        B, K = imgs.shape[0], imgs.shape[1]
        st = torch.cuda.Event(True); mid = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        with torch.no_grad():
            st.record()
            g, rays = m3.cond_and_ray(imgs, feats, grd)
            r = torch.relu(m3.ray_proj(rays))
            mid.record()
            e = m3.pe_proj(ray_pe).unsqueeze(0).expand(B, -1, -1)
            R = r.shape[1]
            outs = []
            for r0 in range(0, R, CHUNK):
                h = e[:, r0:r0 + CHUNK]
                for blk in m3.decoder:
                    h = blk(h, g, r[:, r0:r0 + CHUNK])
                outs.append(m3.head(h))
            torch.cat(outs, 1)
            en.record()
            torch.cuda.synchronize()
        return st.elapsed_time(mid), mid.elapsed_time(en)

    t0 = time.perf_counter()
    grd1 = compute_grid(names[0], dirs.numpy())
    grid_cpu_ms = 1000.0 * (time.perf_counter() - t0)
    grd1 = torch.from_numpy(grd1)[None].to(device)
    t_enc, t_dec = zip(*[m3b_timed(imgs1, feats1, grd1) for _ in range(20)])
    out["models"]["m3b_b1"] = {"t_cond_ms": float(np.median(t_enc)),
                               "t_dec_ms": float(np.median(t_dec)),
                               "t_total_ms": float(np.median(t_enc) + np.median(t_dec)),
                               "t_grid_cpu_ms": grid_cpu_ms}
    out["models"]["m3b_b1"]["peak_mem_mb"] = peak_mem_mb(
        lambda: m3b_timed(imgs1, feats1, grd1))
    t_enc, t_dec = zip(*[m3b_timed(imgs8, feats8, grd8) for _ in range(20)])
    out["models"]["m3b_b8"] = {"t_cond_ms": float(np.median(t_enc)),
                               "t_dec_ms": float(np.median(t_dec)),
                               "t_total_ms": float(np.median(t_enc) + np.median(t_dec)),
                               "per_obj_ms": float(np.median(t_enc) + np.median(t_dec)) / 8.0}
    out["models"]["m3b_b8"]["peak_mem_mb"] = peak_mem_mb(
        lambda: m3b_timed(imgs8, feats8, grd8))
    print("M3b done", flush=True)

    # ---------- M1 VAE recon ----------
    m1 = VAE(dim_g=1024).to(device)
    m1.load_state_dict(torch.load(os.path.join(M1_CKPT, "model.pt"),
                                  map_location=device))
    m1.eval()
    torch.manual_seed(0)

    def m1_recon(prof):
        with torch.no_grad():
            mu, lv = m1.encode(prof)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            m1.decode(z, ray_pe)

    t_encs, t_decs = [], []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            mu, lv = m1.encode(prof1)
        torch.cuda.synchronize()
        t_encs.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            m1.decode(mu + torch.exp(0.5 * lv) * torch.randn_like(mu), ray_pe)
        torch.cuda.synchronize()
        t_decs.append(time.perf_counter() - t0)
    out["models"]["m1"] = {"t_enc_ms": 1000.0 * float(np.median(t_encs)),
                           "t_dec_ms": 1000.0 * float(np.median(t_decs)),
                           "t_total_ms": 1000.0 * (float(np.median(t_encs)) + float(np.median(t_decs)))}
    out["models"]["m1"]["peak_mem_mb"] = peak_mem_mb(lambda: m1_recon(prof1))
    print("M1 done", flush=True)

    # ---------- M2 latent DDIM (50 steps) ----------
    m2 = json.load(open(os.path.join(M2_DIR, "meta.json")))
    dim_g = m2["dim_g"]
    _, _, alpha_bar = get_beta_schedule(m2["T"])
    den = TimeMLP(dim_g).to(device)
    den.load_state_dict(torch.load(os.path.join(M2_DIR, "denoiser.pt"),
                                   map_location="cpu"))
    z_mean = torch.from_numpy(
        np.load(os.path.join(M2_DIR, "z_mean.npy"))).to(device)
    z_std = torch.from_numpy(
        np.load(os.path.join(M2_DIR, "z_std.npy"))).to(device)

    def m2_sample(n=1):
        with torch.no_grad():
            z_white = ddim_sample(den, alpha_bar, n, dim_g, DDIM_STEPS, "v",
                                  0.0, device)
            m1.decode(z_white * z_std + z_mean, ray_pe)

    # correctness first: run the real DDIM once to make sure it executes
    with torch.no_grad():
        z_white = ddim_sample(den, alpha_bar, 1, dim_g, DDIM_STEPS, "v", 0.0,
                              device)
        m1.decode(z_white * z_std + z_mean, ray_pe)
    torch.cuda.synchronize()
    t_diff = med_time(lambda: ddim_sample(den, alpha_bar, 1, dim_g, DDIM_STEPS,
                                          "v", 0.0, device))
    t_dec = med_time(lambda: m1.decode(
        (torch.randn(1, dim_g, device=device)) * z_std + z_mean, ray_pe))
    out["models"]["m2"] = {"t_diff_50ms": t_diff, "t_dec_ms": t_dec,
                           "t_total_ms": t_diff + t_dec}
    out["models"]["m2"]["peak_mem_mb"] = peak_mem_mb(m2_sample)
    print("M2 done", flush=True)

    os.makedirs(os.path.join(ROOT, "runtime_prof"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runtime_prof", "summary.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))
    print("saved -> /root/e0lab/e0/runtime_prof/summary.json", flush=True)


if __name__ == "__main__":
    main()
