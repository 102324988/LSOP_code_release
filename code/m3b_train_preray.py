#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3b: per-ray image-conditioned decoder (morphology refinement).

Goal: sharpen profiles from the D8/global-cond level (FWHM ~2x, peak 0.59)
toward GT (FWHM 1.0x, peak 1.0) by giving each ray DIRECT image evidence.

Mechanism: ray direction d (world) * r_ref -> 8-view pixel (COLMAP K, R, t)
-> bilinear sample frozen multi-scale ResNet18 (layer2/3/4) -> view-averaged
concat (896) -> FiLM-injected into the D8 decoder as a second channel.

TWO-STAGE execution (the naive single-stage ran backbone+grid_sample on
every iteration: ~8 min/epoch -> 40 epochs unaffordable):
  --stage prep   freeze backbone+proj (D8 weights), extract per-object
                 global cond g (512) and 3-scale per-ray concat (8192,896,
                 fp16) to disk.
  --stage train  light head (ray_proj + per-ray FiLM decoder), D8-decoder
                 init, pure v3 L1, features loaded from disk.

Init from D8 (val dmeds 0.0338): FiLMBlock2 uses gamma_g/beta_g (loaded from
D8 gamma/beta via key remap); gamma_r/beta_r/ray_proj start at zero output ->
per-ray channel inert at init, must earn its way in (collapse guard).

Gate: val FWHM ratio <= 1.5, peak >= 0.5, dmeds within 15% of D8 (<=0.039).

Usage:
  python m3b_train_preray.py --stage prep --out m3b_preray
  python m3b_train_preray.py --stage train --out m3b_preray [--epochs 40]
  python m3b_train_preray.py --stage all  --subset 24 --epochs 2   # smoke
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from m1_train_vae import N_BINS, PE_DIM, load_object, ray_encodings
from d8_train_full import IMG_T

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D8_CKPT = "d8_mean_pool"
FIXED_VIEWS = list(range(0, 48, 6))            # 8 eval-protocol views
K = len(FIXED_VIEWS)
W, H = 800, 600                                # render resolution (COLMAP cam)
R_REF = 1.0


def quat2rot(qw, qx, qy, qz):
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def load_cams(rd):
    sd = os.path.join(rd, "sparse", "0")
    fx = fy = cx = cy = None
    for line in open(os.path.join(sd, "cameras.txt")):
        p = line.split()
        if len(p) >= 8 and p[1] == "PINHOLE":
            fx, fy, cx, cy = map(float, p[4:8])
    cams = {}
    lines = open(os.path.join(sd, "images.txt")).read().splitlines()
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        vid = int(p[9].split("_")[1].split(".")[0])
        qw, qx, qy, qz = map(float, p[1:5])
        t = np.array(list(map(float, p[5:8])))
        cams[vid] = (quat2rot(qw, qx, qy, qz), t)
    return (fx, fy, cx, cy), cams


def ray_pixel_grid(dirs, K4, Rc, t, r_ref=R_REF):
    """dirs (R,3) -> (R,2) normalized pixel coords [0,1]."""
    fx, fy, cx, cy = K4
    X = dirs * r_ref
    xc = X @ Rc.T + t
    z = xc[:, 2]
    u = fx * xc[:, 0] / z + cx
    v = fy * xc[:, 1] / z + cy
    return np.stack([u / W, v / H], -1).astype(np.float32)


def load_views(name):
    rd = os.path.join(RENDERS, name)
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    imgs, feats = [], []
    for i in FIXED_VIEWS:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    return torch.stack(imgs), torch.tensor(feats, dtype=torch.float32)


def compute_grid(name, dirs):
    rd = os.path.join(RENDERS, name)
    K4, cams = load_cams(rd)
    gs = np.zeros((dirs.shape[0], K, 2), dtype=np.float32)
    for j, i in enumerate(FIXED_VIEWS):
        Rc, t = cams[i]
        gs[:, j] = ray_pixel_grid(dirs, K4, Rc, t)
    return gs


class FiLMBlock2(nn.Module):
    """D8 FiLMBlock + a per-ray channel: gamma = gamma_g(g) + gamma_r(r)."""

    def __init__(self, din, dout, gdim, rdim):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.gamma_g = nn.Linear(gdim, dout)
        self.beta_g = nn.Linear(gdim, dout)
        self.gamma_r = nn.Linear(rdim, dout)
        self.beta_r = nn.Linear(rdim, dout)

    def forward(self, x, g, r):
        h = torch.relu(self.fc(x))
        gam = self.gamma_g(g).unsqueeze(1) + self.gamma_r(r)
        bet = self.beta_g(g).unsqueeze(1) + self.beta_r(r)
        return h * gam + bet


class Model(nn.Module):
    """Full per-ray model (backbone + proj + ray_proj + FiLM2 decoder).

    Used for --stage prep (feature extraction, D8-loaded, frozen) and for
    inference (load decoder/ray_proj from the trained head ckpt).
    """

    def __init__(self, gdim=512, rdim=128):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # D8 keys
        self.proj = nn.Linear(512 + 4, gdim)
        self.ray_proj = nn.Linear(128 + 256 + 512, rdim)
        self.pe_proj = nn.Linear(PE_DIM, 64)
        self.decoder = nn.Sequential(
            FiLMBlock2(64, 512, gdim, rdim),
            FiLMBlock2(512, 512, gdim, rdim),
            FiLMBlock2(512, 256, gdim, rdim))
        self.head = nn.Linear(256, N_BINS)

    def multi_feats(self, x):
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        x = self.backbone[4](x)
        x = self.backbone[5](x)                       # layer2 -> /8 (128)
        f2 = x
        x = self.backbone[6](x)                       # layer3 -> /16 (256)
        f3 = x
        x = self.backbone[7](x)                       # layer4 -> /32 (512)
        f4 = x
        return f2, f3, f4

    @staticmethod
    def _samp_scale(feat, grids, B, K):
        """feat (B,K,C,H,W), grids (B,R,K,2) -> (B,R,C) view-averaged."""
        R = grids.shape[1]
        outs = []
        for k in range(K):
            gk = (2.0 * grids[:, :, k] - 1.0).unsqueeze(1)
            s = F.grid_sample(feat[:, k], gk, align_corners=False)
            outs.append(s.squeeze(2))
        return torch.stack(outs, 1).mean(1).permute(0, 2, 1)    # (B,R,C)

    def cond_and_ray(self, imgs, feats, grids):
        """One backbone pass -> (g, per-ray 3-scale concat (B,R,896))."""
        B, K = imgs.shape[0], imgs.shape[1]
        f2, f3, f4 = self.multi_feats(imgs.flatten(0, 1))
        def vb(f):
            return f.view(B, K, *f.shape[1:])
        f4v = vb(f4)
        gx = f4v.mean((-2, -1))                                # (B,K,512)
        g = self.proj(torch.cat([gx, feats], -1)).mean(1)      # (B,gdim)
        rays = torch.cat([self._samp_scale(vb(f2), grids, B, K),
                          self._samp_scale(vb(f3), grids, B, K),
                          self._samp_scale(vb(f4), grids, B, K)], -1)  # (B,R,896)
        return g, rays

    def forward(self, imgs, feats, ray_pe, grids, chunk=2048):
        B, K = imgs.shape[0], imgs.shape[1]
        g, rays = self.cond_and_ray(imgs, feats, grids)
        r = torch.relu(self.ray_proj(rays))
        return decode_head(self, g, r, ray_pe, B, chunk)


def decode_head(mod, g, r, ray_pe, B, chunk):
    e = mod.pe_proj(ray_pe).unsqueeze(0).expand(B, -1, -1)
    R = r.shape[1]
    outs = []
    for r0 in range(0, R, chunk):
        h = e[:, r0:r0 + chunk]
        for blk in mod.decoder:
            h = blk(h, g, r[:, r0:r0 + chunk])
        outs.append(mod.head(h))
    return torch.cat(outs, 1)                                  # (B,R,96)


class Model2(nn.Module):
    """Light training head (no backbone): ray_proj + FiLM2 decoder + head."""

    def __init__(self, gdim=512, rdim=128):
        super().__init__()
        self.ray_proj = nn.Linear(128 + 256 + 512, rdim)
        self.pe_proj = nn.Linear(PE_DIM, 64)
        self.decoder = nn.Sequential(
            FiLMBlock2(64, 512, gdim, rdim),
            FiLMBlock2(512, 512, gdim, rdim),
            FiLMBlock2(512, 256, gdim, rdim))
        self.head = nn.Linear(256, N_BINS)

    def forward(self, g, rays, ray_pe, chunk=2048):
        B = g.shape[0]
        r = torch.relu(self.ray_proj(rays))                    # (B,R,rdim)
        return decode_head(self, g, r, ray_pe, B, chunk)


def load_from_d8(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    remap = {}
    for k in sd:
        if ".gamma" in k and ".gamma_" not in k:
            remap[k] = k.replace(".gamma", ".gamma_g")
        elif ".beta" in k and ".beta_" not in k:
            remap[k] = k.replace(".beta", ".beta_g")
    for a, b in remap.items():
        sd[b] = sd.pop(a)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("D8 load: missing=%d (per-ray/backbone-free keys, fresh) unexpected=%d"
          % (len(missing), len(unexpected)), flush=True)


def fwhm_np(prof):
    pk = prof.argmax(-1)
    hm = prof.max(-1) * 0.5
    w = np.zeros(len(prof), dtype=np.float32)
    for i in range(len(prof)):
        p = prof[i]
        lo = hi = pk[i]
        while lo > 0 and p[lo - 1] >= hm[i]:
            lo -= 1
        while hi < N_BINS - 1 and p[hi + 1] >= hm[i]:
            hi += 1
        w[i] = hi - lo
    return w


def eval_head(m2, va_feat, ray_pe, device, names):
    m2.eval()
    axis = np.arange(N_BINS)
    dmeds, fwhm_rat, peaks = [], [], []
    with torch.no_grad():
        for n in names:
            g, rays, sh, cov, peak, rmax = va_feat[n]
            g = torch.from_numpy(g)[None].to(device)
            rays = torch.from_numpy(rays)[None].float().to(device)
            pred = torch.sigmoid(m2(g, rays, ray_pe))[0].cpu().numpy()
            m = cov >= 0.02
            soft = (pred[m] * axis[None, :]).sum(-1) / (pred[m].sum(-1) + 1e-6)
            gtn = peak[m] / rmax
            dmeds.append(np.median(np.abs(soft / (N_BINS - 1) - gtn)))
            fw_p = fwhm_np(pred[m])
            fw_g = fwhm_np(sh[m])
            good = fw_g >= 2
            fwhm_rat.append(fw_p[good] / np.maximum(fw_g[good], 1e-6))
            peaks.append(pred[m].max(-1))
    m2.train()
    fr = np.concatenate(fwhm_rat)
    return {"dmeds_med": float(np.median(dmeds)),
            "dmeds_mean": float(np.mean(dmeds)),
            "fwhm_ratio_med": float(np.median(fr)),
            "peak_med": float(np.median(np.concatenate(peaks)))}


def prep_features(model, names, out, device, dirs_np, ray_pe):
    feat_dir = os.path.join(out, "feat")
    os.makedirs(feat_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, n in enumerate(names):
            imgs, feats = load_views(n)
            grids = compute_grid(n, dirs_np)
            imgs = imgs[None].to(device)
            feats = feats[None].to(device)
            grd = torch.from_numpy(grids)[None].to(device)
            g, rays = model.cond_and_ray(imgs, feats, grd)     # (1,512), (1,R,896)
            np.save(os.path.join(feat_dir, n + "_g.npy"),
                    g[0].cpu().numpy())
            np.save(os.path.join(feat_dir, n + "_rays.npy"),
                    rays[0].cpu().numpy().astype(np.float16))
            if (i + 1) % 50 == 0 or i == len(names) - 1:
                print("prep %d/%d (%s)" % (i + 1, len(names), n), flush=True)


def main():
    global R_REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "train", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--r_ref", type=float, default=R_REF)
    ap.add_argument("--out", type=str, default="m3b_preray")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_REF = args.r_ref

    d8meta = json.load(open(os.path.join(ROOT, D8_CKPT, "meta.json")))
    train_names, val_names = d8meta["train_names"], d8meta["val_names"]
    if args.subset:
        train_names = train_names[:args.subset]
        val_names = val_names[:min(args.subset, len(val_names))]

    ray_pe, dirs = ray_encodings()
    dirs_np = dirs.numpy()
    ray_pe = ray_pe.to(device)
    out_dir = os.path.join(ROOT, args.out)

    if args.stage in ("prep", "all"):
        print("== stage prep (%d+%d objects) ==" % (len(train_names),
                                                     len(val_names)), flush=True)
        model = Model().to(device)
        load_from_d8(model, os.path.join(ROOT, D8_CKPT, "model.pt"))
        for p in model.parameters():
            p.requires_grad_(False)
        os.makedirs(out_dir, exist_ok=True)
        prep_features(model, train_names, out_dir, device, dirs_np, ray_pe)
        prep_features(model, val_names, out_dir, device, dirs_np, ray_pe)

    if args.stage in ("train", "all"):
        print("== stage train (%d train / %d val) ==" % (len(train_names),
                                                          len(val_names)),
              flush=True)
        feat_dir = os.path.join(out_dir, "feat")

        def _load_feats(names):
            from multiprocessing.dummy import Pool

            def one(n):
                g = np.load(os.path.join(feat_dir, n + "_g.npy"))
                rays = np.load(os.path.join(feat_dir, n + "_rays.npy"))
                sh, cov, peak, rmax = load_object(n)
                return (n, (g, rays, sh.numpy(), cov.numpy(), peak.numpy(),
                            rmax))
            with Pool(8) as pool:
                d = dict(pool.map(one, names, chunksize=4))
            print("feat loaded (%d)." % len(names), flush=True)
            return d

        tr_feat = _load_feats(train_names)
        va_feat = _load_feats(val_names)

        m2 = Model2().to(device)
        load_from_d8(m2, os.path.join(ROOT, D8_CKPT, "model.pt"))
        n_params = sum(p.numel() for p in m2.parameters()) / 1e6
        print("head params=%.1fM" % n_params, flush=True)

        # baseline: D8 decoder with per-ray channel inert (g only)
        base = eval_head(m2, va_feat, ray_pe, device, val_names)
        print("head init (D8-decoder, per-ray off): dmeds=%.4f fwhm_ratio=%.2f "
              "peak=%.3f" % (base["dmeds_med"], base["fwhm_ratio_med"],
                             base["peak_med"]), flush=True)

        opt = torch.optim.Adam(m2.parameters(), lr=args.lr)
        lr_min = args.lr * 0.05
        names_list = list(tr_feat.keys())
        for ep in range(args.epochs):
            m2.train()
            tot, nb = 0.0, 0
            lr = lr_min + 0.5 * (args.lr - lr_min) * \
                (1.0 + np.cos(np.pi * ep / args.epochs))
            for g_ in opt.param_groups:
                g_["lr"] = lr
            order = np.random.permutation(len(names_list))
            for b0 in range(0, len(names_list), args.batch):
                bn = [names_list[i] for i in order[b0:b0 + args.batch]]
                g = torch.from_numpy(
                    np.stack([tr_feat[n][0] for n in bn])).to(device)
                rays = torch.from_numpy(
                    np.stack([tr_feat[n][1] for n in bn])).float().to(device)
                sh = torch.from_numpy(
                    np.stack([tr_feat[n][2] for n in bn])).to(device)
                cov = torch.from_numpy(
                    np.stack([tr_feat[n][3] for n in bn])).to(device)
                pred = torch.sigmoid(m2(g, rays, ray_pe))
                w = (cov + 0.05).clamp(max=1.0).unsqueeze(-1)
                loss = (w * (pred - sh).abs()).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            if ep == 0 or (ep + 1) % args.eval_every == 0 or \
                    (ep + 1) == args.epochs:
                r = eval_head(m2, va_feat, ray_pe, device, val_names)
                print("epoch %3d loss=%.5f lr=%.1e | val dmeds=%.4f "
                      "fwhm_ratio=%.2f peak=%.3f"
                      % (ep + 1, tot / nb, lr, r["dmeds_med"],
                         r["fwhm_ratio_med"], r["peak_med"]), flush=True)

        os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
        torch.save(m2.state_dict(), os.path.join(ROOT, args.out, "head.pt"))
        with open(os.path.join(ROOT, args.out, "meta.json"), "w") as f:
            json.dump({"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                       "r_ref": args.r_ref, "rdim": 128, "loss": "v3-L1",
                       "init": D8_CKPT, "n_train": len(train_names),
                       "n_val": len(val_names), "train_names": train_names,
                       "val_names": val_names}, f)
        print("done. -> %s/head.pt" % args.out, flush=True)


if __name__ == "__main__":
    main()
