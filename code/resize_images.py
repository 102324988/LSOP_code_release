"""resize_images.py: 批处理缩放真实照片，供 COLMAP SfM 与 3DGS 训练使用。

真实手机/相机照片常为 3-4K，直接 SfM（CPU colmap）与 GOF 训练都偏慢。
统一缩到最大边 max_side 像素（默认 1600），输出到 <src>/images_resized/ 或指定目录。

用法：python resize_images.py <src_dir> [--max_side 1600] [--out DIR]
"""
import argparse
import os

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src_dir")
ap.add_argument("--max_side", type=int, default=1600)
ap.add_argument("--out", default=None)
args = ap.parse_args()

exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
out = args.out or os.path.join(args.src_dir, "images_resized")
os.makedirs(out, exist_ok=True)

files = sorted(f for f in os.listdir(args.src_dir)
               if f.lower().endswith(exts))
if not files:
    # 若 src 内还有子目录（如 images/），递归一层
    for sub in sorted(os.listdir(args.src_dir)):
        p = os.path.join(args.src_dir, sub)
        if os.path.isdir(p):
            files += sorted(os.path.join(sub, f) for f in os.listdir(p)
                            if f.lower().endswith(exts))
    if not files:
        raise SystemExit(f"no images found in {args.src_dir}")

print(f"[resize] {len(files)} images -> max_side {args.max_side} -> {out}", flush=True)
for f in files:
    src = os.path.join(args.src_dir, f)
    dst = os.path.join(out, os.path.basename(f))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = max(w, h)
    if s > args.max_side:
        nw, nh = (round(w * args.max_side / s), round(h * args.max_side / s))
        im = im.resize((nw, nh), Image.LANCZOS)
    im.save(dst, quality=92)
print("[resize] done")
