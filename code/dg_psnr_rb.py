"""E0: PSNR of random-bg model with CORRECT per-view bg."""
import argparse, os, sys, json
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from gaussian_renderer import render

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--views", type=int, default=12)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

with open(os.path.join(ds.source_path, "bg.json")) as f:
    bgmap = {k.rsplit(".",1)[0]: [float(v) for v in val] for k,val in json.load(f).items()}

def psnr(a, b):
    mse = ((a-b)**2).mean()
    return 10*np.log10(1.0/mse)

for which, bguse in [("correct-bg", True), ("white-bg", False)]:
    psnrs = []
    for i, cam in enumerate(scene.getTrainCameras()):
        if i >= args.views: break
        gt = cam.original_image.cuda()
        if bguse:
            bgc = torch.tensor(bgmap[cam.image_name], device="cuda")
        else:
            bgc = torch.ones(3, device="cuda")
        out = render(cam, g, pipe, bg_color=bgc, kernel_size=ds.kernel_size)
        img = out["render"][:3].clamp(0,1)
        psnrs.append(psnr(img.detach().cpu().numpy(), gt.detach().cpu().numpy()))
    print(f"{which}: PSNR={np.mean(psnrs):.2f} dB over {len(psnrs)} views")
