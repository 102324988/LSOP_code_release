"""E0: render PSNR (train views) with vs without hole_cull."""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from gaussian_renderer import render
from cull_cloud import cull_gaussians

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--views", type=int, default=12)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)

def psnr(a, b):
    mse = ((a-b)**2).mean()
    return 10*np.log10(1.0/mse)

for hole_cull in [False, True]:
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    keep, st = cull_gaussians(g, 0.2, 0.3, hole_cull=hole_cull)
    k, t = st["keep"], st["total"]
    print(f"hole_cull={hole_cull}: keep={k}/{t}")
    psnrs = []
    for i, cam in enumerate(scene.getTrainCameras()):
        if i >= args.views: break
        gt = cam.original_image.cuda()
        out = render(cam, g, pipe, bg_color=torch.ones(3,device="cuda"),
                     kernel_size=ds.kernel_size, scaling_modifier=1.0)
        img = out["render"][:3].clamp(0,1)
        psnrs.append(psnr(img.detach().cpu().numpy(), gt.detach().cpu().numpy()))
    a = np.mean(psnrs)
    print(f"  PSNR over {len(psnrs)} train views: {a:.2f} dB")
