"""E0: full evaluation matrix per model — render PSNR (correct bg), inversion
accuracy vs GT, plus per-object stability across two independent runs.

Usage: eval_matrix.py -s data/<obj>_rb -m output/<obj>_rb_<seed> --obj <obj>
Writes JSON summary to stdout.
"""
import argparse, os, sys, json
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from gaussian_renderer import render
from clean_ray_profile import ray_march

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--obj", required=True)
ap.add_argument("--views", type=int, default=12)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)

# GT bbox extent (approx, for ray targets + hit filtering)
GT = {"torus": dict(R=0.55, r=0.23), "sphere": dict(R=0.4), "vase": dict(R=0.4),
      "rocky": dict(R=0.45), "mug": dict(R=0.4), "bumpy": dict(R=0.45)}

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

with open(os.path.join(ds.source_path, "bg.json")) as f:
    bgmap = {k.rsplit(".",1)[0]: [float(v) for v in val] for k,val in json.load(f).items()}

def psnr(a, b):
    return 10*np.log10(1.0/((a-b)**2).mean())

# 1) render PSNR with correct per-view bg
psnrs = []
for i, cam in enumerate(scene.getTrainCameras()):
    if i >= args.views: break
    gt = cam.original_image.cuda()
    bgc = torch.tensor(bgmap[cam.image_name], device="cuda")
    out = render(cam, g, pipe, bg_color=bgc, kernel_size=ds.kernel_size)
    img = out["render"][:3].clamp(0,1)
    psnrs.append(psnr(img.detach().cpu().numpy(), gt.detach().cpu().numpy()))
psnr_avg = float(np.mean(psnrs))

# 2) scale / ghost diagnostics
scl = torch.exp(g._scaling.detach())
smax = scl.max(1).values
pos = g._xyz.detach().cpu().numpy()
rxy = np.sqrt(pos[:,0]**2+pos[:,1]**2)
ghost = int(((rxy < 0.32) & (np.abs(pos[:,2]) < 0.23)).sum()) if args.obj == "torus" else -1

# 3) inversion: rays from 3 els x 6 azs x 13x13 targets; hit = o local max
def dist_surf(p, obj):
    if obj == "torus":
        R, r = 0.55, 0.23
        rxy = np.sqrt(p[...,0]**2+p[...,1]**2)
        return np.abs(np.sqrt((rxy-R)**2+p[...,2]**2)-r)
    R = GT[obj]["R"]
    return np.abs(np.linalg.norm(p, axis=-1) - R)  # sphere-ish approx

els = [20,45,70]; azs = np.linspace(0,360,6,endpoint=False); Rcam=3.2
tgt_r = np.linspace(0.0, 0.78, 13); tgt_z = np.linspace(-0.30, 0.30, 13)
errs=[]; hits=0
for el_deg in els:
    el=np.deg2rad(el_deg)
    for az_deg in azs:
        az=np.deg2rad(az_deg)
        cam=np.array([Rcam*np.cos(el)*np.cos(az),Rcam*np.cos(el)*np.sin(az),Rcam*np.sin(el)])
        ct,stt=np.cos(az),np.sin(az)
        for tr in tgt_r:
            for tz in tgt_z:
                tgtr=np.array([tr*ct,tr*stt,tz])
                dvec=tgtr-cam
                t,o,Tt,P=ray_march(torch.tensor(cam,dtype=torch.float32,device="cuda"),
                                   torch.tensor(dvec,dtype=torch.float32,device="cuda"),
                                   g,0.02,7.0,K=1200)
                thr=max(0.01,0.05*o.max())
                for i in range(1,len(o)-1):
                    if o[i]>o[i-1] and o[i]>=o[i+1] and o[i]>thr:
                        pt=cam+(dvec/np.linalg.norm(dvec))*t[i]
                        if np.abs(pt).max()<1.2:
                            errs.append(dist_surf(pt,args.obj)); hits+=1
errs=np.array(errs)
res = dict(obj=args.obj, model=os.path.basename(args.model_path),
           gaussians=int(g._xyz.shape[0]), scale_max=float(smax.max()),
           scale_p99=float(smax.quantile(0.99)), ghost_hole=ghost,
           psnr=round(psnr_avg,2), hits=hits,
           inv_p50=float(np.median(errs)) if hits else None,
           inv_p90=float(np.percentile(errs,90)) if hits else None,
           inv_frac01=float(np.mean(errs<0.1)) if hits else None,
           inv_frac02=float(np.mean(errs<0.2)) if hits else None)
print(json.dumps(res, ensure_ascii=False))
