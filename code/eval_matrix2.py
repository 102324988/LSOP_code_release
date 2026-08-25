"""E0: eval matrix v2 — inversion accuracy via TRUE GT-surface distance
(nearest-neighbor on the GT mesh), works for any object shape."""
import argparse, os, sys, json
import numpy as np, torch
import trimesh
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
ap.add_argument("--peak", choices=["all", "first", "p"], default="all",
                help="all: every o(r) local max is a hit; first: only the ray's "
                     "first max; p: first peak of P(r)=T*o (visibility-weighted)")
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)

mesh = trimesh.load(os.path.join(ds.source_path, "..", "meshes", args.obj + ".ply"),
                    force="mesh")
mesh_v = mesh.vertices.astype(np.float64)
mesh_tri = trimesh.Trimesh(vertices=mesh_v, faces=mesh.faces)
tree = mesh_tri.triangles_tree

def dist_to_surface(points):
    closest, dist, tri = mesh_tri.nearest.on_surface(np.asarray(points, dtype=float))
    return dist

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
with open(os.path.join(ds.source_path, "bg.json")) as f:
    bgmap = {k.rsplit(".",1)[0]: [float(v) for v in val] for k,val in json.load(f).items()}

def psnr(a, b):
    return 10*np.log10(1.0/((a-b)**2).mean())

psnrs = []
for i, cam in enumerate(scene.getTrainCameras()):
    if i >= args.views: break
    gt = cam.original_image.cuda()
    bgc = torch.tensor(bgmap[cam.image_name], device="cuda")
    out = render(cam, g, pipe, bg_color=bgc, kernel_size=ds.kernel_size)
    img = out["render"][:3].clamp(0,1)
    psnrs.append(psnr(img.detach().cpu().numpy(), gt.detach().cpu().numpy()))
psnr_avg = float(np.mean(psnrs))

scl = torch.exp(g._scaling.detach()); smax = scl.max(1).values
pos = g._xyz.detach().cpu().numpy()

els = [20,45,70]; azs = np.linspace(0,360,6,endpoint=False); Rcam=3.2
tgt_r = np.linspace(0.0, 0.78, 13); tgt_z = np.linspace(-0.30, 0.30, 13)
pts=[]; hits=0
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
                arr = P if args.peak == "p" else o
                thr=max(0.01,0.05*arr.max())
                for i in range(1,len(arr)-1):
                    if arr[i]>arr[i-1] and arr[i]>=arr[i+1] and arr[i]>thr:
                        pt=cam+(dvec/np.linalg.norm(dvec))*t[i]
                        if np.abs(pt).max()<1.2:
                            pts.append(pt); hits+=1
                        if args.peak!="all":
                            break
pts=np.array(pts)
dist = dist_to_surface(pts) if hits else np.array([])
res = dict(obj=args.obj, model=os.path.basename(args.model_path),
           peak=args.peak,
           gaussians=int(g._xyz.shape[0]), scale_max=float(smax.max()),
           scale_p99=float(smax.quantile(0.99)), psnr=round(psnr_avg,2),
           hits=hits,
           inv_p50=float(np.median(dist)) if hits else None,
           inv_p90=float(np.percentile(dist,90)) if hits else None,
           inv_frac01=float(np.mean(dist<0.1)) if hits else None,
           inv_frac02=float(np.mean(dist<0.2)) if hits else None)
print(json.dumps(res, ensure_ascii=False))
