"""Synthetic EXTRACTED-centres cloud with the hard case:
   dense gym + smaller dense annex + wide sparse spray (floaters)."""
import numpy as np
from plyfile import PlyData, PlyElement
rng=np.random.default_rng(5)

def box_surfaces(cx,cy,w,d,h,n,noise=0.02):
    pts=[]
    # floor + ceiling
    for z in (0,h):
        a=rng.uniform(-w/2,w/2,n);b=rng.uniform(-d/2,d/2,n)
        pts.append(np.stack([cx+a,cy+b,np.full(n,z)],1))
    # 4 walls
    for sx in (-w/2,w/2):
        b=rng.uniform(-d/2,d/2,n);z=rng.uniform(0,h,n)
        pts.append(np.stack([np.full(n,cx+sx),cy+b,z],1))
    for sy in (-d/2,d/2):
        a=rng.uniform(-w/2,w/2,n);z=rng.uniform(0,h,n)
        pts.append(np.stack([cx+a,np.full(n,cy+sy),z],1))
    P=np.vstack(pts); return P+rng.normal(0,noise,P.shape)

gym  = box_surfaces(0,0,   40,25,10, 12000)   # dense main hall
annex= box_surfaces(70,50, 18,14,6,   4000)   # smaller dense second region (separated)
# wide sparse spray across the whole 175x148x48 volume (floaters)
spray= rng.uniform([-96,-82,-8],[80,66,40],(60000,3))

xyz=np.vstack([gym,annex,spray]).astype(np.float32)
v=np.zeros(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4')])
v['x'],v['y'],v['z']=xyz[:,0],xyz[:,1],xyz[:,2]
PlyData([PlyElement.describe(v,'vertex')],text=False).write('synth_noisy.ply')
print(f"wrote synth_noisy.ply  total={len(xyz):,}  gym={len(gym):,} annex={len(annex):,} spray={len(spray):,}")
print(f"full extent: {(xyz.max(0)-xyz.min(0)).round(1).tolist()}")
