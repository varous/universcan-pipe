"""Synthetic 3DGS .ply: the venue as Gaussians + floaters + blobs to test filtering."""
import numpy as np
from plyfile import PlyData, PlyElement

rng = np.random.default_rng(11)
def plane(n,p0,u,v,su,sv,noise=0.01):
    a=rng.uniform(0,su,n);b=rng.uniform(0,sv,n)
    return p0+np.outer(a,u)+np.outer(b,v)+rng.normal(0,noise,(n,3))

surf=[]
surf.append(plane(20000,[0,0,0],[1,0,0],[0,1,0],20,15))   # floor
surf.append(plane(15000,[0,0,8],[1,0,0],[0,1,0],20,15))   # ceiling
surf.append(plane(10000,[0,0,0],[0,1,0],[0,0,1],15,8))    # left
surf.append(plane(10000,[20,0,0],[0,1,0],[0,0,1],15,8))   # right
surf.append(plane(10000,[0,15,0],[1,0,0],[0,0,1],20,8))   # rear
n=12000;x=rng.uniform(2,18,n);y=rng.uniform(4,14,n);z=0.1+(y-4)*0.18
surf.append(np.stack([x,y,z],1)+rng.normal(0,0.01,(n,3)))  # rake
surf.append(plane(5000,[4,0,1.0],[1,0,0],[0,1,0],12,3.5))  # stage
real=np.vstack(surf)

# floaters: scattered mid-air, LOW opacity (should be dropped)
floaters=rng.uniform([0,0,1],[20,15,7],(6000,3))
# blobs: few, huge scale background (should be dropped)
blobs=rng.uniform([-5,-5,-2],[25,20,10],(800,3))

xyz=np.vstack([real,floaters,blobs]).astype(np.float32)
nR,nF,nB=len(real),len(floaters),len(blobs)
N=len(xyz)

# opacity (raw logit): real high (+4 -> sig~0.98), floaters low (-3 -> ~0.05), blobs mid
op=np.concatenate([np.full(nR,4.0),np.full(nF,-3.0),np.full(nB,1.0)]).astype(np.float32)
# log-scale: real small (exp(-4)=0.018), floaters small, blobs large (exp(1.2)=3.3)
sc_real=np.full((nR,3),-4.0);sc_fl=np.full((nF,3),-3.5);sc_bl=np.full((nB,3),1.2)
scale=np.vstack([sc_real,sc_fl,sc_bl]).astype(np.float32)
rot=np.tile([1,0,0,0],(N,1)).astype(np.float32)
fdc=rng.uniform(-1,1,(N,3)).astype(np.float32)

dt=[('x','f4'),('y','f4'),('z','f4'),
    ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
    ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
    ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),('opacity','f4')]
a=np.zeros(N,dtype=dt)
a['x'],a['y'],a['z']=xyz[:,0],xyz[:,1],xyz[:,2]
for i in range(3):a[f'scale_{i}']=scale[:,i]
for i in range(4):a[f'rot_{i}']=rot[:,i]
for i in range(3):a[f'f_dc_{i}']=fdc[:,i]
a['opacity']=op
PlyData([PlyElement.describe(a,'vertex')],text=False).write('synth_splat.ply')
print(f"wrote synth_splat.ply  total={N:,}  (real={nR:,} floaters={nF:,} blobs={nB:,})")
