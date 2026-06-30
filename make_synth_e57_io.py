"""Synthetic E57: indoor room + outdoor ground at a global (UTM-like) offset."""
import numpy as np, pye57
rng=np.random.default_rng(4)
def plane(n,p0,u,v,su,sv,noise=0.008):
    a=rng.uniform(0,su,n);b=rng.uniform(0,sv,n)
    return p0+np.outer(a,u)+np.outer(b,v)+rng.normal(0,noise,(n,3))
# room 10x8x3 (enclosed) — dense
R=[plane(80000,[10,11,0],[1,0,0],[0,1,0],10,8),plane(60000,[10,11,3],[1,0,0],[0,1,0],10,8),
   plane(40000,[10,11,0],[0,1,0],[0,0,1],8,3),plane(40000,[20,11,0],[0,1,0],[0,0,1],8,3),
   plane(40000,[10,11,0],[1,0,0],[0,0,1],10,3),plane(40000,[10,19,0],[1,0,0],[0,0,1],10,3)]
room=np.vstack(R)
ground=plane(300000,[-15,-15,0],[1,0,0],[0,1,0],60,60)  # big outdoor
xyz=np.vstack([room,ground]).astype(np.float64)
G=np.array([398765.43,2712345.67,55.0])  # global UTM-like origin
xyz_g=xyz+G; N=len(xyz_g)
col=rng.uniform(40,210,(N,3))
data={'cartesianX':xyz_g[:,0].copy(),'cartesianY':xyz_g[:,1].copy(),'cartesianZ':xyz_g[:,2].copy(),
      'colorRed':col[:,0].copy(),'colorGreen':col[:,1].copy(),'colorBlue':col[:,2].copy()}
e=pye57.E57('synth_io.e57',mode='w'); e.write_scan_raw(data)
print(f"wrote synth_io.e57  {N:,} pts (room={len(room):,} ground={len(ground):,}) global={G.tolist()}")
