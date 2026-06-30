"""Synthetic indoor+outdoor: an enclosed room (with a doorway gap) sitting on a
large outdoor ground plane with outdoor clutter. Tests that flood-fill keeps the
room's surfaces and discards everything outdoors."""
import numpy as np, open3d as o3d
rng=np.random.default_rng(7)

def plane(n,p0,u,v,su,sv,noise=0.01):
    a=rng.uniform(0,su,n);b=rng.uniform(0,sv,n)
    return p0+np.outer(a,u)+np.outer(b,v)+rng.normal(0,noise,(n,3))

# ROOM: 10 x 8 x 3 m box centred at (15,15)
R=[]
R.append(plane(40000,[10,11,0],[1,0,0],[0,1,0],10,8))   # floor
R.append(plane(30000,[10,11,3],[1,0,0],[0,1,0],10,8))   # ceiling
R.append(plane(20000,[10,11,0],[0,1,0],[0,0,1],8,3))    # left wall (x=10)
R.append(plane(20000,[20,11,0],[0,1,0],[0,0,1],8,3))    # right wall (x=20)
R.append(plane(20000,[10,11,0],[1,0,0],[0,0,1],10,3))   # rear wall (y=11)
# front wall (y=19) WITH a 1.2 m doorway gap in the middle
fw=plane(20000,[10,19,0],[1,0,0],[0,0,1],10,3)
fw=fw[~((fw[:,0]>14.4)&(fw[:,0]<15.6))]                 # cut doorway
R.append(fw)
room=np.vstack(R)

# OUTDOOR: big ground plane 40x40 around it, + a few outdoor blobs (cars/trees)
ground=plane(200000,[-5,-5,0],[1,0,0],[0,1,0],45,45)
blob1=rng.normal([35,35,1],0.5,(8000,3))
blob2=rng.normal([0,30,1.5],0.6,(8000,3))
outdoor=np.vstack([ground,blob1,blob2])

xyz=np.vstack([room,outdoor]).astype(np.float32)
p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud('synth_io.ply',p)
print(f"wrote synth_io.ply  total={len(xyz):,}  room={len(room):,} outdoor={len(outdoor):,}")
print(f"extent: {(xyz.max(0)-xyz.min(0)).round(1).tolist()}")
