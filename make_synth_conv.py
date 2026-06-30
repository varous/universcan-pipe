"""Convention-centre in miniature: large hall, MULTI-LEVEL ceiling (3 heights),
floor at a z-OFFSET (z=12 like the real scan), perimeter walls, and heavy
furniture clutter (tables + chairs). Tests: 1 floor, merged multi-level ceiling,
4 walls, furniture dropped."""
import numpy as np, open3d as o3d
rng=np.random.default_rng(11)
def plane(n,p0,u,v,su,sv,noise=0.01):
    a=rng.uniform(0,su,n);b=rng.uniform(0,sv,n)
    return p0+np.outer(a,u)+np.outer(b,v)+rng.normal(0,noise,(n,3))
Z=12.0  # floor z-offset, like the real convention centre
W,D=60,40
S=[]
S.append(plane(300000,[0,0,Z],[1,0,0],[0,1,0],W,D))            # FLOOR at z=12
# multi-level ceiling: 3 coffered heights (Z+8, Z+9, Z+11) in patches
for (x0,x1,h) in [(0,20,8),(20,40,9),(40,60,11)]:
    S.append(plane(120000,[x0,0,Z+h],[1,0,0],[0,1,0],x1-x0,D))
# perimeter walls (Z..Z+11)
S.append(plane(80000,[0,0,Z],[0,1,0],[0,0,1],D,11))            # LEFT  x=0
S.append(plane(80000,[W,0,Z],[0,1,0],[0,0,1],D,11))           # RIGHT x=60
S.append(plane(80000,[0,0,Z],[1,0,0],[0,0,1],W,11))           # FRONT y=0
S.append(plane(80000,[0,D,Z],[1,0,0],[0,0,1],W,11))           # REAR  y=40
arch=np.vstack(S)
# FURNITURE: 40 round tables (tops at Z+0.75) + chairs (short verticals)
F=[]
for tx,ty in rng.uniform([5,5],[55,35],(40,2)):
    th=rng.uniform(Z,Z+0.75,1500);ta=rng.uniform(0,2*np.pi,1500)
    F.append(np.c_[tx+0.5*np.cos(ta),ty+0.5*np.sin(ta),th]+rng.normal(0,0.01,(1500,3)))
    F.append(np.c_[tx+rng.uniform(-0.5,0.5,600),ty+rng.uniform(-0.5,0.5,600),np.full(600,Z+0.75)])  # top
    for cx,cy in rng.uniform([tx-1,ty-1],[tx+1,ty+1],(4,2)):  # chairs
        F.append(np.c_[cx+rng.uniform(-0.2,0.2,300),cy+rng.uniform(-0.2,0.2,300),rng.uniform(Z,Z+0.9,300)])
pts=np.vstack(S+F).astype(np.float32)
p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(pts)
o3d.io.write_point_cloud('synth_conv.ply',p)
print(f"wrote synth_conv.ply {len(pts):,} pts | arch={len(arch):,} furniture={len(np.vstack(F)):,}")
print(f"expect: 1 FLOOR(z={Z}), CEILING(3 levels merged), 4 WALLS, furniture DROPPED")
