"""Generate a synthetic venue point cloud (meters, Z-up) to test the pipeline."""
import numpy as np
from plyfile import PlyData, PlyElement

rng = np.random.default_rng(7)
def plane(n, p0, u, v, su, sv, noise=0.01):
    a = rng.uniform(0, su, n); b = rng.uniform(0, sv, n)
    pts = p0 + np.outer(a, u) + np.outer(b, v)
    pts += rng.normal(0, noise, pts.shape)
    return pts

parts = []
# Floor 20x15 at z=0
parts.append(plane(40000, [0,0,0], [1,0,0], [0,1,0], 20, 15))
# Ceiling at z=8
parts.append(plane(30000, [0,0,8], [1,0,0], [0,1,0], 20, 15))
# Left wall x=0, Right wall x=20 (y-z planes)
parts.append(plane(20000, [0,0,0], [0,1,0], [0,0,1], 15, 8))
parts.append(plane(20000, [20,0,0], [0,1,0], [0,0,1], 15, 8))
# Rear wall y=15 (x-z plane)
parts.append(plane(20000, [0,15,0], [1,0,0], [0,0,1], 20, 8))
# Raked audience: slope rising toward rear (y from 4..14), x full width
n=25000
x=rng.uniform(2,18,n); y=rng.uniform(4,14,n); z=0.1+(y-4)*0.18
aud=np.stack([x,y,z],1)+rng.normal(0,0.01,(n,3))
parts.append(aud)
# Stage: elevated platform z=1.0, front (y 0..4)
parts.append(plane(8000, [4,0,1.0], [1,0,0], [0,1,0], 12, 3.5))

pts = np.vstack(parts).astype(np.float32)
# add fake RGB
rgb = (rng.uniform(40,200,pts.shape)).astype(np.uint8)
verts = np.zeros(len(pts), dtype=[('x','f4'),('y','f4'),('z','f4'),
                                  ('red','u1'),('green','u1'),('blue','u1')])
verts['x'],verts['y'],verts['z']=pts[:,0],pts[:,1],pts[:,2]
verts['red'],verts['green'],verts['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
PlyData([PlyElement.describe(verts,'vertex')],text=False).write('synth_venue.ply')
print("wrote synth_venue.ply", len(pts), "points")
