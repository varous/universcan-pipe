#!/usr/bin/env python3
"""
abstract.py  —  the conversion core (SKELETON, runnable end-to-end).
Reads a geometry .ply, segments it into planes, classifies each into a canonical
tag from tags.yaml, bounds it to a polygon, and emits:
    - manifest.json  (faces: tag, 3D vertices, normal, area)  -> feed build_skp.rb
    - venue.dxf      (3DFACEs on layers = canonical tag)       -> import to SketchUp

This is the *happy-path* version. The classify/bound heuristics are deliberately
simple; hardening them across messy real venues is the ongoing work (validate
against measured SPL, then tune tags.yaml). Run inspect_ply.py FIRST.

Deps: open3d, numpy, ezdxf, pyyaml, shapely (shapely optional; bbox fallback used here)

Usage:
    python abstract.py scan.ply --tags tags.yaml --out out/ [--voxel 0.05]
"""
import argparse, json, os
import numpy as np
import open3d as o3d
import ezdxf
import yaml


def load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def in_plane_basis(normal):
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def oriented_quad(pts, normal):
    """Project inliers to the plane, take the axis-aligned rectangle in the
    in-plane basis, lift the 4 corners back to 3D. (Upgrade path: concave hull
    via shapely/alphashape for irregular surfaces.)"""
    c = pts.mean(0)
    u, v = in_plane_basis(normal)
    pu = (pts - c) @ u
    pv = (pts - c) @ v
    umin, umax, vmin, vmax = pu.min(), pu.max(), pv.min(), pv.max()
    corners2d = [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]
    return [ (c + cu * u + cv * v).tolist() for cu, cv in corners2d ], c


def classify(normal, cpts, bounds, cfg):
    """Return a canonical tag from normal direction + position + cluster extent.
    cpts = the cluster's 3D points; bounds = (min,max) of the whole cloud."""
    cl = cfg["classify"]
    nz = abs(normal[2])
    mn, mx = bounds
    centroid = cpts.mean(0)
    height = mx[2] - mn[2]
    z = centroid[2]
    zfrac = (z - mn[2]) / height if height else 0
    z_above_floor = cpts[:, 2].min() - mn[2]
    z_span = cpts[:, 2].max() - cpts[:, 2].min()

    # ---- horizontal surfaces (floor / ceiling / stage) -----------
    if nz >= cl["vertical_cos"]:
        st = cl["stage"]
        depth = mx[1] - mn[1]
        yfrac = (centroid[1] - mn[1]) / depth if depth else 0
        if (st["enabled"] and st["min_z_m"] <= (z - mn[2]) <= st["max_z_m"]
                and yfrac <= st["front_zone_frac"]):
            return "STAGE"
        return "FLOOR" if zfrac < cl["floor_ceiling_split_frac"] else "CEILING"

    # ---- vertical surfaces (walls / balcony faces) ---------------
    if nz <= cl["horizontal_cos"]:
        bf = cl["balcony_face"]
        if z_above_floor >= bf["min_z_m"] and z_span <= bf["max_height_m"]:
            return "BALCONY_FACE"           # short + elevated => balcony face
        cx = (mn[0] + mx[0]) / 2
        cy = (mn[1] + mx[1]) / 2
        if abs(normal[1]) > abs(normal[0]):   # normal along Y => front/rear wall
            return "WALL_REAR" if centroid[1] > cy else "WALL_FRONT"
        return "WALL_LEFT" if centroid[0] < cx else "WALL_RIGHT"

    # ---- sloped surfaces => raked audience -----------------------
    return "AUDIENCE_MAIN"


def run(ply_path, tags_path, out_dir, voxel_override=None):
    cfg = load_cfg(tags_path)
    P = cfg["pipeline"]
    voxel = voxel_override or P["voxel_m"]
    os.makedirs(out_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise SystemExit(f"No points read from {ply_path}")
    raw_n = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    pts_all = np.asarray(pcd.points)
    bounds = (pts_all.min(0), pts_all.max(0))

    # DBSCAN eps must scale with the cloud's ACTUAL spacing, not a fixed value:
    # splat-derived / cropped clouds are far sparser than dense LiDAR, and a
    # fixed eps labels everything as noise. Measure spacing, set eps >= 5x it.
    from scipy.spatial import cKDTree
    _s = pts_all if len(pts_all) <= 20000 else pts_all[
        np.random.default_rng(0).choice(len(pts_all), 20000, replace=False)]
    _spacing = float(np.median(cKDTree(_s).query(_s, k=2)[0][:, 1]))
    eps = max(P["dbscan_eps_m"], 5.0 * _spacing)
    print(f"  loaded {raw_n:,} pts -> {len(pts_all):,} after voxel {voxel} m "
          f"(spacing {_spacing:.3f} m, dbscan eps {eps:.2f} m)")

    faces = []
    rest = pcd
    for i in range(P["max_planes"]):
        if len(rest.points) < P["min_plane_points"]:
            break
        model, inl = rest.segment_plane(
            P["ransac_dist_m"], P["ransac_n"], P["ransac_iters"])
        if len(inl) < P["min_plane_points"]:
            break
        plane = rest.select_by_index(inl)
        rest = rest.select_by_index(inl, invert=True)
        normal = np.array(model[:3]); normal /= np.linalg.norm(normal)

        # split coplanar-but-separate patches
        labels = np.array(plane.cluster_dbscan(
            eps=eps, min_points=P["dbscan_min_points"]))
        ppts = np.asarray(plane.points)
        for lab in sorted(set(labels)):
            if lab < 0:
                continue
            cpts = ppts[labels == lab]
            if len(cpts) < P["min_plane_points"]:
                continue
            corners, centroid = oriented_quad(cpts, normal)
            tag = classify(normal, cpts, bounds, cfg)
            faces.append({
                "tag": tag,
                "normal": normal.round(4).tolist(),
                "centroid": np.round(centroid, 3).tolist(),
                "n_points": int(len(cpts)),
                "vertices": [list(np.round(v, 3)) for v in corners],
            })

    # ---- emit manifest.json --------------------------------------
    manifest = {"source": ply_path, "voxel_m": voxel,
                "n_faces": len(faces), "faces": faces}
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ---- emit venue.dxf (3DFACE per quad, layer = tag) -----------
    doc = ezdxf.new(); msp = doc.modelspace()
    for f in faces:
        if f["tag"] not in doc.layers:
            doc.layers.add(f["tag"])
        v = f["vertices"]
        msp.add_3dface([v[0], v[1], v[2], v[3]], dxfattribs={"layer": f["tag"]})
    dpath = os.path.join(out_dir, "venue.dxf")
    doc.saveas(dpath)

    # ---- summary -------------------------------------------------
    from collections import Counter
    counts = Counter(f["tag"] for f in faces)
    print(f"  emitted {len(faces)} faces:")
    for t, c in counts.most_common():
        print(f"     {t:<18} {c}")
    print(f"  -> {mpath}")
    print(f"  -> {dpath}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--tags", default="tags.yaml")
    ap.add_argument("--out", default="out")
    ap.add_argument("--voxel", type=float, default=None)
    a = ap.parse_args()
    run(a.ply, a.tags, a.out, a.voxel)