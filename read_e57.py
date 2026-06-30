#!/usr/bin/env python3
"""
read_e57.py — streaming E57 -> clean .ply for huge survey clouds.

A 16 GB / 271M-point E57 cannot be loaded whole (OOM on a laptop). This reads it
in BLOCKS, voxel-downsamples each block immediately, accumulates only the reduced
result, recenters from the global/registered origin to a local frame, and writes a
manageable .ply the rest of the pipeline consumes. Peak memory = one block, not
the file.

CLI:
    python read_e57.py probe in.e57
    python read_e57.py convert in.e57 out.ply [--voxel 0.03] [--target-points 3000000]
"""
import sys, argparse
import numpy as np
import open3d as o3d
import pye57
from pye57 import libe57


def probe(path):
    e = pye57.E57(path)
    info = []
    total = 0
    for i in range(e.scan_count):
        h = e.get_header(i)
        info.append({"scan": i, "points": int(h.point_count),
                     "fields": list(h.point_fields)})
        total += h.point_count
    return {"scan_count": e.scan_count, "total_points": int(total), "scans": info}


def _voxel_np(pts, cols, voxel):
    """Voxel-downsample a numpy point (+optional color) block via Open3D."""
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        p.colors = o3d.utility.Vector3dVector(cols)
    d = p.voxel_down_sample(voxel)
    return np.asarray(d.points), (np.asarray(d.colors) if cols is not None else None)


def convert(src, dst, voxel=0.03, target_points=3_000_000,
            block=5_000_000, with_color=True, scan_index=0, bounds=None):
    """
    bounds: optional (xmin, ymin, xmax, ymax) in the E57's GLOBAL frame. Points
    outside this XY box are skipped during streaming (all Z kept). Used by stage 2
    to reconvert only the indoor hall at high resolution.
    """
    e = pye57.E57(src)
    h = e.get_header(scan_index)
    n_total = h.point_count
    fields = list(h.point_fields)
    has_color = with_color and {"colorRed", "colorGreen", "colorBlue"}.issubset(fields)

    use_fields = ["cartesianX", "cartesianY", "cartesianZ"]
    if has_color:
        use_fields += ["colorRed", "colorGreen", "colorBlue"]

    # block-sized buffers (NOT full point_count) -> bounded memory
    buffers = libe57.VectorSourceDestBuffer()
    arrays = {}
    for f in use_fields:
        arr, buf = e.make_buffer(f, block)
        arrays[f] = arr
        buffers.append(buf)

    reader = h.points.reader(buffers)
    kept_pts, kept_cols = [], []
    read_total = 0
    color_scale = None

    while True:
        n = reader.read()
        if n == 0:
            break
        read_total += n
        xyz = np.stack([arrays["cartesianX"][:n],
                        arrays["cartesianY"][:n],
                        arrays["cartesianZ"][:n]], axis=1).astype(np.float64)
        cols = None
        if has_color:
            c = np.stack([arrays["colorRed"][:n],
                          arrays["colorGreen"][:n],
                          arrays["colorBlue"][:n]], axis=1).astype(np.float64)
            if color_scale is None:
                color_scale = 65535.0 if c.max() > 255 else 255.0
            cols = np.clip(c / color_scale, 0, 1)

        # stage-2 bounds crop: keep only points inside the hall footprint (global XY)
        if bounds is not None:
            bx0, by0, bx1, by1 = bounds
            m = ((xyz[:, 0] >= bx0) & (xyz[:, 0] <= bx1) &
                 (xyz[:, 1] >= by0) & (xyz[:, 1] <= by1))
            xyz = xyz[m]
            if cols is not None:
                cols = cols[m]
            if len(xyz) == 0:
                continue   # whole block outside the hall

        dp, dc = _voxel_np(xyz, cols, voxel)
        kept_pts.append(dp)
        if dc is not None:
            kept_cols.append(dc)
        # consolidate every 10 blocks so accumulation stays bounded on huge files
        if len(kept_pts) >= 10:
            cp = np.vstack(kept_pts)
            cc = np.vstack(kept_cols) if kept_cols else None
            cp, cc = _voxel_np(cp, cc, voxel)
            kept_pts = [cp]
            kept_cols = [cc] if cc is not None else []
        print(f"  read {read_total:,}/{n_total:,}  "
              f"kept-so-far ~{sum(len(k) for k in kept_pts):,}", end="\r")
    reader.close()
    print()

    pts = np.vstack(kept_pts)
    cols = np.vstack(kept_cols) if kept_cols else None

    # final voxel pass to dedup chunk-boundary overlaps
    pts, cols = _voxel_np(pts, cols, voxel)

    # if still above budget, coarsen on the (now small) cloud until under target
    v = voxel
    while len(pts) > target_points:
        v *= 1.3
        pts, cols = _voxel_np(pts, cols, v)
    final_voxel = round(v, 4)

    # recenter from global/registered origin to a local frame (metres from min corner)
    origin = pts.min(axis=0)
    pts_local = pts - origin

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_local)
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(dst, out)

    meta = {
        "src": src, "dst": dst,
        "total_points_in": int(n_total),
        "points_read": int(read_total),
        "points_out": int(len(pts_local)),
        "voxel_start_m": voxel,
        "voxel_final_m": final_voxel,
        "had_color": bool(cols is not None),
        "global_origin_subtracted": origin.tolist(),   # ADD THIS to local coords -> global
        "bounds_global": list(bounds) if bounds is not None else None,
        "local_extent_m": (pts_local.max(0) - pts_local.min(0)).round(2).tolist(),
    }
    import json
    with open(dst + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("probe"); p1.add_argument("src")
    p2 = sub.add_parser("convert")
    p2.add_argument("src"); p2.add_argument("dst")
    p2.add_argument("--voxel", type=float, default=0.03)
    p2.add_argument("--target-points", type=int, default=3_000_000)
    p2.add_argument("--block", type=int, default=5_000_000)
    p2.add_argument("--bounds", type=str, default=None,
                    help="global XY crop: xmin,ymin,xmax,ymax")
    a = ap.parse_args()

    if a.cmd == "probe":
        r = probe(a.src)
        print(f"scans: {r['scan_count']}   total points: {r['total_points']:,}")
        for s in r["scans"]:
            print(f"  scan {s['scan']}: {s['points']:,}  fields={s['fields']}")
    else:
        bounds = tuple(float(x) for x in a.bounds.split(",")) if a.bounds else None
        r = convert(a.src, a.dst, voxel=a.voxel,
                    target_points=a.target_points, block=a.block, bounds=bounds)
        print(f"  in ............. {r['total_points_in']:,} points")
        print(f"  out ............ {r['points_out']:,} points")
        print(f"  voxel .......... {r['voxel_start_m']} -> {r['voxel_final_m']} m")
        print(f"  origin removed . {r['global_origin_subtracted']}")
        if r['bounds_global']:
            print(f"  bounds (global)  {r['bounds_global']}")
        print(f"  local extent ... {r['local_extent_m']} m")
        print(f"  -> {r['dst']}  (+ .meta.json)")