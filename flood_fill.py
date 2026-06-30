#!/usr/bin/env python3
"""
flood_fill.py — isolate enclosed interior surfaces from a point cloud.

Bounds indoor from outdoor AND repairs gappy walls AND removes competing outdoor
geometry, in one pass, so wall detection becomes "fit planes to a clean watertight
boundary" instead of "hope RANSAC finds sparse verticals".

Algorithm (all O(N + V), V = voxel count; see complexity notes in code):
  1. voxelize points -> dense occupancy grid (padded so the exterior is free)
  2. seal gaps (binary_closing) so doorways/windows/data-holes don't leak
  3. label free space; components touching the grid border = EXTERIOR
  4. interior = enclosed free components (not border-touching), above min volume
     -> "keep all rooms" falls out for free: every sealed room is its own component
  5. boundary = occupied voxels adjacent to interior = the enclosing surfaces
  6. mask the ORIGINAL full-res points by boundary voxels -> clean wall/floor/ceiling

The fill grid is COARSE (fill_voxel, ~human-step) and only used as a MASK; the
output keeps real full-resolution points, so wall sharpness is preserved.

CLI: python flood_fill.py in.ply out.ply [--fill-voxel 0.2] [--seal-m 1.0]
"""
import sys, argparse, time
import numpy as np
import open3d as o3d
from scipy import ndimage


def flood_fill_interior(src, dst,
                        fill_voxel=0.2,
                        seal_vox=2,
                        min_room_m3=2.0,
                        mask_dilate=1) -> dict:
    """
    Indoor = free space COVERED by a ceiling above and standing on a floor below.
    This is doorway-immune (unlike morphological sealing): a doorway's air still
    has roof above and floor below, so it stays 'indoor'; outdoor ground has floor
    below but open sky above, so it's excluded. The up-axis MUST be Z (the inspector
    guarantees gravity_aligned before this runs).
    """
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(src)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    n_in = len(pts)
    if n_in == 0:
        raise ValueError(f"no points in {src}")

    mn = pts.min(0)
    pad = 2
    dims = (np.ceil((pts.max(0) - mn) / fill_voxel).astype(int) + 1 + 2 * pad)
    n_vox = int(np.prod(dims))

    # ---- 1. voxelize: O(N) ----
    vidx = (np.floor((pts - mn) / fill_voxel).astype(np.int64) + pad)
    occ = np.zeros(dims, dtype=bool)
    occ[vidx[:, 0], vidx[:, 1], vidx[:, 2]] = True

    # ---- 2. close only small DATA-GAP holes (not doorways): O(V*seal_vox) ----
    struct = ndimage.generate_binary_structure(3, 1)
    occ_sealed = ndimage.binary_closing(occ, iterations=seal_vox) if seal_vox else occ

    # ---- 3. covered-space test along Z (up-axis): O(V) ----
    # bel[...,z] = any occupied at or below z ; abv[...,z] = any occupied at or above z
    bel = np.maximum.accumulate(occ_sealed, axis=2)
    abv = np.flip(np.maximum.accumulate(np.flip(occ_sealed, axis=2), axis=2), axis=2)
    indoor_free = (~occ_sealed) & bel & abv          # roofed AND floored == indoor

    # ---- 4. keep significant connected interior region(s): O(V) ----
    lbl, n_comp = ndimage.label(indoor_free, structure=struct)
    counts = np.bincount(lbl.ravel())
    min_vox = max(1, int(min_room_m3 / (fill_voxel ** 3)))
    interior = np.zeros(dims, dtype=bool)
    kept = []
    for cid in range(1, n_comp + 1):
        if counts[cid] < min_vox:
            continue
        interior |= (lbl == cid)
        kept.append(int(counts[cid]))
    if not kept:
        raise ValueError("no covered interior found — capture may be roofless/"
                         "outdoor, or the up-axis isn't Z")

    # ---- 5. boundary = occupied voxels adjacent to interior: O(V) ----
    interior_dil = ndimage.binary_dilation(interior, structure=struct,
                                           iterations=mask_dilate)
    boundary = interior_dil & occ_sealed

    # ---- 6. mask ORIGINAL full-res points by boundary voxels: O(N) ----
    keep = boundary[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    out_pts = pts[keep]
    out_cols = cols[keep] if cols is not None else None

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(out_pts)
    if out_cols is not None:
        out.colors = o3d.utility.Vector3dVector(out_cols)
    o3d.io.write_point_cloud(dst, out)

    return {
        "src": src, "dst": dst,
        "n_in": int(n_in), "n_kept": int(len(out_pts)),
        "dropped_frac": round(1 - len(out_pts) / n_in, 3),
        "grid_dims": [int(d) for d in dims], "n_voxels": n_vox,
        "fill_voxel_m": fill_voxel,
        "n_interior_components": int(n_comp),
        "n_interior_regions_kept": len(kept),
        "interior_region_voxels": kept,
        "seconds": round(time.time() - t0, 2),
        "out_extent_m": (out_pts.max(0) - out_pts.min(0)).round(2).tolist()
        if len(out_pts) else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--fill-voxel", type=float, default=0.2)
    ap.add_argument("--seal-vox", type=int, default=2,
                    help="closing iterations for data-gap holes only (NOT doorways)")
    ap.add_argument("--min-room-m3", type=float, default=2.0)
    a = ap.parse_args()
    r = flood_fill_interior(a.src, a.dst, fill_voxel=a.fill_voxel,
                            seal_vox=a.seal_vox, min_room_m3=a.min_room_m3)
    print(f"  in ............... {r['n_in']:,} pts")
    print(f"  grid ............. {r['grid_dims']}  ({r['n_voxels']:,} voxels)")
    print(f"  interior regions . {r['n_interior_regions_kept']}  "
          f"(voxels: {r['interior_region_voxels']})")
    print(f"  kept ............. {r['n_kept']:,} pts  "
          f"(dropped {r['dropped_frac']*100:.1f}%)")
    print(f"  out extent ....... {r['out_extent_m']} m")
    print(f"  time ............. {r['seconds']} s")
    print(f"  -> {r['dst']}")
