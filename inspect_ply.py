#!/usr/bin/env python3
"""
inspect_ply.py  —  Step 1: confirm what a Green Valley (or any) .ply actually contains
BEFORE you trust it downstream. Reports format, splat-vs-geometry, fields, units,
up-axis, density, and a recommended voxel size. Prints a human report and (optionally)
writes a JSON summary.

Dependencies: numpy, plyfile   (intentionally NO open3d, so this runs anywhere, today)

Usage:
    python inspect_ply.py path/to/scan.ply
    python inspect_ply.py path/to/scan.ply --json out.json
"""
import argparse, json, sys
import numpy as np
from plyfile import PlyData

# Property names that mark a file as a 3D Gaussian splat, NOT a geometry cloud.
SPLAT_MARKERS = {"scale_0", "rot_0", "opacity", "f_dc_0"}


def load(path):
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise SystemExit(f"No 'vertex' element in {path}; elements present: "
                         f"{[e.name for e in ply.elements]}")
    v = ply["vertex"]
    names = list(v.data.dtype.names)
    xyz = np.stack([np.asarray(v[a], dtype=np.float64) for a in ("x", "y", "z")], axis=1)
    return ply, v, names, xyz


def dominant_plane_normal(xyz, iters=300, thresh=0.02, sample=20000):
    """Tiny RANSAC to find the biggest plane (usually the floor) and return its
    unit normal + inlier fraction. Pure numpy — no open3d needed."""
    pts = xyz if len(xyz) <= sample else xyz[np.random.default_rng(0).choice(len(xyz), sample, replace=False)]
    best_n, best_inl = None, 0
    rng = np.random.default_rng(1)
    for _ in range(iters):
        i = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        d = np.abs((pts - p0) @ n)
        inl = int((d < thresh).sum())
        if inl > best_inl:
            best_inl, best_n = inl, n
    if best_n is None:
        return np.array([0, 0, 1.0]), 0.0
    if best_n[2] < 0:            # orient roughly up
        best_n = -best_n
    return best_n, best_inl / len(pts)


def nn_spacing(xyz, tree_max=200000, query_n=5000):
    """Median nearest-neighbour distance — a proxy for point spacing/density.
    Builds the tree on the FULL-density cloud (capped at tree_max) and queries a
    sample against it, so spacing reflects true local density, not the sample's."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(2)
    tree_pts = xyz if len(xyz) <= tree_max else \
        xyz[rng.choice(len(xyz), tree_max, replace=False)]
    tree = cKDTree(tree_pts)
    q = tree_pts if len(tree_pts) <= query_n else \
        tree_pts[rng.choice(len(tree_pts), query_n, replace=False)]
    d, _ = tree.query(q, k=2)          # k=2: self + nearest neighbour
    return float(np.median(d[:, 1]))


def report(path, as_json=None):
    ply, v, names, xyz = load(path)
    fmt = "ascii" if ply.text else "binary"

    is_splat = SPLAT_MARKERS.issubset(set(names))
    has_normals = {"nx", "ny", "nz"}.issubset(names)
    has_color = ({"red", "green", "blue"}.issubset(names)
                 or {"r", "g", "b"}.issubset(names))
    scalar_fields = [n for n in names
                     if n not in {"x", "y", "z", "nx", "ny", "nz",
                                  "red", "green", "blue", "r", "g", "b"}
                     and not n.startswith(("f_dc", "f_rest", "scale", "rot"))
                     and n != "opacity"]

    mins, maxs = xyz.min(0), xyz.max(0)
    ext = maxs - mins
    diag = float(np.linalg.norm(ext))

    # Units heuristic: a single building/venue is ~tens of meters across.
    if diag > 2000:
        units = "millimetres (likely) — values look ~1000x too big for metres"
    elif diag < 2:
        units = "unknown / sub-metre — diagonal < 2, check export scale"
    else:
        units = "metres (likely)"

    n_hat, inl_frac = dominant_plane_normal(xyz)
    up_axis = int(np.argmax(np.abs(n_hat)))
    up_letter = "XYZ"[up_axis]
    z_up = (up_axis == 2 and abs(n_hat[2]) > 0.9)

    try:
        spacing = nn_spacing(xyz)
    except Exception:
        spacing = None
    # voxel ~ 3-5x point spacing is a safe downsample for plane fitting
    voxel = round(max(0.02, (spacing or 0.01) * 4), 3)

    R = {
        "file": path,
        "format": fmt,
        "n_points": int(len(xyz)),
        "is_gaussian_splat": is_splat,
        "has_normals": has_normals,
        "has_color": has_color,
        "scalar_fields": scalar_fields,
        "all_properties": names,
        "bbox_min": mins.round(3).tolist(),
        "bbox_max": maxs.round(3).tolist(),
        "extent_xyz": ext.round(3).tolist(),
        "diagonal": round(diag, 3),
        "inferred_units": units,
        "dominant_plane_normal": n_hat.round(3).tolist(),
        "dominant_plane_inlier_frac": round(inl_frac, 3),
        "inferred_up_axis": up_letter,
        "gravity_aligned_Zup": bool(z_up),
        "point_spacing_m": round(spacing, 4) if spacing else None,
        "recommended_voxel_m": voxel,
    }

    # ---------- human-readable report ----------
    print("=" * 64)
    print(f"  PLY INSPECTION  —  {path}")
    print("=" * 64)
    print(f"  format ................ {fmt}")
    print(f"  points ................ {len(xyz):,}")
    print(f"  properties ............ {', '.join(names)}")
    print("-" * 64)
    if is_splat:
        print("  !! GAUSSIAN SPLAT DETECTED  (scale_/rot_/opacity/f_dc present)")
        print("     This is NOT a geometry cloud. Do NOT feed straight to the")
        print("     plane pipeline — extract/filter centres first (Tier 1), or")
        print("     better, use the paired LiDAR .ply instead.")
        print("-" * 64)
    print(f"  has normals ........... {has_normals}"
          + ("" if has_normals else "   (will estimate in abstract.py)"))
    print(f"  has color ............. {has_color}")
    print(f"  other scalar fields ... {scalar_fields or 'none'}"
          + ("   <- possible classification field?" if scalar_fields else ""))
    print("-" * 64)
    print(f"  bbox min .............. {mins.round(2).tolist()}")
    print(f"  bbox max .............. {maxs.round(2).tolist()}")
    print(f"  extent (X,Y,Z) ........ {ext.round(2).tolist()}")
    print(f"  diagonal .............. {diag:.2f}")
    print(f"  >> inferred units ..... {units}")
    print("-" * 64)
    print(f"  dominant plane normal . {n_hat.round(3).tolist()}  "
          f"(inliers {inl_frac*100:.0f}%)")
    print(f"  >> inferred up-axis ... {up_letter}"
          + ("   (Z-up, gravity-aligned — good)" if z_up
             else "   (NOT clean Z-up — check / re-level!)"))
    print("-" * 64)
    if spacing:
        print(f"  point spacing (median). {spacing:.4f} m")
    print(f"  >> recommended voxel .. {voxel} m  (use in abstract.py)")
    print("=" * 64)

    # ---------- verdict ----------
    flags = []
    if is_splat:
        flags.append("SPLAT not geometry — wrong input for the plane pipeline")
    if "metres" not in units:
        flags.append(f"units suspect ({units}) — fix scale before proceeding")
    if not z_up:
        flags.append("not clean Z-up — re-level to the floor plane first")
    if flags:
        print("  VERDICT:  needs attention before processing")
        for f in flags:
            print(f"    - {f}")
    else:
        print("  VERDICT:  clean geometry cloud, metres, Z-up — ready for abstract.py")
        print(f"            start with --voxel {voxel}")
    print("=" * 64)

    if as_json:
        with open(as_json, "w") as fh:
            json.dump(R, fh, indent=2)
        print(f"  wrote summary -> {as_json}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect a .ply before processing.")
    ap.add_argument("path")
    ap.add_argument("--json", help="also write a JSON summary to this path")
    a = ap.parse_args()
    report(a.path, a.json)
