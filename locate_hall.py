#!/usr/bin/env python3
"""
locate_hall.py — STAGE 2 of the venue pipeline (no human, no CloudCompare).

Takes the coarse cloud from stage 1, runs flood-fill to LOCATE the indoor hall,
translates that interior bounding box from the coarse cloud's local frame back to
the E57's global frame, and reconverts ONLY that footprint from the full-scale
E57 at high resolution — so walls become resolvable.

  coarse.ply (+ .meta.json)  ──►  flood-fill  ──►  interior bbox (local)
        │                                              │
        │  origin (from meta)                          │  + margin, + origin
        └──────────────────────────────────────────────┴──►  global bbox
                                                              │
   full E57  ──► read_e57 convert --bounds <global> --voxel 0.03 ──► hall.ply

Usage:
    python locate_hall.py <full.e57> <coarse.ply> <hall.ply> [--voxel 0.03]
                          [--margin 3.0] [--target-points 6000000]
"""
import sys, os, json, argparse
import numpy as np
from flood_fill import flood_fill_interior
import read_e57


def locate_and_reconvert(e57, coarse_ply, hall_ply,
                         voxel=0.03, margin=3.0, target_points=6_000_000,
                         fill_voxel=0.2):
    # --- read stage-1 metadata for the global origin ---
    meta_path = coarse_ply + ".meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} missing — run stage 1 (read_e57 convert) first")
    with open(meta_path) as fh:
        origin = np.array(json.load(fh)["global_origin_subtracted"], dtype=np.float64)

    # --- flood-fill the coarse cloud to LOCATE the interior (bbox only) ---
    tmp = hall_ply + ".locate.ply"
    ff = flood_fill_interior(coarse_ply, tmp, fill_voxel=fill_voxel)
    os.remove(tmp)
    if not ff.get("interior_bbox_local"):
        raise ValueError("flood-fill found no indoor interior in the coarse cloud")
    lo, hi = (np.array(b, dtype=np.float64) for b in ff["interior_bbox_local"])

    # --- local -> global (add origin), then expand by margin to capture walls ---
    g_lo = lo + origin
    g_hi = hi + origin
    bounds = (float(g_lo[0] - margin), float(g_lo[1] - margin),
              float(g_hi[0] + margin), float(g_hi[1] + margin))

    print(f"  located interior (local) : {lo[:2].round(1).tolist()} .. {hi[:2].round(1).tolist()}")
    print(f"  global bounds (+{margin}m)  : {[round(b,1) for b in bounds]}")
    print(f"  reconverting hall at {voxel} m ...")

    # --- stage-2 reconvert: hall footprint only, high resolution ---
    r = read_e57.convert(e57, hall_ply, voxel=voxel,
                         target_points=target_points, bounds=bounds)
    return {
        "interior_bbox_local": ff["interior_bbox_local"],
        "global_bounds": bounds,
        "hall_points_out": r["points_out"],
        "hall_voxel_final_m": r["voxel_final_m"],
        "hall_local_extent_m": r["local_extent_m"],
        "hall_ply": hall_ply,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("e57"); ap.add_argument("coarse_ply"); ap.add_argument("hall_ply")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--target-points", type=int, default=6_000_000)
    a = ap.parse_args()
    r = locate_and_reconvert(a.e57, a.coarse_ply, a.hall_ply,
                             voxel=a.voxel, margin=a.margin,
                             target_points=a.target_points)
    print(f"  hall points ...... {r['hall_points_out']:,}")
    print(f"  hall voxel ....... {r['hall_voxel_final_m']} m")
    print(f"  hall extent ...... {r['hall_local_extent_m']} m")
    print(f"  -> {r['hall_ply']}")
