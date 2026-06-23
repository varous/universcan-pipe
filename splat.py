#!/usr/bin/env python3
"""
splat.py — Tier-1 Gaussian-splat front end.

A 3DGS .ply is NOT geometry: each "point" is a Gaussian (position + scale +
rotation + opacity + SH colour). This module turns one into a plain geometry
point cloud the rest of the pipeline can consume, by:
  1. reading the Gaussian centres,
  2. dropping low-opacity floaters and oversized background blobs,
  3. (optionally) recovering per-point colour from the SH DC term,
  4. writing a clean xyz(+rgb) .ply.

This is the *loose* path — splat centres sit in a fuzzy shell around real
surfaces and density is inverted (few Gaussians on big flat walls). Use the
paired LiDAR cloud when you have it; use this when a splat is all you have.

CLI:
    python splat.py in_splat.ply out_centres.ply
"""
import sys
import numpy as np
from plyfile import PlyData, PlyElement

SPLAT_MARKERS = {"scale_0", "rot_0", "opacity", "f_dc_0"}
SH_C0 = 0.28209479177387814        # 0th-order spherical-harmonic constant


def is_splat_ply(path: str) -> bool:
    """Header sniff: True if the .ply carries Gaussian-splat attributes."""
    ply = PlyData.read(path)
    if "vertex" not in ply:
        return False
    return SPLAT_MARKERS.issubset(set(ply["vertex"].data.dtype.names))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_centres(src: str, dst: str,
                    opacity_thresh: float = 0.5,
                    scale_pct: float = 99.0,
                    scale_max=None,
                    color: bool = True) -> dict:
    """
    Filter a 3DGS .ply to its meaningful centres and write a geometry .ply.

    opacity_thresh : keep Gaussians with sigmoid(opacity) above this (drops floaters)
    scale_pct      : drop Gaussians whose size exceeds this percentile (drops blobs).
                     Percentile adapts to scene scale, so it works for a small room
                     and a 175 m site without retuning.
    scale_max      : absolute size cap (metres). If set, overrides scale_pct.
    """
    ply = PlyData.read(src)
    v = ply["vertex"].data
    names = set(v.dtype.names)
    if not SPLAT_MARKERS.issubset(names):
        raise ValueError("not a 3DGS splat .ply (missing scale_/rot_/opacity/f_dc)")

    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float64))
    # 3DGS stores log-scale; real size is exp(scale). Use the largest axis.
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)
                   .astype(np.float64)).max(1)
    n = len(xyz)

    keep = opacity > opacity_thresh
    cutoff = float(scale_max) if scale_max is not None else float(np.percentile(scale, scale_pct))
    keep &= scale < cutoff
    kept = int(keep.sum())
    if kept == 0:
        raise ValueError("filter removed all points — relax opacity_thresh / scale cutoff")

    pts = xyz[keep]

    if color and {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)[keep]
        rgb = np.clip(0.5 + SH_C0 * fdc, 0, 1)
        cols = (rgb * 255).astype(np.uint8)
        out = np.zeros(kept, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        out['red'], out['green'], out['blue'] = cols[:, 0], cols[:, 1], cols[:, 2]
    else:
        out = np.zeros(kept, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    out['x'], out['y'], out['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(dst)

    return {
        "src": src, "dst": dst,
        "n_gaussians": n, "n_kept": kept,
        "dropped_frac": round(1 - kept / n, 3),
        "opacity_thresh": opacity_thresh,
        "scale_cutoff": round(cutoff, 4),
        "scale_rule": f"absolute {scale_max}" if scale_max is not None else f"p{scale_pct}",
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python splat.py in_splat.ply out_centres.ply")
        sys.exit(1)
    r = extract_centres(sys.argv[1], sys.argv[2])
    print(f"  gaussians ......... {r['n_gaussians']:,}")
    print(f"  kept .............. {r['n_kept']:,}  (dropped {r['dropped_frac']*100:.1f}%)")
    print(f"  scale cutoff ...... {r['scale_cutoff']}  ({r['scale_rule']})")
    print(f"  -> {r['dst']}")
