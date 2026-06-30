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


def _plane_area(pts, normal):
    """Area of the cluster's bounding rectangle in its own plane (m^2)."""
    u, v = in_plane_basis(normal)
    c = pts.mean(0)
    pu = (pts - c) @ u
    pv = (pts - c) @ v
    return float((pu.max() - pu.min()) * (pv.max() - pv.min()))


def staged_classify(raw, bounds, cfg):
    """
    Stage the WHOLE set of RANSAC planes, not each in isolation. Order:
      1. orientation-split: horizontal / vertical / sloped
      2. height-cluster horizontals into z-bands; merge coplanar fragments
         -> lowest large band = FLOOR, highest large band(s) = CEILING,
            middle large = STRUCTURE, small mid-height = FURNITURE (dropped)
      3. bbox-relative walls: vertical + near a bbox edge + tall = WALL_*;
         interior or short verticals = furniture/structure (dropped/STRUCTURE)
      4. merge fragments per surface so each real surface is ONE face
    Floor/ceiling are identified by BAND ORDER (robust to z-offset: the real
    convention-centre floor sits at z~12, not z~0), not absolute height.
    """
    cl = cfg["classify"]
    mn, mx = np.array(bounds[0]), np.array(bounds[1])
    H = float(mx[2] - mn[2]) or 1.0
    W = float(mx[0] - mn[0]) or 1.0
    D = float(mx[1] - mn[1]) or 1.0
    vcos = cl["vertical_cos"]; hcos = cl["horizontal_cos"]
    band_tol = cl.get("band_tol_m", 1.0)
    perim_frac = cl.get("wall_perim_frac", 0.12)
    min_arch_area = cl.get("min_arch_area_m2", 8.0)     # below this = furniture
    ceil_span = cl.get("ceiling_span_m", 3.0)           # top bands within this = ceiling

    horiz, vert, slope = [], [], []
    for r in raw:
        n = np.array(r["normal"]); nz = abs(n[2])
        r["zc"] = float(r["pts"][:, 2].mean())
        r["zspan"] = float(r["pts"][:, 2].max() - r["pts"][:, 2].min())
        r["area"] = _plane_area(r["pts"], n)
        (horiz if nz >= vcos else vert if nz <= hcos else slope).append(r)

    faces = []

    def emit(tag, pts, normal):
        corners, centroid = oriented_quad(pts, normal)
        faces.append({"tag": tag, "normal": list(np.round(normal, 4)),
                      "centroid": list(np.round(centroid, 3)),
                      "n_points": int(len(pts)),
                      "vertices": [list(np.round(v, 3)) for v in corners]})

    # ---- 2. HORIZONTALS: cluster into z-bands, merge, tag by band order ----
    horiz.sort(key=lambda r: r["zc"])
    bands = []
    for r in horiz:
        if bands and r["zc"] - bands[-1]["zc_last"] <= band_tol:
            bands[-1]["items"].append(r); bands[-1]["zc_last"] = r["zc"]
        else:
            bands.append({"items": [r], "zc_last": r["zc"]})
    band_recs = []
    for b in bands:
        pts = np.vstack([r["pts"] for r in b["items"]])
        band_recs.append({"pts": pts, "zc": float(pts[:, 2].mean()),
                          "area": _plane_area(pts, np.array([0, 0, 1.0]))})
    # large bands are architectural; small mid-height bands are furniture
    arch = [b for b in band_recs if b["area"] >= min_arch_area]
    arch.sort(key=lambda b: b["zc"])
    floor_zc = ceil_zc = None
    if arch:
        floor_zc = arch[0]["zc"]
        ceil_zc = arch[-1]["zc"]
        for i, b in enumerate(arch):
            up = np.array([0, 0, 1.0])
            if b["zc"] <= floor_zc + band_tol:
                emit("FLOOR", b["pts"], up)
            elif b["zc"] >= ceil_zc - ceil_span:
                emit("CEILING", b["pts"], up)           # top band(s) -> ceiling
            else:
                emit("STRUCTURE", b["pts"], up)          # mezzanine / platform
    # (small horizontal bands = furniture: intentionally dropped)

    # ---- 3. VERTICALS: bbox-relative walls; interior/short -> dropped ----
    # CRITICAL: gate wall height against the REAL floor-to-ceiling height (from the
    # detected bands), NOT bbox height. The scan's bbox includes sub-floor structure
    # (floor sits at z~12, bbox min at 0), so bbox H is ~2x the real room height —
    # using it rejects every real wall as "too short".
    if floor_zc is not None and ceil_zc is not None and ceil_zc > floor_zc:
        room_height = ceil_zc - floor_zc
    else:
        room_height = H                                  # fallback: no bands found
    min_wall_h = cl.get("min_wall_height_frac", 0.30) * room_height
    wall_groups = {"WALL_LEFT": [], "WALL_RIGHT": [],
                   "WALL_FRONT": [], "WALL_REAR": [], "STRUCTURE": []}
    for r in vert:
        if r["zspan"] < min_wall_h:
            continue                                     # short vertical = furniture, DROP
        n = np.array(r["normal"]); c = r["pts"].mean(0)
        if abs(n[0]) >= abs(n[1]):                       # normal along X -> L/R wall
            near_lo = (c[0] - mn[0]) <= perim_frac * W
            near_hi = (mx[0] - c[0]) <= perim_frac * W
            if near_lo:   wall_groups["WALL_LEFT"].append(r)
            elif near_hi: wall_groups["WALL_RIGHT"].append(r)
            elif r["area"] >= min_arch_area: wall_groups["STRUCTURE"].append(r)
        else:                                            # normal along Y -> F/R wall
            near_lo = (c[1] - mn[1]) <= perim_frac * D
            near_hi = (mx[1] - c[1]) <= perim_frac * D
            if near_lo:   wall_groups["WALL_FRONT"].append(r)
            elif near_hi: wall_groups["WALL_REAR"].append(r)
            elif r["area"] >= min_arch_area: wall_groups["STRUCTURE"].append(r)
    # ---- 4. merge fragments per side -> one face per wall ----
    for tag, items in wall_groups.items():
        if not items:
            continue
        pts = np.vstack([r["pts"] for r in items])
        nrm = np.mean([np.array(r["normal"]) for r in items], axis=0)
        nrm /= (np.linalg.norm(nrm) or 1.0)
        emit(tag, pts, nrm)

    # ---- sloped surfaces -> raked audience (only if sizeable) ----
    if slope:
        big = [r for r in slope if r["area"] >= min_arch_area]
        if big:
            pts = np.vstack([r["pts"] for r in big])
            nrm = np.mean([np.array(r["normal"]) for r in big], axis=0)
            nrm /= (np.linalg.norm(nrm) or 1.0)
            emit("AUDIENCE_MAIN", pts, nrm)

    return faces


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

    # DBSCAN eps must scale with the cloud's ACTUAL spacing, not a fixed value.
    # CRITICAL: build the tree on ALL points and only SAMPLE the queries. Building
    # the tree on a random subsample inflates spacing (a sample's nearest neighbour
    # is far away because the sample is sparse), which corrupts eps. Tree-on-all,
    # sample-the-queries gives each query its TRUE nearest neighbour in the cloud.
    from scipy.spatial import cKDTree
    _tree = cKDTree(pts_all)
    _q = pts_all if len(pts_all) <= 20000 else pts_all[
        np.random.default_rng(0).choice(len(pts_all), 20000, replace=False)]
    _spacing = float(np.median(_tree.query(_q, k=2)[0][:, 1]))   # k=2: skip self (dist 0)
    eps = max(P["dbscan_eps_m"], 5.0 * _spacing)
    print(f"  loaded {raw_n:,} pts -> {len(pts_all):,} after voxel {voxel} m "
          f"(spacing {_spacing:.3f} m, dbscan eps {eps:.2f} m)")

    # Adaptive minimum plane size: on a 13M-point hall, a fixed 800 lets RANSAC
    # peel hundreds of tiny ceiling fragments and exhaust max_planes BEFORE reaching
    # the (smaller) walls. Scale the floor with cloud size so big clouds only peel
    # large planes (real floor/ceiling/walls), leaving budget to reach verticals.
    # Small clouds (tests) keep the base value.
    min_pp = max(P["min_plane_points"], int(0.0008 * len(pts_all)))
    if min_pp != P["min_plane_points"]:
        print(f"  min_plane_points: {P['min_plane_points']} -> {min_pp} (adaptive, {len(pts_all):,} pts)")

    raw_clusters = []
    rest = pcd
    for i in range(P["max_planes"]):
        if len(rest.points) < min_pp:
            break
        model, inl = rest.segment_plane(
            P["ransac_dist_m"], P["ransac_n"], P["ransac_iters"])
        if len(inl) < min_pp:
            break
        plane = rest.select_by_index(inl)
        rest = rest.select_by_index(inl, invert=True)
        normal = np.array(model[:3]); normal /= np.linalg.norm(normal)

        # split coplanar-but-separate patches into clusters
        labels = np.array(plane.cluster_dbscan(
            eps=eps, min_points=P["dbscan_min_points"]))
        ppts = np.asarray(plane.points)
        for lab in sorted(set(labels)):
            if lab < 0:
                continue
            cpts = ppts[labels == lab]
            if len(cpts) < min_pp:
                continue
            raw_clusters.append({"normal": normal, "pts": cpts})

    # ---- DIAGNOSTIC: what did RANSAC actually find? ----
    cl = cfg["classify"]
    vcos, hcos = cl["vertical_cos"], cl["horizontal_cos"]
    nh = nv = ns = 0
    vspans = []
    for r in raw_clusters:
        nz = abs(np.array(r["normal"])[2])
        zspan = r["pts"][:, 2].max() - r["pts"][:, 2].min()
        if nz >= vcos: nh += 1
        elif nz <= hcos:
            nv += 1; vspans.append((round(float(zspan), 1),
                                    [round(float(x), 1) for x in r["pts"].mean(0)]))
        else: ns += 1
    print(f"  RAW PLANES: {len(raw_clusters)}  | horizontal={nh} vertical={nv} sloped={ns}")
    print(f"    (vcos>={vcos} horiz, <= {hcos} vert)")
    if vspans:
        print(f"  VERTICAL planes (zspan, centroid): {vspans[:12]}")
    else:
        print("  >> NO VERTICAL PLANES FOUND — walls not in RANSAC output, not a classifier issue")

    # staged classification over the WHOLE set (height-band, merge, walls, furniture guard)
    faces = staged_classify(raw_clusters, bounds, cfg)

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