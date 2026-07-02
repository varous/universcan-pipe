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


class StageLog:
    """Readable terminal trace + structured debug.json sidecar. Every surface that
    is kept or dropped is recorded WITH ITS REASON, so a single run shows exactly
    why each plane became a face or was discarded."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.data = {"stages": {}}
        self._cur = None

    def stage(self, name):
        self._cur = name
        self.data["stages"][name] = []
        if self.enabled:
            print(f"\n  ── {name} ──")

    def _push(self, rec):
        if self._cur:
            self.data["stages"][self._cur].append(rec)

    def info(self, msg, **kv):
        self._push({"kind": "info", "msg": msg, **kv})
        if self.enabled:
            extra = "  ".join(f"{k}={v}" for k, v in kv.items())
            print(f"     {msg}" + (f"   {extra}" if extra else ""))

    def keep(self, what, tag, **kv):
        self._push({"kind": "keep", "what": what, "tag": tag, **kv})
        if self.enabled:
            extra = " ".join(f"{k}={v}" for k, v in kv.items())
            print(f"     [KEEP] {what:20s} -> {tag:14s} {extra}")

    def drop(self, what, reason, **kv):
        self._push({"kind": "drop", "what": what, "reason": reason, **kv})
        if self.enabled:
            extra = " ".join(f"{k}={v}" for k, v in kv.items())
            print(f"     [DROP] {what:20s} -- {reason:28s} {extra}")

    def write(self, path):
        try:
            with open(path, "w") as fh:
                json.dump(self.data, fh, indent=2, default=str)
        except Exception:
            pass


def detect_walls_hist(pts_all, band_z, band_tol, bounds, min_pp, eps, cfg, log=None):
    """Find walls STATISTICALLY as peaks in the X and Y histograms — no normals.

    A wall is SPARSE in 3D (few points relative to floor/ceiling) so RANSAC ranks it
    below dense horizontal clutter and never reaches it. But a wall perpendicular to X
    has ALL its points at nearly constant X: projected onto the X axis, every point at
    every height collapses onto one bin -> a sharp PEAK, dense despite being sparse in 3D.
    Same statistical principle as the z-histogram (floor/ceiling = peaks in Z):
        floor/ceiling = peaks in Z    left/right walls = peaks in X    front/rear = peaks in Y
    For each peak we RE-ASSOCIATE the full-cloud points at that coordinate (full height,
    floor->ceiling) and validate wall-ness by z-span and perpendicular length.
    Assumes axis-aligned walls (venues usually are; PCA-align first if not).
    """
    mn, mx = np.array(bounds[0]), np.array(bounds[1])
    z = pts_all[:, 2]
    floor_z, ceil_z = (band_z[0], band_z[-1]) if band_z else (float(z.min()), float(z.max()))
    room_h = ceil_z - floor_z
    # Strip ONLY floor & ceiling (the dense surfaces that dominate the X/Y baseline), with a
    # tolerance scaled to room height so a short room isn't wiped out. Mid-level slabs are
    # left in: a horizontal slab spreads uniformly across X/Y, so it can't fake a wall peak.
    strip_tol = min(band_tol, max(0.3, room_h * 0.12))
    keep = (np.abs(z - floor_z) > strip_tol) & (np.abs(z - ceil_z) > strip_tol)
    mid = pts_all[keep]
    if len(mid) < min_pp:                       # fallback: don't strip, higher baseline is ok
        mid = pts_all
        if log: log.info("wall-hist: strip left too few pts -> using full cloud")
    min_wall_h = cfg["classify"].get("min_wall_height_frac", 0.3) * room_h
    min_len_frac = cfg["classify"].get("min_wall_length_frac", 0.25)
    bin_m = cfg["classify"].get("wall_hist_bin_m", 0.1)
    tol = cfg["classify"].get("wall_assoc_tol_m", 0.15)

    clusters = []
    for axis in (0, 1):                       # 0=X -> left/right walls, 1=Y -> front/rear
        span = 1 - axis
        lo, hi = float(mn[axis]), float(mx[axis])
        nb = max(10, int((hi - lo) / bin_m))
        hist, edges = np.histogram(mid[:, axis], bins=nb, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2
        nz = hist[hist > 0]
        if len(nz) == 0:
            continue
        # a wall bin stands far above the (stripped) baseline
        thr = max(hist.max() * 0.20, float(np.median(nz)) * 4)
        above = hist >= thr
        runs, i = [], 0
        while i < len(hist):                  # merge contiguous bins into one peak
            if above[i]:
                j = i
                while j < len(hist) and above[j]:
                    j += 1
                w = hist[i:j]; c = centers[i:j]
                runs.append(float((c * w).sum() / w.sum()))
                i = j
            else:
                i += 1
        axname = "X" if axis == 0 else "Y"
        if log: log.info(f"wall-hist {axname}: {len(runs)} peak(s) (thr={thr:.0f})")
        for pos in runs:
            wp = pts_all[np.abs(pts_all[:, axis] - pos) <= tol]   # re-associate FULL height
            if len(wp) < min_pp:
                continue
            zspan = float(wp[:, 2].max() - wp[:, 2].min())
            perp = float(wp[:, span].max() - wp[:, span].min())
            perp_frac = perp / max(float(mx[span] - mn[span]), 1e-6)
            if zspan >= min_wall_h and perp_frac >= min_len_frac:
                kind = "WALL"
            elif zspan >= min_wall_h:
                kind = "column->STRUCTURE"
            else:
                kind = "skip (too short)"
            if log:
                log.info(f"  {axname}={pos:6.1f}  zspan={zspan:4.1f}m perp_frac={perp_frac:.2f} -> {kind}")
            if kind.startswith("skip"):
                continue
            normal = np.zeros(3); normal[axis] = 1.0
            clusters.append({"normal": normal, "pts": wp})
    return clusters


def staged_classify(raw, bounds, cfg, log=None):
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

    log = log or StageLog(enabled=False)
    horiz, vert, slope = [], [], []
    for r in raw:
        n = np.array(r["normal"]); nz = abs(n[2])
        r["zc"] = float(r["pts"][:, 2].mean())
        r["zspan"] = float(r["pts"][:, 2].max() - r["pts"][:, 2].min())
        r["area"] = _plane_area(r["pts"], n)
        (horiz if nz >= vcos else vert if nz <= hcos else slope).append(r)

    log.stage("orientation_split")
    log.info(f"{len(raw)} planes", horizontal=len(horiz), vertical=len(vert),
             sloped=len(slope))

    faces = []

    def emit(tag, pts, normal):
        corners, centroid = oriented_quad(pts, normal)
        faces.append({"tag": tag, "normal": list(np.round(normal, 4)),
                      "centroid": list(np.round(centroid, 3)),
                      "n_points": int(len(pts)),
                      "vertices": [list(np.round(v, 3)) for v in corners]})

    # ---- 2. HORIZONTALS: cluster into z-bands, merge, tag by band order ----
    log.stage("height_bands")
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
                          "area": _plane_area(pts, np.array([0, 0, 1.0])),
                          "n": int(len(pts)), "frags": len(b["items"])})
    log.info(f"{len(horiz)} horizontal planes -> {len(bands)} z-bands")
    # large bands are architectural; small mid-height bands are furniture
    arch = []
    for b in band_recs:
        if b["area"] >= min_arch_area:
            arch.append(b)
        else:
            log.drop(f"band z={b['zc']:.1f}", "area < min_arch (furniture)",
                     area=f"{b['area']:.1f}", min=min_arch_area, n=b["n"])
    arch.sort(key=lambda b: b["zc"])
    floor_zc = ceil_zc = None
    if arch:
        floor_zc = arch[0]["zc"]
        ceil_zc = arch[-1]["zc"]
        for i, b in enumerate(arch):
            up = np.array([0, 0, 1.0])
            if b["zc"] <= floor_zc + band_tol:
                tag = "FLOOR"
            elif b["zc"] >= ceil_zc - ceil_span:
                tag = "CEILING"
            else:
                tag = "STRUCTURE"
            log.keep(f"band z={b['zc']:.1f}", tag, area=f"{b['area']:.0f}m2",
                     n=b["n"], frags=b["frags"])
            emit(tag, b["pts"], up)
    else:
        log.info("no architectural horizontal bands found")

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
    log.stage("walls")
    log.info(f"room_height={room_height:.1f}m  min_wall_height={min_wall_h:.1f}m  "
             f"perim_band: x={perim_frac*W:.1f}m y={perim_frac*D:.1f}m")
    wall_groups = {"WALL_LEFT": [], "WALL_RIGHT": [],
                   "WALL_FRONT": [], "WALL_REAR": [], "STRUCTURE": []}
    for r in vert:
        c = r["pts"].mean(0)
        cstr = f"[{c[0]:.0f},{c[1]:.0f},{c[2]:.0f}]"
        if r["zspan"] < min_wall_h:
            log.drop(f"vert {cstr}", "too short (zspan<min_wall_h)",
                     zspan=f"{r['zspan']:.1f}", need=f"{min_wall_h:.1f}")
            continue
        n = np.array(r["normal"])
        # distances to each bbox edge (m)
        dxl, dxh = c[0] - mn[0], mx[0] - c[0]
        dyl, dyh = c[1] - mn[1], mx[1] - c[1]
        if abs(n[0]) >= abs(n[1]):                       # normal along X -> L/R wall
            if dxl <= perim_frac * W:
                wall_groups["WALL_LEFT"].append(r)
                log.keep(f"vert {cstr}", "WALL_LEFT", dist_to_edge=f"{dxl:.1f}m")
            elif dxh <= perim_frac * W:
                wall_groups["WALL_RIGHT"].append(r)
                log.keep(f"vert {cstr}", "WALL_RIGHT", dist_to_edge=f"{dxh:.1f}m")
            elif r["area"] >= min_arch_area:
                wall_groups["STRUCTURE"].append(r)
                log.keep(f"vert {cstr}", "STRUCTURE",
                         reason="interior (not near x-edge)",
                         nearest_x_edge=f"{min(dxl,dxh):.1f}m",
                         need=f"{perim_frac*W:.1f}m")
            else:
                log.drop(f"vert {cstr}", "interior + small",
                         nearest_x_edge=f"{min(dxl,dxh):.1f}m", area=f"{r['area']:.0f}")
        else:                                            # normal along Y -> F/R wall
            if dyl <= perim_frac * D:
                wall_groups["WALL_FRONT"].append(r)
                log.keep(f"vert {cstr}", "WALL_FRONT", dist_to_edge=f"{dyl:.1f}m")
            elif dyh <= perim_frac * D:
                wall_groups["WALL_REAR"].append(r)
                log.keep(f"vert {cstr}", "WALL_REAR", dist_to_edge=f"{dyh:.1f}m")
            elif r["area"] >= min_arch_area:
                wall_groups["STRUCTURE"].append(r)
                log.keep(f"vert {cstr}", "STRUCTURE",
                         reason="interior (not near y-edge)",
                         nearest_y_edge=f"{min(dyl,dyh):.1f}m",
                         need=f"{perim_frac*D:.1f}m")
            else:
                log.drop(f"vert {cstr}", "interior + small",
                         nearest_y_edge=f"{min(dyl,dyh):.1f}m", area=f"{r['area']:.0f}")
    # ---- 4. merge fragments per side -> one face per wall ----
    for tag, items in wall_groups.items():
        if not items:
            continue
        pts = np.vstack([r["pts"] for r in items])
        nrm = np.mean([np.array(r["normal"]) for r in items], axis=0)
        nrm /= (np.linalg.norm(nrm) or 1.0)
        if len(items) > 1:
            log.info(f"merged {len(items)} fragments -> {tag}")
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


def run(ply_path, tags_path, out_dir, voxel_override=None, debug=False):
    cfg = load_cfg(tags_path)
    P = cfg["pipeline"]
    voxel = voxel_override or P["voxel_m"]
    os.makedirs(out_dir, exist_ok=True)
    debug = debug or bool(cfg.get("debug", False))
    log = StageLog(enabled=debug)

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
    log.stage("load")
    log.info(f"{raw_n:,} pts -> {len(pts_all):,} after voxel {voxel}m",
             spacing=f"{_spacing:.3f}m", eps=f"{eps:.2f}m",
             bbox=[round(float(x), 1) for x in bounds[1]])
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

    log.stage("ransac_peel")

    def _peel(subset, budget, label):
        """Peel planes from a point subset with its own budget. Returns raw clusters."""
        clusters = []
        rest = subset
        for i in range(budget):
            if len(rest.points) < min_pp:
                break
            model, inl = rest.segment_plane(
                P["ransac_dist_m"], P["ransac_n"], P["ransac_iters"])
            if len(inl) < min_pp:
                break
            plane = rest.select_by_index(inl)
            rest = rest.select_by_index(inl, invert=True)
            normal = np.array(model[:3]); normal /= np.linalg.norm(normal)
            nz = abs(normal[2])
            ori = "H" if nz >= cfg["classify"]["vertical_cos"] else \
                  "V" if nz <= cfg["classify"]["horizontal_cos"] else "S"
            log.info(f"{label} plane {i:3d}", inliers=len(inl), ori=ori,
                     normal=[round(float(x), 2) for x in normal])
            labels = np.array(plane.cluster_dbscan(
                eps=eps, min_points=P["dbscan_min_points"]))
            ppts = np.asarray(plane.points)
            for lab in sorted(set(labels)):
                if lab < 0:
                    continue
                cpts = ppts[labels == lab]
                if len(cpts) < min_pp:
                    continue
                clusters.append({"normal": normal, "pts": cpts})
        return clusters

    def _peel_vmodels(subset, budget):
        """Peel from a subset, returning VERTICAL plane MODELS (a,b,c,d) only.
        Used for wall detection on the z-stripped (mid-height) cloud."""
        models = []
        rest = subset
        for i in range(budget):
            if len(rest.points) < min_pp:
                break
            model, inl = rest.segment_plane(
                P["ransac_dist_m"], P["ransac_n"], P["ransac_iters"])
            if len(inl) < min_pp:
                break
            rest = rest.select_by_index(inl, invert=True)
            normal = np.array(model[:3]); normal /= np.linalg.norm(normal)
            nz = abs(normal[2])
            ori = "H" if nz >= cfg["classify"]["vertical_cos"] else \
                  "V" if nz <= cfg["classify"]["horizontal_cos"] else "S"
            log.info(f"Vdet plane {i:3d}", inliers=len(inl), ori=ori)
            if nz <= cfg["classify"]["horizontal_cos"]:      # keep verticals only
                models.append(np.array(model))
        return models

    # ---- FLOOR/CEILING via Z-HISTOGRAM (normal-free, robust) ----
    # Per-point normals are too noisy to reliably route a complex/trussed ceiling into a
    # horizontal pool. But the floor and ceiling are simply where points PILE UP in height
    # -- the big peaks in the z-distribution. This uses only the measured z coordinate.
    band_tol = cfg["classify"].get("band_tol_m", 1.0)
    strip_margin = P.get("wall_strip_margin_m", 1.0)
    z = pts_all[:, 2]
    lo, hi = float(z.min()), float(z.max())
    nb = max(10, int((hi - lo) / 0.3))
    hist, edges = np.histogram(z, bins=nb)
    centers = (edges[:-1] + edges[1:]) / 2
    thr = 0.08 * hist.max()                           # a "surface level" = >=8% of peak bin
    zb = []
    for c, h in zip(centers, hist):
        if h < thr:
            continue
        if zb and c - zb[-1]["zmax"] <= 0.6:
            zb[-1]["items"].append((c, h)); zb[-1]["zmax"] = c
        else:
            zb.append({"items": [(c, h)], "zmax": c})
    band_z = sorted(sum(c * h for c, h in b["items"]) / sum(h for c, h in b["items"])
                    for b in zb)
    log.info(f"z-histogram surface levels: {[round(float(x), 1) for x in band_z]}")

    # horizontal clusters = points at each surface level (feeds FLOOR/CEILING/STRUCTURE)
    horiz_clusters = []
    for bz in band_z:
        hp = pts_all[np.abs(z - bz) <= band_tol]
        if len(hp) >= min_pp:
            horiz_clusters.append({"normal": np.array([0, 0, 1.0]), "pts": hp})
    floor_z = band_z[0] if band_z else None
    ceil_z = band_z[-1] if band_z else None

    # ---- WALLS via X/Y HISTOGRAMS (statistical, normal-free) ----
    # RANSAC ranks planes by inlier count, so sparse walls always lose to dense horizontal
    # clutter — even with a separate budget. Instead: walls perp to X are PEAKS in the X
    # histogram, perp to Y are peaks in Y. Re-associate full-cloud points at each peak so
    # walls span floor->ceiling. (Same idea as floor/ceiling = peaks in Z.)
    vert_clusters = detect_walls_hist(pts_all, band_z, band_tol, bounds,
                                      min_pp, eps, cfg, log=log)

    raw_clusters = horiz_clusters + vert_clusters

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
    if not vspans:
        print("  >> NO VERTICAL PLANES FOUND — walls not in RANSAC output, not a classifier issue")

    # staged classification over the WHOLE set (height-band, merge, walls, furniture guard)
    faces = staged_classify(raw_clusters, bounds, cfg, log=log)

    log.stage("summary")
    from collections import Counter
    log.info("final faces", **{k: v for k, v in Counter(f["tag"] for f in faces).items()})
    if debug:
        dbg_path = os.path.join(out_dir, "debug.json")
        log.write(dbg_path)
        print(f"  debug trace -> {dbg_path}")

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
    ap.add_argument("--debug", action="store_true",
                    help="print per-stage trace and write debug.json")
    a = ap.parse_args()
    run(a.ply, a.tags, a.out, a.voxel, debug=a.debug)