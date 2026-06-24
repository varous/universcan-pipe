#!/usr/bin/env python3
"""
crop.py — automated density crop / denoise for splat-derived clouds.

A splat's extracted centres are a dense real core (or several) buried in wide
sparse spray. This finds the dense region(s) and keeps them, discarding floaters
— WITHOUT assuming a single region. Multiple genuine clusters separated by sparse
zones are all retained; only sparse noise and small junk pockets are dropped.

Pipeline: downsample -> radius-outlier-removal (kill isolated floaters)
          -> DBSCAN (connectivity) -> keep every significant cluster.

Keep rule (the heart): a cluster is kept iff its size >= max(min_cluster_points,
keep_rel_to_largest * largest_cluster_size). This retains a gym + a smaller annex
while dropping noise pockets — the case a naive 'keep densest blob' breaks.

CLI:  python crop.py in.ply out.ply
"""
import sys
import numpy as np
import open3d as o3d


def auto_crop(src, dst,
              work_voxel_m=0.25,
              radius_outlier_nb=8,
              radius_outlier_r_m=0.75,
              dbscan_eps_m=1.0,
              dbscan_min_points=20,
              keep_rel_to_largest=0.15,
              min_cluster_points=500) -> dict:
    pcd = o3d.io.read_point_cloud(src)
    n_in = len(pcd.points)
    if n_in == 0:
        raise ValueError(f"no points in {src}")

    # 1. downsample so clustering is tractable AND density is uniform (so the
    #    floater-vs-surface distinction is by neighbour count, not raw density)
    work = pcd.voxel_down_sample(work_voxel_m)
    n_work = len(work.points)

    # 2. strip isolated floaters: drop points with < nb neighbours within radius
    work, keep_idx = work.remove_radius_outlier(nb_points=radius_outlier_nb,
                                                radius=radius_outlier_r_m)
    n_after_outlier = len(work.points)
    if n_after_outlier == 0:
        raise ValueError("radius-outlier removal emptied the cloud — relax nb/radius")

    # 3. DBSCAN: connectivity clustering. Sparse gaps => separate clusters;
    #    floaters that survived => label -1 (noise).
    labels = np.array(work.cluster_dbscan(eps=dbscan_eps_m,
                                          min_points=dbscan_min_points))
    pts = np.asarray(work.points)

    valid = labels[labels >= 0]
    if valid.size == 0:
        raise ValueError("no dense cluster found — input may be all noise; "
                         "check the capture or relax dbscan params")

    # 4. size each cluster, apply the keep rule
    ids, counts = np.unique(valid, return_counts=True)
    largest = counts.max()
    floor = max(min_cluster_points, keep_rel_to_largest * largest)
    kept_ids = ids[counts >= floor]
    dropped_ids = ids[counts < floor]

    keep_mask = np.isin(labels, kept_ids)
    kept_pts = pts[keep_mask]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept_pts)
    if work.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(work.colors)[keep_mask])
    o3d.io.write_point_cloud(dst, out)

    # per-cluster geometry for the report (so you can see WHAT was kept)
    def bbox_of(label):
        p = pts[labels == label]
        return (p.max(0) - p.min(0)).round(1).tolist(), len(p)

    kept_info = []
    for cid in kept_ids[np.argsort(-counts[np.isin(ids, kept_ids)])]:
        ext, cnt = bbox_of(cid)
        kept_info.append({"points": int(cnt), "extent_m": ext})

    return {
        "src": src, "dst": dst,
        "n_in": int(n_in),
        "n_work": int(n_work),
        "n_after_outlier": int(n_after_outlier),
        "n_clusters_found": int(len(ids)),
        "n_clusters_kept": int(len(kept_ids)),
        "n_clusters_dropped": int(len(dropped_ids)),
        "n_noise_points": int((labels < 0).sum()),
        "n_kept": int(len(kept_pts)),
        "kept_clusters": kept_info,
        "final_extent_m": (kept_pts.max(0) - kept_pts.min(0)).round(1).tolist(),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python crop.py in.ply out.ply")
        sys.exit(1)
    r = auto_crop(sys.argv[1], sys.argv[2])
    print(f"  in ................ {r['n_in']:,} pts")
    print(f"  after downsample .. {r['n_work']:,}")
    print(f"  after de-floater .. {r['n_after_outlier']:,}  "
          f"(noise points: {r['n_noise_points']:,})")
    print(f"  clusters .......... found {r['n_clusters_found']}, "
          f"kept {r['n_clusters_kept']}, dropped {r['n_clusters_dropped']}")
    for i, c in enumerate(r['kept_clusters']):
        print(f"     kept #{i+1}: {c['points']:,} pts, extent {c['extent_m']} m")
    print(f"  kept .............. {r['n_kept']:,} pts")
    print(f"  final extent ...... {r['final_extent_m']} m")
    print(f"  -> {r['dst']}")
