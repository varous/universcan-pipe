"""app/pipeline.py — thin adapter over the existing CLI modules so the API and
the command line share exactly one implementation."""
import os, sys

# repo root (parent of app/) must be importable for inspect_ply / abstract
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inspect_ply import report as _report          # noqa: E402
from abstract import run as _run                    # noqa: E402
from splat import is_splat_ply, extract_centres     # noqa: E402
from crop import auto_crop as _auto_crop            # noqa: E402
import yaml                                          # noqa: E402

TAGS_PATH = os.path.join(ROOT, "tags.yaml")


def _cfg(key, tags_path=TAGS_PATH):
    with open(tags_path) as fh:
        return (yaml.safe_load(fh) or {}).get(key, {})


def inspect_file(path: str) -> dict:
    return _report(path, as_json=None)


def is_splat(path: str) -> bool:
    return is_splat_ply(path)


def extract_splat(src: str, dst: str, tags_path=TAGS_PATH) -> dict:
    """Filter a 3DGS .ply to geometry centres using tags.yaml:splat params."""
    c = _cfg("splat", tags_path)
    return extract_centres(
        src, dst,
        opacity_thresh=c.get("opacity_thresh", 0.5),
        scale_pct=c.get("scale_pct", 99.0),
        scale_max=c.get("scale_max_m"),
        color=c.get("color_from_sh", True),
    )


def crop_enabled(tags_path=TAGS_PATH) -> bool:
    return bool(_cfg("crop", tags_path).get("enabled", False))


def auto_crop_file(src: str, dst: str, tags_path=TAGS_PATH) -> dict:
    """Density-crop a cloud to its significant dense region(s) per tags.yaml:crop."""
    c = _cfg("crop", tags_path)
    return _auto_crop(
        src, dst,
        work_voxel_m=c.get("work_voxel_m", 0.25),
        radius_outlier_nb=c.get("radius_outlier_nb", 8),
        radius_outlier_r_m=c.get("radius_outlier_r_m", 0.75),
        dbscan_eps_m=c.get("dbscan_eps_m", 1.0),
        dbscan_min_points=c.get("dbscan_min_points", 20),
        keep_rel_to_largest=c.get("keep_rel_to_largest", 0.15),
        min_cluster_points=c.get("min_cluster_points", 500),
    )


def abstract_file(ply_path: str, out_dir: str, voxel=None, tags_path: str = TAGS_PATH) -> dict:
    return _run(ply_path, tags_path, out_dir, voxel)