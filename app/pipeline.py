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
import yaml                                          # noqa: E402

TAGS_PATH = os.path.join(ROOT, "tags.yaml")


def _splat_cfg(tags_path=TAGS_PATH):
    with open(tags_path) as fh:
        return (yaml.safe_load(fh) or {}).get("splat", {})


def inspect_file(path: str) -> dict:
    return _report(path, as_json=None)


def is_splat(path: str) -> bool:
    return is_splat_ply(path)


def extract_splat(src: str, dst: str, tags_path=TAGS_PATH) -> dict:
    """Filter a 3DGS .ply to geometry centres using tags.yaml:splat params."""
    c = _splat_cfg(tags_path)
    return extract_centres(
        src, dst,
        opacity_thresh=c.get("opacity_thresh", 0.5),
        scale_pct=c.get("scale_pct", 99.0),
        scale_max=c.get("scale_max_m"),
        color=c.get("color_from_sh", True),
    )


def abstract_file(ply_path: str, out_dir: str, voxel=None, tags_path: str = TAGS_PATH) -> dict:
    return _run(ply_path, tags_path, out_dir, voxel)