"""app/pipeline.py — thin adapter over the existing CLI modules so the API and
the command line share exactly one implementation."""
import os, sys

# repo root (parent of app/) must be importable for inspect_ply / abstract
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inspect_ply import report as _report          # noqa: E402
from abstract import run as _run                    # noqa: E402

TAGS_PATH = os.path.join(ROOT, "tags.yaml")


def inspect_file(path: str) -> dict:
    return _report(path, as_json=None)


def abstract_file(ply_path: str, out_dir: str, voxel=None, tags_path: str = TAGS_PATH) -> dict:
    return _run(ply_path, tags_path, out_dir, voxel)
