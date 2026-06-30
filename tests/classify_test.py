"""Staged classifier: multi-level ceiling + furniture -> clean bounding surfaces."""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import abstract

def test_staged():
    src = os.path.join(ROOT, "synth_conv.ply")
    if not os.path.exists(src):
        subprocess.run([sys.executable, os.path.join(ROOT,"make_synth_conv.py")], check=True)
    m = abstract.run(src, os.path.join(ROOT,"tags.yaml"), os.path.join(ROOT,"out_conv"), 0.05)
    from collections import Counter
    c = Counter(f["tag"] for f in m["faces"])
    print("face_counts:", dict(c))
    assert c["FLOOR"] >= 1, "no floor"
    assert c["CEILING"] >= 1, "no ceiling"
    assert sum(v for t,v in c.items() if t.startswith("WALL")) >= 4, "need 4 walls"
    assert m["n_faces"] <= 12, f"furniture not collapsed: {m['n_faces']} faces"
    assert "FURNITURE" not in c, "furniture leaked as faces"
    print("\nSTAGED CLASSIFIER (walls + bounding surfaces, furniture dropped): PASSED")

if __name__ == "__main__":
    test_staged()
