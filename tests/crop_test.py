"""Auto-crop: multi-cluster + spray must keep BOTH dense regions, drop floaters."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crop import auto_crop

def test_multi_cluster():
    src = os.path.join(ROOT, "synth_noisy.ply")
    if not os.path.exists(src):
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "make_synth_noisy.py")], check=True)
    r = auto_crop(src, os.path.join(ROOT, "synth_noisy_cropped.ply"))
    print("clusters found/kept:", r["n_clusters_found"], "/", r["n_clusters_kept"])
    print("kept:", [(c["points"], c["extent_m"]) for c in r["kept_clusters"]])
    print("dropped to:", r["n_kept"], "pts, extent", r["final_extent_m"])
    # both dense regions (gym + annex) kept; ~60k spray dropped
    assert r["n_clusters_kept"] == 2, "should keep BOTH dense clusters"
    assert r["n_kept"] < 60000, "spray should be dropped"
    # the gym cluster should be ~40x25 (largest kept), annex smaller
    assert r["kept_clusters"][0]["points"] > r["kept_clusters"][1]["points"]
    print("\nMULTI-CLUSTER CROP: PASSED ✔")

if __name__ == "__main__":
    test_multi_cluster()
