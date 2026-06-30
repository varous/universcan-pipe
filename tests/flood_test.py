"""Flood-fill via API: indoor+outdoor cloud -> keep interior, drop outdoor, find walls."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mongomock
from fastapi.testclient import TestClient
from app import db; db.init_db(mongomock.MongoClient(), db_name="flood_test")
from app import main; main.app.router.on_startup.clear()
client = TestClient(main.app)

def test_flood():
    src = os.path.join(ROOT, "synth_io.ply")
    if not os.path.exists(src):
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "make_synth_indoor_outdoor.py")], check=True)
    v = client.post("/venues", json={"name":"IO","city":"T"}).json()
    with open(src, "rb") as fh:
        r = client.post("/scans", data={"venue_id":v["_id"],"voxel":"0.05"},
                        files={"file":("synth_io.ply",fh,"application/octet-stream")})
    assert r.status_code == 200, r.text
    s = r.json()
    ff = s["flood_fill"]
    print("flood_fill:", {k:ff[k] for k in ["n_in","n_kept","dropped_frac","n_interior_regions_kept"]})
    print("face_counts:", s["face_counts"])
    assert ff["dropped_frac"] > 0.5, "should drop the outdoor majority"
    assert any("WALL" in t for t in s["face_counts"]), "walls should be recovered"
    print("\nFLOOD-FILL INDOOR ISOLATION: PASSED")

if __name__ == "__main__":
    test_flood()
