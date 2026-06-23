import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mongomock
from fastapi.testclient import TestClient
from app import db
db.init_db(mongomock.MongoClient(), db_name="scanpipe_splat_test")
from app import main
main.app.router.on_startup.clear()
client = TestClient(main.app)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_splat_routing():
    v = client.post("/venues", json={"name":"Splat Hall","city":"Kolkata"}).json()
    vid = v["_id"]
    with open(os.path.join(ROOT,"synth_splat.ply"),"rb") as fh:
        r = client.post("/scans", data={"venue_id":vid,"voxel":"0.05"},
                        files={"file":("synth_splat.ply",fh,"application/octet-stream")})
    assert r.status_code == 200, r.text
    s = r.json()
    print("is_splat_derived:", s["is_splat_derived"])
    print("extraction:", s["splat_extraction"]["n_gaussians"], "->", s["splat_extraction"]["n_kept"])
    print("face_counts:", s["face_counts"])
    assert s["is_splat_derived"] is True
    assert s["splat_extraction"]["n_kept"] < s["splat_extraction"]["n_gaussians"]
    assert s["n_faces"] >= 5

def test_bad_file_422():
    # a .laz-like binary that isn't a PLY -> must be a clean 422, not a 500
    r = client.post("/inspect",
                    files={"file":("x.laz", b"LASF\x00\x00garbage", "application/octet-stream")})
    print("bad-file status:", r.status_code)
    assert r.status_code == 422, r.text

if __name__ == "__main__":
    test_splat_routing()
    test_bad_file_422()
    print("\nSPLAT PATH + GRACEFUL ERROR: PASSED ✔")
