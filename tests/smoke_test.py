"""tests/smoke_test.py — boot the API against mongomock and exercise the full flow."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mongomock
from fastapi.testclient import TestClient

from app import db
db.init_db(mongomock.MongoClient(), db_name="scanpipe_test")   # inject mock BEFORE app import

from app import main
main.app.router.on_startup.clear()                              # stop real Mongo init
client = TestClient(main.app)

PLY = os.path.join(os.path.dirname(__file__), "..", "synth_venue.ply")


def test_full_flow():
    # health
    r = client.get("/health"); assert r.status_code == 200, r.text
    print("health:", r.json())

    # create venue
    r = client.post("/venues", json={
        "name": "Test Auditorium", "city": "Kolkata",
        "location": {"lat": 22.57, "lng": 88.36},
        "capacity": 1200, "venue_class": "theatre",
        "stage_dims_m": [12, 8, 9]})
    assert r.status_code == 200, r.text
    vid = r.json()["_id"]
    print("venue:", vid)

    # post a scan (upload the synthetic .ply)
    with open(PLY, "rb") as fh:
        r = client.post("/scans",
                        data={"venue_id": vid, "voxel": "0.05", "device": "Lixel K2"},
                        files={"file": ("synth_venue.ply", fh, "application/octet-stream")})
    assert r.status_code == 200, r.text
    scan = r.json(); sid = scan["_id"]
    print("scan:", sid, "faces:", scan["face_counts"])
    assert scan["n_faces"] == 7
    assert set(scan["face_counts"]) >= {"FLOOR", "CEILING", "STAGE", "AUDIENCE_MAIN"}

    # fetch scan
    r = client.get(f"/scans/{sid}"); assert r.status_code == 200
    assert r.json()["n_faces"] == 7

    # venue now lists the scan
    r = client.get(f"/venues/{vid}"); assert r.status_code == 200
    assert len(r.json()["scans"]) == 1

    # download dxf
    r = client.get(f"/scans/{sid}/dxf"); assert r.status_code == 200
    assert b"SECTION" in r.content or b"3DFACE" in r.content
    print("dxf bytes:", len(r.content))

    # manifest
    r = client.get(f"/scans/{sid}/manifest"); assert r.status_code == 200
    assert len(r.json()["faces"]) == 7

    # add a measurement (paired-data hook)
    r = client.post(f"/scans/{sid}/measurements", json={
        "seat_label": "R12-S8", "position": [10, 9, 1.2],
        "prediction_tool": "MAPP", "predicted_spl_db": 98.5, "measured_spl_db": 96.1,
        "ir_metrics": {"rt60": 1.7, "c80": 2.3}})
    assert r.status_code == 200, r.text
    print("measurement:", r.json()["_id"])

    print("\nALL ENDPOINTS PASSED ✔")


if __name__ == "__main__":
    test_full_flow()
