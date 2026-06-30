"""
app/main.py — FastAPI service wrapping the scan pipeline + venue database.

Flow: upload .ply -> run abstraction -> write manifest/dxf to DATA_DIR ->
store a scan record (with embedded faces) in Mongo, linked to a venue.

Big point clouds stay on the local DATA_DIR volume (data gravity); Mongo holds
metadata + the small manifest + file paths. Run the package on the workstation
that holds the clouds — do not ship GB files to a cloud runtime.
"""
import os, shutil, uuid
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app import db
from app.models import VenueIn, MeasurementIn
from app.pipeline import (inspect_file, abstract_file, is_splat, extract_splat,
                          crop_enabled, auto_crop_file, flood_enabled, flood_fill_file)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUT_DIR = os.path.join(DATA_DIR, "out")
for d in (UPLOAD_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="scanpipe", version="0.1.0",
              description="Venue .ply -> tagged surfaces -> venue database")


@app.on_event("startup")
def _startup():
    db.init_db()


@app.get("/health")
def health():
    try:
        db.get_db().command("ping")
        mongo = "ok"
    except Exception as e:
        mongo = f"unreachable: {e.__class__.__name__}"
    return {"status": "ok", "mongo": mongo}


# ---------- inspect (no DB write) ----------
@app.post("/inspect")
def inspect(file: UploadFile = File(...)):
    tmp = os.path.join(UPLOAD_DIR, f"_inspect_{uuid.uuid4().hex}.ply")
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        return inspect_file(tmp)
    except Exception as e:
        raise HTTPException(422, f"could not read as a point cloud "
                                 f"(.ply expected; .las/.laz not yet supported): {e}")
    finally:
        os.remove(tmp)


def _with_urls(doc: dict) -> dict:
    """Add client-usable download URLs derived from the scan's Mongo _id.
    Relative on purpose: the client prepends whatever host it reached us on."""
    sid = doc.get("_id")
    if sid and doc.get("dxf_path"):
        doc["dxf_url"] = f"/scans/{sid}/dxf"
    if sid and doc.get("manifest_path"):
        doc["manifest_url"] = f"/scans/{sid}/manifest"
    return doc


# ---------- venues ----------
@app.post("/venues")
def create_venue(v: VenueIn):
    doc = db.upsert_venue(db.get_db(), v.model_dump())
    return db.ser(doc)


@app.get("/venues")
def list_venues(limit: int = 100):
    return [db.ser(x) for x in db.list_venues(db.get_db(), limit)]


@app.get("/venues/{vid}")
def get_venue(vid: str):
    d = db.get_db()
    v = db.get_venue(d, vid)
    if not v:
        raise HTTPException(404, "venue not found")
    out = db.ser(v)
    out["scans"] = [_with_urls(db.ser(s)) for s in db.scans_for_venue(d, str(v["_id"]))]
    return out


@app.get("/venues/near/")
def near(lat: float, lng: float, km: float = 50):
    try:
        return [db.ser(x) for x in db.venues_near(db.get_db(), lat, lng, km)]
    except Exception as e:
        raise HTTPException(400, f"geo query needs a real MongoDB: {e}")


# ---------- scans (the main event) ----------
@app.post("/scans")
def create_scan(
    venue_id: str = Form(...),
    file: UploadFile = File(...),
    scan_date: str = Form(None),
    device: str = Form("Lixel K2"),
    operator: str = Form(None),
    voxel: float = Form(None),
):
    d = db.get_db()
    if not db.get_venue(d, venue_id):
        raise HTTPException(404, "venue_id not found — create the venue first")

    sid = uuid.uuid4().hex
    ply_path = os.path.join(UPLOAD_DIR, f"{sid}.ply")
    with open(ply_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    out_dir = os.path.join(OUT_DIR, sid)
    try:
        summary = inspect_file(ply_path)
    except Exception as e:
        raise HTTPException(422, f"could not read uploaded file as a point cloud: {e}")

    # --- Tier-1 splat routing: if it's a 3DGS splat, extract centres first ---
    splat_info = None
    crop_info = None
    flood_info = None
    cloud_for_abstract = ply_path
    if summary.get("is_gaussian_splat") or is_splat(ply_path):
        derived = os.path.join(UPLOAD_DIR, f"{sid}_centres.ply")
        try:
            splat_info = extract_splat(ply_path, derived)
        except Exception as e:
            raise HTTPException(422, f"splat extraction failed: {e}")
        cloud_for_abstract = derived

        # --- auto density-crop: drop floaters, keep dense region(s) ---
        if crop_enabled():
            cropped = os.path.join(UPLOAD_DIR, f"{sid}_cropped.ply")
            try:
                crop_info = auto_crop_file(derived, cropped)
                cloud_for_abstract = cropped
            except Exception as e:
                raise HTTPException(422, f"auto-crop failed: {e}")

    # --- flood-fill indoor isolation: LiDAR/geometry clouds only (splats use crop) ---
    if flood_enabled() and splat_info is None:
        flooded = os.path.join(UPLOAD_DIR, f"{sid}_interior.ply")
        try:
            flood_info = flood_fill_file(cloud_for_abstract, flooded)
            cloud_for_abstract = flooded
        except Exception as e:
            flood_info = {"skipped": str(e)}   # no interior found -> keep prior cloud

    try:
        manifest = abstract_file(cloud_for_abstract, out_dir, voxel=voxel)
    except Exception as e:
        raise HTTPException(422, f"pipeline failed: {e}")

    from collections import Counter
    counts = dict(Counter(f["tag"] for f in manifest["faces"]))

    scan_doc = {
        "venue_id": venue_id,
        "scan_date": scan_date or datetime.now(timezone.utc).isoformat(),
        "device": device,
        "operator": operator,
        "source_ply_path": ply_path,
        "is_splat_derived": splat_info is not None,
        "splat_extraction": splat_info,        # null for LiDAR; stats for splats
        "auto_crop": crop_info,                # null unless splat-cropped
        "flood_fill": flood_info,              # null unless indoor-isolated
        "units": summary.get("inferred_units"),
        "up_axis": summary.get("inferred_up_axis"),
        "gravity_aligned": summary.get("gravity_aligned_Zup"),
        "n_points_raw": summary.get("n_points"),
        "pipeline_params": {"voxel_m": manifest["voxel_m"]},
        "face_counts": counts,
        "n_faces": manifest["n_faces"],
        "faces": manifest["faces"],            # embedded — small, queryable
        "manifest_path": os.path.join(out_dir, "manifest.json"),
        "dxf_path": os.path.join(out_dir, "venue.dxf"),
        "skp_path": None,                      # filled when build_skp.rb runs
        "splat_path": None,
        "status": "abstracted",
    }
    doc = db.create_scan(d, scan_doc)
    return _with_urls(db.ser(doc))


@app.get("/scans/{sid}")
def read_scan(sid: str):
    s = db.get_scan(db.get_db(), sid)
    if not s:
        raise HTTPException(404, "scan not found")
    return _with_urls(db.ser(s))


@app.get("/scans/{sid}/dxf")
def download_dxf(sid: str):
    s = db.get_scan(db.get_db(), sid)
    if not s or not os.path.exists(s.get("dxf_path", "")):
        raise HTTPException(404, "dxf not found")
    return FileResponse(s["dxf_path"], filename=f"{sid}.dxf",
                        media_type="image/vnd.dxf")


@app.get("/scans/{sid}/manifest")
def read_manifest(sid: str):
    s = db.get_scan(db.get_db(), sid)
    if not s:
        raise HTTPException(404, "scan not found")
    return {"n_faces": s["n_faces"], "faces": s["faces"]}


# ---------- measurements (paired-data hook) ----------
@app.post("/scans/{sid}/measurements")
def add_measurement(sid: str, m: MeasurementIn):
    d = db.get_db()
    s = db.get_scan(d, sid)
    if not s:
        raise HTTPException(404, "scan not found")
    rec = m.model_dump()
    rec["scan_id"] = sid
    rec["venue_id"] = s["venue_id"]
    return db.ser(db.add_measurement(d, rec))