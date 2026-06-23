"""
app/db.py — MongoDB layer for the venue scan database.
Sync pymongo on purpose: the pipeline work is CPU-bound and runs in FastAPI's
threadpool via `def` endpoints, so async Mongo would buy nothing but complexity.
init_db() accepts an injected client so tests can pass mongomock.
"""
import os
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE

_db = None


def now():
    return datetime.now(timezone.utc)


def init_db(client=None, db_name=None):
    global _db
    if client is None:
        uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _db = client[db_name or os.environ.get("MONGO_DB", "scanpipe")]
    try:
        _ensure_indexes(_db)
    except Exception:
        pass
    return _db


def get_db():
    if _db is None:
        init_db()
    return _db


def _ensure_indexes(db):
    # Wrapped: mongomock tolerates most of these but not all geo internals.
    try:
        db.venues.create_index([("name", ASCENDING), ("city", ASCENDING)], unique=True)
        db.venues.create_index([("location", GEOSPHERE)])
        db.scans.create_index([("venue_id", ASCENDING), ("scan_date", DESCENDING)])
        db.measurements.create_index([("scan_id", ASCENDING)])
        db.measurements.create_index([("venue_id", ASCENDING)])
    except Exception:
        pass


# ---------- serialization ----------
def ser(doc):
    """Recursively stringify ObjectIds and datetimes for JSON responses."""
    if doc is None:
        return None
    from bson import ObjectId
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, list):
            out[k] = [ser(x) if isinstance(x, dict) else
                      (str(x) if isinstance(x, ObjectId) else x) for x in v]
        elif isinstance(v, dict):
            out[k] = ser(v)
        else:
            out[k] = v
    return out


# ---------- venues ----------
def upsert_venue(db, v: dict):
    """Create or update a venue keyed on (name, city). Returns the venue doc."""
    key = {"name": v["name"], "city": v["city"]}
    update = {k: val for k, val in v.items() if val is not None}
    if "location" in v and v.get("location"):
        lat, lng = v["location"]["lat"], v["location"]["lng"]
        update["location"] = {"type": "Point", "coordinates": [lng, lat]}  # GeoJSON: [lng,lat]
    update["updated_at"] = now()
    db.venues.update_one(key, {"$set": update,
                               "$setOnInsert": {"created_at": now(), "scan_ids": []}},
                         upsert=True)
    return db.venues.find_one(key)


def get_venue(db, vid):
    from bson import ObjectId
    return db.venues.find_one({"_id": ObjectId(vid)})


def list_venues(db, limit=100):
    return list(db.venues.find().sort("name", ASCENDING).limit(limit))


def venues_near(db, lat, lng, km):
    """Geo query — requires a real MongoDB (2dsphere). Not supported under mongomock."""
    return list(db.venues.find({"location": {"$near": {
        "$geometry": {"type": "Point", "coordinates": [lng, lat]},
        "$maxDistance": km * 1000}}}).limit(50))


# ---------- scans ----------
def create_scan(db, scan: dict):
    from bson import ObjectId
    scan["created_at"] = now()
    res = db.scans.insert_one(scan)
    db.venues.update_one({"_id": ObjectId(scan["venue_id"])},
                         {"$push": {"scan_ids": res.inserted_id},
                          "$set": {"updated_at": now()}})
    return db.scans.find_one({"_id": res.inserted_id})


def get_scan(db, sid):
    from bson import ObjectId
    return db.scans.find_one({"_id": ObjectId(sid)})


def scans_for_venue(db, vid):
    return list(db.scans.find({"venue_id": vid}).sort("scan_date", DESCENDING))


# ---------- measurements (the paired-data / moat hook) ----------
def add_measurement(db, m: dict):
    m["created_at"] = now()
    res = db.measurements.insert_one(m)
    return db.measurements.find_one({"_id": res.inserted_id})
