# scanpipe — Deployment & Database RUNBOOK

Build online (Replit), package as a Docker image, run it on the workstation that
holds the point clouds, write results to MongoDB. Follow top to bottom.

---

## 0. The one principle that shapes everything

**Build in the cloud; run the package locally.** Your `.ply` clouds are multi-GB
and your NVIDIA workstation does the processing. Do **not** ship GB files to a
cloud runtime over Indian bandwidth. So: author/iterate on Replit, but the
*artifact* is a Docker image that runs on the workstation, next to the data.

```
ingest .ply ──▶ FastAPI service (in Docker, on the workstation)
                   │  runs inspect_ply + abstract
                   ├─▶ manifest.json + venue.dxf  → DATA_DIR volume (local disk)
                   └─▶ scan record (metadata + faces + paths) → MongoDB (Atlas or local)
```

---

## 1. Repo layout

```
scanpipe/
  app/
    main.py        # FastAPI endpoints
    db.py          # Mongo connection, indexes, CRUD
    models.py      # request models
    pipeline.py    # adapter over inspect_ply + abstract
  inspect_ply.py   # CLI inspector (also imported by the API)
  abstract.py      # CLI pipeline   (also imported by the API)
  tags.yaml        # the canonical tag contract
  build_skp.rb     # SketchUp-side reader (run on a SketchUp box)
  tests/smoke_test.py
  Dockerfile  docker-compose.yml  .replit  replit.nix  .env.example
  requirements.txt  RUNBOOK.md
```

---

## 2. Database setup

### Option A — MongoDB Atlas (recommended for the shared catalog)

1. Create a free/shared cluster at cloud.mongodb.com.
2. **Database Access** → Add user (username + password). Give it `readWrite` on
   the `scanpipe` DB.
3. **Network Access** → Add the workstation's public IP (or `0.0.0.0/0` only for
   a quick test — lock it down after).
4. **Connect** → *Drivers* → copy the SRV string:
   `mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
5. That string is your `MONGO_URI`. Indexes are created automatically on first
   startup (`db.py:_ensure_indexes`), including the `2dsphere` geo index that
   powers "venues near me".

### Option B — Local MongoDB (offline / dev)

`docker compose up` (Section 5) starts a local `mongo:7` automatically; nothing
to configure. Data persists in the `mongodata` volume. Note: the geo `/venues/near`
query needs a real Mongo — it works here and on Atlas, but **not** under the
mongomock used in tests.

---

## 3. The schema

Three collections. Indexes in parentheses.

### `venues` — one document per physical venue
*(unique index on `name`+`city`; `2dsphere` on `location`)*

| field | type | notes |
|---|---|---|
| `_id` | ObjectId | |
| `name`, `city` | string | dedup key |
| `location` | GeoJSON Point | `{type:"Point", coordinates:[lng,lat]}` — note lng first |
| `capacity` | int | |
| `venue_class` | string | theatre / club / arena / outdoor — drives tag tuning |
| `stage_dims_m` | [w,d,h] | |
| `rigging` | [{id, xyz, wll_kg}] | hang points + working load limits |
| `power` | dict | `{phases, tie_ins:[...]}` |
| `loadin_notes`, `sightline_notes` | string | |
| `scan_ids` | [ObjectId] | links to every scan/visit |
| `created_at`, `updated_at` | datetime | |

### `scans` — one document per scan (per visit; venues get re-scanned)
*(index on `venue_id`+`scan_date`)*

| field | type | notes |
|---|---|---|
| `_id` | ObjectId | |
| `venue_id` | string | → venues |
| `scan_date`, `device`, `operator` | | provenance |
| `source_ply_path` | string | raw cloud on DATA_DIR (not in Mongo — too big) |
| `units`, `up_axis`, `gravity_aligned` | | from the inspector |
| `n_points_raw` | int | |
| `pipeline_params` | dict | `{voxel_m}` etc — reproducibility |
| `face_counts` | dict | `{FLOOR:1, WALL_LEFT:1, ...}` |
| `n_faces` | int | |
| `faces` | [ {tag, normal, centroid, n_points, vertices} ] | **embedded** — small + queryable |
| `manifest_path`, `dxf_path`, `skp_path`, `splat_path` | string | artifacts on disk |
| `status` | string | `abstracted` → `validated` once measured |

### `measurements` — the paired-data / moat hook (one per mic position)
*(index on `scan_id`, `venue_id`)*

| field | type | notes |
|---|---|---|
| `scan_id`, `venue_id` | string | links |
| `position`, `seat_label` | | where the mic was |
| `source_config` | string | which array / aim |
| `prediction_tool` | string | MAPP / ArrayCalc / VS / Soundvision |
| `predicted_spl_db`, `measured_spl_db` | float | the delta is the signal |
| `ir_metrics` | dict | `{rt60, c80, reflections:[{t_ms, level_db}]}` |
| `applied_calibration` | dict | `{delays_ms, gains_db, eq:[...]}` ← the calibration-memory asset |
| `attribution` | dict | `{modeled_surfaces, rig, mic_pos}` ← keeps the residual model honest |

> Why embed `faces` but not the cloud: a face list is tens of KB and you'll want
> to query it ("all venues with a rear wall > 30 m back"); a point cloud is GB and
> belongs on disk. This split is deliberate.

---

## 4. Replit path (build & iterate online)

1. **Create Repl** → *Import from GitHub* (push this folder to a repo first), or
   create a blank Python Repl and drag the files in.
2. Replit reads `replit.nix` and installs Open3D's native libs (`libGL`, `glib`,
   `libstdc++`) and sets `LD_LIBRARY_PATH`. This is the step that makes Open3D
   import on Replit — bare `pip install open3d` is **not** enough.
3. **Secrets** (lock icon) → add `MONGO_URI` (your Atlas string) and `MONGO_DB=scanpipe`.
4. Press **Run**. Replit serves the API; open the webview URL + `/docs` for the
   interactive Swagger UI.
5. Iterate the `classify` rules in `tags.yaml` here against a *small* test cloud
   (use `make_synth.py` to generate one — don't upload GB clouds to Replit).

> If Open3D still fails to import on Replit with a `libGL.so` error, open the Shell
> and run `find /nix/store -name 'libGL.so*' | head` then put that directory in
> `LD_LIBRARY_PATH` in `.replit`. This is the one common snag.

---

## 5. Docker package (the artifact you actually run)

### Build
```bash
docker build -t scanpipe:latest .
```

### Run against Atlas
```bash
docker run -d --name scanpipe -p 8000:8000 \
  -e MONGO_URI="mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority" \
  -e MONGO_DB=scanpipe \
  -v /data/scanpipe:/data \
  scanpipe:latest
```
`-v /data/scanpipe:/data` maps a big local disk to the container so GB clouds and
artifacts live on the workstation, not inside the image.

### Or run fully local (API + Mongo together)
```bash
docker compose up --build -d      # starts mongo + api; data in named volumes
```

Verify:
```bash
curl localhost:8000/health        # {"status":"ok","mongo":"ok"}
```

---

## 6. Using the API (curl, end to end)

```bash
# 1. inspect a cloud first (no DB write) — confirm units/up-axis/splat
curl -F "file=@/data/scans/venue1.ply" localhost:8000/inspect

# 2. create the venue
curl -X POST localhost:8000/venues -H "Content-Type: application/json" -d '{
  "name":"Kala Mandir","city":"Kolkata",
  "location":{"lat":22.5448,"lng":88.3426},
  "capacity":1000,"venue_class":"theatre","stage_dims_m":[12,8,9]}'
# -> note the returned "_id"  (VENUE_ID)

# 3. upload a scan -> it processes and stores
curl -X POST localhost:8000/scans \
  -F "venue_id=VENUE_ID" -F "device=Lixel K2" -F "voxel=0.05" \
  -F "file=@/data/scans/venue1.ply"
# -> returns scan with face_counts; note "_id" (SCAN_ID)

# 4. read it back / download artifacts
curl localhost:8000/scans/SCAN_ID
curl localhost:8000/scans/SCAN_ID/manifest
curl -OJ localhost:8000/scans/SCAN_ID/dxf        # -> SCAN_ID.dxf for SketchUp import

# 5. venue with all its scans
curl localhost:8000/venues/VENUE_ID

# 6. nearby venues (Atlas / local mongo only)
curl "localhost:8000/venues/near/?lat=22.57&lng=88.36&km=25"

# 7. attach a measurement (the paired-data that builds the moat)
curl -X POST localhost:8000/scans/SCAN_ID/measurements -H "Content-Type: application/json" -d '{
  "seat_label":"R12-S8","position":[10,9,1.2],
  "prediction_tool":"MAPP","predicted_spl_db":98.5,"measured_spl_db":96.1,
  "ir_metrics":{"rt60":1.7,"c80":2.3},
  "applied_calibration":{"delays_ms":[0,4.2,8.1]}}'
```

Interactive docs: `http://localhost:8000/docs`.

---

## 7. Close the loop into SketchUp

The API gives you `manifest.json` + `venue.dxf` per scan. To get the tagged `.skp`:
- **Quick:** import the DXF into SketchUp Pro, save `.skp`.
- **Clean tags:** on the SketchUp box, Ruby Console →
  `load "build_skp.rb"; ScanPipe.build_and_save("manifest.json","venue.skp")`,
  then `PATCH`/update the scan's `skp_path` (add that endpoint when you wire it).

From the `.skp`, the five-tool fan-out is one click each (MAPP / ArrayCalc / VS /
Soundvision).

---

## 8. Run the tests

```bash
pip install -r requirements.txt
python make_synth.py                 # makes synth_venue.ply
python tests/smoke_test.py           # boots the API on mongomock, exercises every endpoint
```
Expected tail: `ALL ENDPOINTS PASSED ✔`.

---

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `libGL.so.1: cannot open shared object` | Docker: the apt line in the Dockerfile covers it — rebuild. Replit: add the lib dir to `LD_LIBRARY_PATH` (Section 4). |
| `/health` shows `mongo: unreachable` | wrong `MONGO_URI`, or Atlas IP allow-list missing the workstation IP. |
| `/scans` returns 422 `pipeline failed` | run `/inspect` on that file — usually a splat `.ply`, wrong units, or not Z-up. |
| geo `/venues/near` 400 | you're on mongomock; use Atlas or the compose-local Mongo. |
| huge image / slow build | `.dockerignore` already excludes `data/`, `*.ply`, `.venv`. |

---

## 10. Don't forget the validation gate

Every scan lands as `status: abstracted`. Before you trust a model for design,
attach measurements and compare predicted vs measured (Smaart/SIM3), then promote
to `status: validated`. The `measurements` collection exists precisely so the
abstraction's silent mistakes — a dropped balcony face, a merged stage — get
caught by real data, and so the paired dataset accrues from day one.
