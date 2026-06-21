# scanpipe — venue `.ply` → tagged `.skp` starter scaffold

Turns a Green Valley (or any) LiDAR point cloud into a small set of **tagged
planar surfaces** ready to import into MAPP 3D / ArrayCalc / Venue Synthesis /
Soundvision. This is the **starter scaffold**: the inspector and the Python
core run today; the classification heuristics are deliberately simple and are
the thing you harden over time against measured SPL.

## Files
| File | Half | What it does |
|---|---|---|
| `requirements.txt` | — | Python deps (all open-source) |
| `inspect_ply.py` | Python | **Run first.** Reports format, splat-vs-geometry, fields, units, up-axis, density, recommended voxel |
| `tags.yaml` | both | **The contract.** Canonical tag vocabulary + classification rules + pipeline params + downstream layer map |
| `abstract.py` | Python | Core pipeline: load → downsample → segment → cluster → classify → bound → emit `manifest.json` + `venue.dxf` |
| `build_skp.rb` | SketchUp | Reads `manifest.json`, builds faces, assigns Tags natively, saves `.skp` (Path 2) |
| `make_synth.py` | — | Generates a synthetic venue cloud so you can test without real data |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```
Note: `abstract.py` uses Open3D (CPU only here — no GPU needed). The GPU in
your budget is for LixelStudio/splats, not this.

## Run sequence
```bash
# 0. (optional) make a test cloud if you don't have one handy
python make_synth.py

# 1. INSPECT — confirm units / up-axis / fields BEFORE trusting the file
python inspect_ply.py your_scan.ply --json summary.json
#    -> tells you the recommended --voxel and flags splat/units/orientation issues

# 2. ABSTRACT — produce tagged surfaces
python abstract.py your_scan.ply --tags tags.yaml --out out --voxel 0.05
#    -> out/manifest.json  (tagged faces)
#    -> out/venue.dxf      (3DFACEs on layers = tags)

# 3a. SketchUp PATH 1 (start here): File > Import out/venue.dxf, then save .skp.
#     Zero Ruby. Validates geometry immediately while tags are rough.

# 3b. SketchUp PATH 2 (once geometry is trusted): in the Ruby Console:
#       load "build_skp.rb"
#       ScanPipe.build_and_save("out/manifest.json", "out/venue.skp")
#     Clean native Tag assignment — what the fan-out depends on.
```

## The honest part
- Steps 1–2 plumbing is done and tested. The **hard, ongoing** work is the
  `classify` rules in `tags.yaml` + `abstract.py`: rake-vs-floor, balcony-vs-wall,
  stage detection. They will misfire on messy real venues.
- `oriented_quad()` bounds each surface with an oriented rectangle. Irregular
  surfaces need a concave hull (shapely/alphashape) — the upgrade path is noted
  in the code.
- **Always run the validation gate**: build the model, predict SPL, measure
  (Smaart/SIM3), compare, then tune `tags.yaml`. The script's silent mistakes —
  a dropped balcony face, a merged stage — are exactly what measurement catches.

## First real-data task
Run step 1 on one Green Valley scan and read the VERDICT line. If it says
"ready", run step 2 with the recommended voxel and open `venue.dxf` over the
source cloud in CloudCompare to eyeball it. That's your end-to-end proof.
