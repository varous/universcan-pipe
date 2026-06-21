"""app/models.py — request models. Responses are serialized dicts (see db.ser)."""
from typing import Optional, List
from pydantic import BaseModel, Field


class GeoPoint(BaseModel):
    lat: float
    lng: float


class RigPoint(BaseModel):
    id: str
    xyz: List[float] = Field(..., min_length=3, max_length=3)
    wll_kg: Optional[float] = None     # working load limit


class VenueIn(BaseModel):
    name: str
    city: str
    location: Optional[GeoPoint] = None
    capacity: Optional[int] = None
    venue_class: Optional[str] = None  # theatre | club | arena | outdoor | ...
    stage_dims_m: Optional[List[float]] = None   # [width, depth, height]
    rigging: Optional[List[RigPoint]] = None
    power: Optional[dict] = None        # {phases, tie_ins:[...]}
    loadin_notes: Optional[str] = None
    sightline_notes: Optional[str] = None


class MeasurementIn(BaseModel):
    """The paired acoustic/operational record — the moat data. One per mic position."""
    position: Optional[List[float]] = None        # [x,y,z] seat/mic location
    seat_label: Optional[str] = None
    source_config: Optional[str] = None           # which array / aim
    prediction_tool: Optional[str] = None         # MAPP | ArrayCalc | VS | Soundvision
    predicted_spl_db: Optional[float] = None
    measured_spl_db: Optional[float] = None
    ir_metrics: Optional[dict] = None             # {rt60, c80, reflections:[{t_ms,level_db}]}
    applied_calibration: Optional[dict] = None    # {delays_ms, gains_db, eq:[...]}
    attribution: Optional[dict] = None            # {modeled_surfaces, rig, mic_pos}
    notes: Optional[str] = None
