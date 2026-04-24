"""
StormDataPro MESH Processor — Phase 5 (multi-product MRMS consensus)

Phase 1: MESH decode (done)
Phase 2: Rolling-window accumulator (done)
Phase 3: Polygonization (done)
Phase 4: hailmaps.html wired to /swaths (done)
Phase 5: Multi-product accumulation + consensus scoring for tighter, more-accurate swaths

New products pulled every 2 min alongside MESH:
  MESH                — Maximum Estimated Size of Hail (mm)  [primary]
  POSH                — Probability of Severe Hail (%)        [confidence filter]
  Reflectivity_-20C   — reflectivity at -20C isotherm (dBZ)   [independent hail indicator]
  EchoTop_50          — 50 dBZ echo top height (km MSL)       [storm intensity]
  VIL_Density         — VIL density (g/m³)                    [legacy hail indicator]

Pulled hourly:
  MESHMax1440min      — NOAA's own 24h MESH accumulation      [sanity check]

Consensus scoring (for each pixel, max 100 points):
  +40  MESH ≥ threshold (1" / 1.5" / 2" / 2.75")
  +30  POSH ≥ 50%
  +20  Reflectivity_-20C ≥ 55 dBZ
  +10  EchoTop_50 ≥ 40 kft (12.2 km)
Threshold to draw polygon: score ≥ 50

Memory budget for the 6 persistent grids (total ~343 MB):
  mesh_max_x10   int16  49 MB   max MESH ever seen × 10
  posh_max       int8    25 MB   max POSH %
  refl_max_x10   int16  49 MB   max reflectivity × 10 (range 0-800 = 0-80 dBZ)
  echo_top_x10   int16  49 MB   max 50dBZ echo top × 10 (range 0-200 = 0-20 km)
  vil_dens_x10   int16  49 MB   max VIL density × 10
  last_seen_h    int32  98 MB   hour-since-epoch when pixel last saw hail
                 (total arrays: 319 MB; peak during tick ~440 MB)
"""
import os
import gzip
import tempfile
import logging
import threading
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from scipy import ndimage
from rasterio import features as rio_features
from rasterio.transform import Affine
from shapely.geometry import shape as shp_shape, mapping as shp_mapping

# ── Setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="StormDataPro MESH Processor", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

USER_AGENT = "StormDataPro/0.5 (colton@transcendentpdr.com)"
MRMS_BASE = "https://mrms.ncep.noaa.gov/data/2D"

# ── MRMS product catalog ──
# Each product is fetched from {MRMS_BASE}/{dir}/MRMS_{name}.latest.grib2.gz
# Value sentinels: MESH/POSH use -1 (missing) and -3 (no coverage); reflectivity uses -99/-999.
PRODUCTS = {
    "mesh": {
        "dir": "MESH",
        "name": "MESH",
        "missing_below": 0,          # any value < 0 is missing/sentinel
        "unit": "mm",
    },
    "posh": {
        "dir": "POSH",
        "name": "POSH",
        "missing_below": 0,
        "unit": "%",
    },
    "refl_m20c": {
        "dir": "Reflectivity_-20C",
        "name": "Reflectivity_-20C",
        "missing_below": -90,         # -99 is the missing sentinel; real dBZ can be negative (-5 to -10 in clear air)
        "unit": "dBZ",
    },
    "echo_top": {
        "dir": "EchoTop_50",
        "name": "EchoTop_50",
        "missing_below": 0,           # -1 missing; real heights are positive km MSL
        "unit": "km",
    },
    "vil_density": {
        "dir": "VIL_Density",
        "name": "VIL_Density",
        "missing_below": 0,
        "unit": "g/m3",
    },
}

# Hourly sanity-check product
MESH_1440_PRODUCT = {"dir": "MESHMax1440min", "name": "MESHMax1440min", "missing_below": 0, "unit": "mm"}

# MRMS CONUS grid at 0.01° resolution
GRID_ROWS = 3500
GRID_COLS = 7000
GRID_LAT1 = 54.995
GRID_LON1_360 = 230.005
GRID_DEG = 0.01

# Scale factor for int16 products (stores value × 10)
SCALE = 10

# Rolling window
ROLLING_WINDOW_HOURS = 24
MIN_HAIL_STORED_MM = 5.0
MIN_HAIL_STORED = int(MIN_HAIL_STORED_MM * SCALE)

# Persistence
STATE_DIR = os.environ.get("STATE_DIR", "/data" if os.path.isdir("/data") else "/tmp/mesh-state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "accumulator_v5.npz")
POLYGONS_CACHE_FILE = os.path.join(STATE_DIR, "polygons.json")


# ── Multi-product Accumulator ──
class Accumulator:
    """Holds per-pixel max of each MRMS product over the rolling window."""

    def __init__(self):
        self.lock = threading.Lock()
        # All persistent grids
        self.mesh_max_x10 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16)
        self.posh_max = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)
        self.refl_max_x10 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16)
        self.echo_top_x10 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16)
        self.vil_dens_x10 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16)
        self.last_seen_h = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        # Bookkeeping
        self.update_count = 0
        self.last_update_ts: Optional[str] = None
        self.last_tick_products: dict = {}   # which products successfully fetched this tick
        self.last_tick_maxes: dict = {}       # per-product max of the incoming grid
        self.last_error: Optional[str] = None
        # MESH 1440min sanity check
        self.mesh_1440min_max_inches: Optional[float] = None
        self.mesh_1440min_last_check: Optional[str] = None
        self.load()

    def load(self):
        if os.path.exists(STATE_FILE):
            try:
                data = np.load(STATE_FILE)
                self.mesh_max_x10 = data['mesh_max_x10']
                self.posh_max = data['posh_max']
                self.refl_max_x10 = data['refl_max_x10']
                self.echo_top_x10 = data['echo_top_x10']
                self.vil_dens_x10 = data['vil_dens_x10']
                self.last_seen_h = data['last_seen_h']
                self.update_count = int(data['update_count'][0])
                log.info(f"Loaded accumulator v5 state (updates={self.update_count})")
            except Exception as e:
                log.warning(f"Could not load v5 state, starting fresh: {e}")

    def save(self):
        tmp = STATE_FILE + ".tmp"
        try:
            np.savez_compressed(
                tmp,
                mesh_max_x10=self.mesh_max_x10,
                posh_max=self.posh_max,
                refl_max_x10=self.refl_max_x10,
                echo_top_x10=self.echo_top_x10,
                vil_dens_x10=self.vil_dens_x10,
                last_seen_h=self.last_seen_h,
                update_count=np.array([self.update_count]),
            )
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            log.error(f"Save failed: {e}")

    def apply_mesh_only(self, mesh_vals: np.ndarray, ts_unix: int):
        """Legacy helper kept for reference. Main tick loop inlines this for serial processing."""
        with self.lock:
            hail_mask = mesh_vals >= 0.5
            n_hail = int(hail_mask.sum())
            if n_hail > 0:
                rows, cols = np.where(hail_mask)
                hail_mesh_x10 = np.clip(mesh_vals[hail_mask] * SCALE, 0, 32000).astype(np.int16)
                existing = self.mesh_max_x10[rows, cols]
                bigger = hail_mesh_x10 > existing
                self.mesh_max_x10[rows[bigger], cols[bigger]] = hail_mesh_x10[bigger]
                active = hail_mesh_x10 >= MIN_HAIL_STORED
                self.last_seen_h[rows[active], cols[active]] = np.int32(ts_unix // 3600)
            self.update_count += 1
            self.last_update_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            return n_hail

    def prune_old(self, now_unix: int):
        """DEPRECATED. Previously used for rolling-window pruning. Replaced by reset_all()
        (called at 00:00 UTC daily). Kept for reference; no longer scheduled.
        """
        cutoff_h = np.int32((now_unix - (ROLLING_WINDOW_HOURS * 3600)) // 3600)
        with self.lock:
            stale = (self.last_seen_h > 0) & (self.last_seen_h < cutoff_h)
            n_stale = int(stale.sum())
            if n_stale > 0:
                self.mesh_max_x10[stale] = 0
                self.posh_max[stale] = 0
                self.refl_max_x10[stale] = 0
                self.echo_top_x10[stale] = 0
                self.vil_dens_x10[stale] = 0
                self.last_seen_h[stale] = 0
                log.info(f"Pruned {n_stale} stale pixels")
            return n_stale

    def reset_all(self):
        """Daily reset — wipe the entire accumulator so each UTC day starts fresh.
        
        Called by scheduler at 00:00 UTC every day. This is different from prune_old,
        which only clears pixels older than the rolling window. reset_all wipes
        everything so the map shows only today's hail accumulation, not the
        trailing 24 hours from yesterday.
        """
        with self.lock:
            n_nonzero = int((self.mesh_max_x10 > 0).sum())
            self.mesh_max_x10.fill(0)
            self.posh_max.fill(0)
            self.refl_max_x10.fill(0)
            self.echo_top_x10.fill(0)
            self.vil_dens_x10.fill(0)
            self.last_seen_h.fill(0)
            log.info(f"Daily reset at 00:00 UTC — cleared {n_nonzero} accumulated hail pixels for new storm day")
            return n_nonzero

    def snapshot_stats(self) -> dict:
        with self.lock:
            hail_1in = self.mesh_max_x10 >= int(25.4 * SCALE)
            return {
                "update_count": self.update_count,
                "last_update": self.last_update_ts,
                "last_tick_products": self.last_tick_products,
                "last_tick_maxes": self.last_tick_maxes,
                "accumulated_hail_pixels_1in": int(hail_1in.sum()),
                "accumulated_max_mm": round(float(self.mesh_max_x10.max()) / SCALE, 2),
                "accumulated_max_inches": round(float(self.mesh_max_x10.max()) / SCALE / 25.4, 2),
                "accumulated_max_posh": int(self.posh_max.max()),
                "accumulated_max_refl_dbz": round(float(self.refl_max_x10.max()) / SCALE, 1),
                "accumulated_max_echo_top_km": round(float(self.echo_top_x10.max()) / SCALE, 1),
                "mesh_1440min_max_inches": self.mesh_1440min_max_inches,
                "mesh_1440min_last_check": self.mesh_1440min_last_check,
                "state_file": STATE_FILE,
                "last_error": self.last_error,
            }


accumulator = Accumulator()


# ── Polygonizer with consensus scoring ──
# Size thresholds — IHM/HailTrace style
# DEPRECATED (kept for reference only — replaced by POLYGON_BANDS below).
# The threshold-based approach produced nested/overlapping polygons; bands produce
# non-overlapping polygons so clicks at different points reveal different sizes.
POLYGON_THRESHOLDS = [
    {"min_mm": 25.4, "inches": 1.0,  "desc": "1.0\" (quarter)",   "color": "#eab308"},
    {"min_mm": 38.1, "inches": 1.5,  "desc": "1.5\" (walnut)",    "color": "#f97316"},
    {"min_mm": 50.8, "inches": 2.0,  "desc": "2.0\" (golf ball)", "color": "#ef4444"},
    {"min_mm": 69.85,"inches": 2.75, "desc": "2.75\"+ (baseball)","color": "#a855f7"},
]

# Size bands (non-overlapping). Each band is [min_mm, max_mm_exclusive).
# These replace the concentric nested polygons - each band is a ring that's only
# drawn where the MESH value falls within that specific range. Click any part
# of the swath and see exactly what size hail fell there, not just the peak.
POLYGON_BANDS = [
    {"min_mm": 25.4,  "max_mm": 38.1,  "min_inches": 1.0,  "max_inches": 1.5,  "desc": "1.0\"–1.5\" (Quarter–Ping Pong)",   "color": "#eab308"},
    {"min_mm": 38.1,  "max_mm": 50.8,  "min_inches": 1.5,  "max_inches": 2.0,  "desc": "1.5\"–2.0\" (Ping Pong–Hen Egg)",   "color": "#f97316"},
    {"min_mm": 50.8,  "max_mm": 69.85, "min_inches": 2.0,  "max_inches": 2.75, "desc": "2.0\"–2.75\" (Hen Egg–Baseball)",   "color": "#ef4444"},
    {"min_mm": 69.85, "max_mm": None,  "min_inches": 2.75, "max_inches": None, "desc": "2.75\"+ (Baseball & larger)",       "color": "#a855f7"},
]

# Consensus scoring thresholds
CONSENSUS_POSH_MIN = 50           # POSH ≥ 50% contributes
CONSENSUS_REFL_DBZ_MIN = 55        # Reflectivity@-20C ≥ 55 dBZ contributes
CONSENSUS_ECHO_TOP_KM_MIN = 12.2   # ~40 kft; Echo Top ≥ this contributes
CONSENSUS_MIN_SCORE = 50           # pixel must score >= this to draw polygon

GRID_TRANSFORM = Affine(
    GRID_DEG, 0.0, GRID_LON1_360 - 360,
    0.0, -GRID_DEG, GRID_LAT1
)


class Polygonizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_run_ts: Optional[str] = None
        self.last_run_elapsed_ms: int = 0
        self.last_error: Optional[str] = None
        self.cached_geojson: dict = {
            "type": "FeatureCollection",
            "metadata": {"note": "polygons not yet generated"},
            "features": [],
        }
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(POLYGONS_CACHE_FILE):
            try:
                with open(POLYGONS_CACHE_FILE, "r") as f:
                    self.cached_geojson = json.load(f)
                n = len(self.cached_geojson.get("features", []))
                log.info(f"Loaded {n} cached polygons")
            except Exception as e:
                log.warning(f"Polygon cache load failed: {e}")

    def _save_cache(self):
        tmp = POLYGONS_CACHE_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.cached_geojson, f, separators=(",", ":"))
            os.replace(tmp, POLYGONS_CACHE_FILE)
        except Exception as e:
            log.error(f"Polygon cache save failed: {e}")

    def run(self):
        """Rebuild polygons using multi-product consensus scoring.

        Memory strategy: we NEVER snapshot full 25M-pixel grids. Instead, we find
        the bounding box of all hail pixels, then copy only the crop for polygonization.
        The accumulator lock is held briefly for each crop copy.
        """
        t0 = datetime.now(timezone.utc)
        try:
            # Step 1: find the bbox of ALL hail pixels (at any band) using a cheap read
            # We use the lowest band's min since any polygon must have MESH >= that somewhere
            min_thresh_x10 = int(POLYGON_BANDS[0]["min_mm"] * SCALE)
            with accumulator.lock:
                rows_any = np.any(accumulator.mesh_max_x10 >= min_thresh_x10, axis=1)
                cols_any = np.any(accumulator.mesh_max_x10 >= min_thresh_x10, axis=0)
                update_count = accumulator.update_count
                last_update_ts = accumulator.last_update_ts

            if not rows_any.any():
                # No hail anywhere at any threshold — emit empty polygon set
                new_geojson = {
                    "type": "FeatureCollection",
                    "metadata": {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_ms": int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
                        "polygon_count": 0,
                        "accumulator_update_count": update_count,
                        "accumulator_last_update": last_update_ts,
                        "reset_schedule": "00:00 UTC daily",
                        "bands": [
                            {"min_inches": b["min_inches"], "max_inches": b["max_inches"], "desc": b["desc"]}
                            for b in POLYGON_BANDS
                        ],
                        "note": "No hail at or above 1\" anywhere since last 00:00 UTC reset",
                    },
                    "features": [],
                }
                with self.lock:
                    self.cached_geojson = new_geojson
                    self.last_run_ts = new_geojson["metadata"]["generated_at"]
                    self.last_error = None
                self._save_cache()
                rows_any = cols_any = None
                log.info("Polygonization: 0 polygons (no qualifying hail in window)")
                return

            gr0 = int(np.argmax(rows_any))
            gr1 = GRID_ROWS - int(np.argmax(rows_any[::-1]))
            gc0 = int(np.argmax(cols_any))
            gc1 = GRID_COLS - int(np.argmax(cols_any[::-1]))
            # Pad 5 pixels for morphology edge effects at polygon edges
            gr0 = max(0, gr0 - 5)
            gr1 = min(GRID_ROWS, gr1 + 5)
            gc0 = max(0, gc0 - 5)
            gc1 = min(GRID_COLS, gc1 + 5)
            rows_any = cols_any = None

            # Step 2: copy only the bbox from each accumulator grid
            with accumulator.lock:
                crop_mesh = accumulator.mesh_max_x10[gr0:gr1, gc0:gc1].copy()
                crop_posh = accumulator.posh_max[gr0:gr1, gc0:gc1].copy()
                crop_refl = accumulator.refl_max_x10[gr0:gr1, gc0:gc1].copy()
                crop_echo = accumulator.echo_top_x10[gr0:gr1, gc0:gc1].copy()
                crop_last_seen = accumulator.last_seen_h[gr0:gr1, gc0:gc1].copy()

            all_features = []
            refl_dbz_x10_thresh = int(CONSENSUS_REFL_DBZ_MIN * SCALE)
            echo_km_x10_thresh = int(CONSENSUS_ECHO_TOP_KM_MIN * SCALE)

            # Produce NON-OVERLAPPING bands: each polygon is only where MESH
            # falls within that band's range (e.g. 1.0-1.5" band excludes pixels
            # that are 1.5"+). Click any band to see the size range for that spot.
            for band in POLYGON_BANDS:
                band_min_x10 = int(band["min_mm"] * SCALE)
                band_max_x10 = int(band["max_mm"] * SCALE) if band["max_mm"] is not None else None

                if band_max_x10 is None:
                    crop_mesh_meets = crop_mesh >= band_min_x10
                else:
                    crop_mesh_meets = (crop_mesh >= band_min_x10) & (crop_mesh < band_max_x10)

                if not crop_mesh_meets.any():
                    continue

                # Consensus score on the crop
                crop_score = np.full(crop_mesh_meets.shape, 40, dtype=np.uint8)
                crop_score[~crop_mesh_meets] = 0
                crop_score += ((crop_posh >= CONSENSUS_POSH_MIN) & crop_mesh_meets).astype(np.uint8) * 30
                crop_score += ((crop_refl >= refl_dbz_x10_thresh) & crop_mesh_meets).astype(np.uint8) * 20
                crop_score += ((crop_echo >= echo_km_x10_thresh) & crop_mesh_meets).astype(np.uint8) * 10

                pass_mask = crop_score >= CONSENSUS_MIN_SCORE
                crop_mesh_meets = crop_score = None
                gc.collect()

                if not pass_mask.any():
                    pass_mask = None
                    continue

                # Morphology on the crop (fast — crop is small)
                cleaned = ndimage.binary_opening(pass_mask, iterations=1)
                cleaned = ndimage.binary_closing(cleaned, iterations=2)
                pass_mask = None
                if not cleaned.any():
                    cleaned = None
                    gc.collect()
                    continue

                # Polygonize with a crop-local Affine transform
                crop_transform = Affine(
                    GRID_DEG, 0, (GRID_LON1_360 - 360) + gc0 * GRID_DEG,
                    0, -GRID_DEG, GRID_LAT1 - gr0 * GRID_DEG
                )
                uint8_mask = cleaned.astype(np.uint8)
                raw_polys = []
                for geom, val in rio_features.shapes(uint8_mask, mask=cleaned, transform=crop_transform):
                    if val == 1:
                        raw_polys.append(shp_shape(geom))

                simplified = []
                for p in raw_polys:
                    sp = p.simplify(0.015, preserve_topology=True)
                    if sp.area >= 0.0001 and not sp.is_empty:
                        simplified.append(sp)

                # Attribute each polygon from the cropped arrays
                for poly in simplified:
                    minx, miny, maxx, maxy = poly.bounds
                    # Convert back to crop-local indices
                    sub_c0 = max(0, int((minx - (GRID_LON1_360 - 360) - gc0 * GRID_DEG) / GRID_DEG))
                    sub_c1 = min(crop_mesh.shape[1], int((maxx - (GRID_LON1_360 - 360) - gc0 * GRID_DEG) / GRID_DEG) + 1)
                    sub_r1 = max(0, int((GRID_LAT1 - gr0 * GRID_DEG - maxy) / GRID_DEG))
                    sub_r0 = min(crop_mesh.shape[0], int((GRID_LAT1 - gr0 * GRID_DEG - miny) / GRID_DEG) + 1)

                    peak_x10 = band_min_x10
                    peak_posh = 0
                    peak_refl_x10 = 0
                    peak_echo_x10 = 0
                    peak_score = CONSENSUS_MIN_SCORE
                    last_unix = first_unix = 0
                    duration_min = 0

                    if sub_c1 > sub_c0 and sub_r0 > sub_r1:
                        sub_mesh = crop_mesh[sub_r1:sub_r0, sub_c0:sub_c1]
                        if band_max_x10 is None:
                            sub_mask = sub_mesh >= band_min_x10
                        else:
                            sub_mask = (sub_mesh >= band_min_x10) & (sub_mesh < band_max_x10)
                        if sub_mask.any():
                            peak_x10 = int(sub_mesh[sub_mask].max())
                            sub_posh = crop_posh[sub_r1:sub_r0, sub_c0:sub_c1]
                            sub_refl = crop_refl[sub_r1:sub_r0, sub_c0:sub_c1]
                            sub_echo = crop_echo[sub_r1:sub_r0, sub_c0:sub_c1]
                            peak_posh = int(sub_posh[sub_mask].max())
                            peak_refl_x10 = int(sub_refl[sub_mask].max())
                            peak_echo_x10 = int(sub_echo[sub_mask].max())
                            sc = 40
                            if peak_posh >= CONSENSUS_POSH_MIN: sc += 30
                            if peak_refl_x10 >= refl_dbz_x10_thresh: sc += 20
                            if peak_echo_x10 >= echo_km_x10_thresh: sc += 10
                            peak_score = sc

                            sub_last = crop_last_seen[sub_r1:sub_r0, sub_c0:sub_c1]
                            sub_last_nz = sub_last[sub_last > 0]
                            if len(sub_last_nz) > 0:
                                last_unix = int(sub_last_nz.max()) * 3600
                                first_unix = int(sub_last_nz.min()) * 3600
                                duration_min = round((last_unix - first_unix) / 60.0, 0)

                    area_sqmi = round(poly.area * 3800, 1)
                    geom_json = shp_mapping(poly)

                    feature = {
                        "type": "Feature",
                        "geometry": geom_json,
                        "properties": {
                            # Band info (new)
                            "bandMinInches": band["min_inches"],
                            "bandMaxInches": band["max_inches"],  # null for top band
                            "bandLabel": band["desc"],
                            # Kept for frontend backward compatibility
                            "thresholdInches": band["min_inches"],
                            "thresholdLabel": band["desc"],
                            "color": band["color"],
                            # Peak within this band (will be in [bandMin, bandMax))
                            "peakSizeMM": round(peak_x10 / SCALE, 1),
                            "peakSizeInches": round(peak_x10 / SCALE / 25.4, 2),
                            "peakPOSH": peak_posh,
                            "peakReflectivityM20C_dBZ": round(peak_refl_x10 / SCALE, 1),
                            "peakEchoTop_km": round(peak_echo_x10 / SCALE, 1),
                            "peakEchoTop_kft": round(peak_echo_x10 / SCALE * 3.281, 1),
                            "consensusScore": peak_score,
                            "areaSquareMiles": area_sqmi,
                            "firstSeen": datetime.fromtimestamp(first_unix, tz=timezone.utc).isoformat() if first_unix > 0 else None,
                            "lastSeen": datetime.fromtimestamp(last_unix, tz=timezone.utc).isoformat() if last_unix > 0 else None,
                            "durationMinutes": duration_min,
                        },
                    }
                    all_features.append(feature)

                cleaned = uint8_mask = raw_polys = simplified = None
                gc.collect()

            # Free the crops
            crop_mesh = crop_posh = crop_refl = crop_echo = crop_last_seen = None
            gc.collect()

            all_features.sort(key=lambda f: f["properties"]["bandMinInches"])
            elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

            new_geojson = {
                "type": "FeatureCollection",
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "polygon_count": len(all_features),
                    "accumulator_update_count": update_count,
                    "accumulator_last_update": last_update_ts,
                    "reset_schedule": "00:00 UTC daily",
                    "bands": [
                        {"min_inches": b["min_inches"], "max_inches": b["max_inches"], "desc": b["desc"]}
                        for b in POLYGON_BANDS
                    ],
                    "consensus_min_score": CONSENSUS_MIN_SCORE,
                    "scoring": {
                        "mesh_threshold_points": 40,
                        "posh_50_points": 30,
                        "reflectivity_m20c_55dbz_points": 20,
                        "echo_top_40kft_points": 10,
                    },
                    "bbox_used": [gr0, gr1, gc0, gc1],
                },
                "features": all_features,
            }

            with self.lock:
                self.cached_geojson = new_geojson
                self.last_run_ts = new_geojson["metadata"]["generated_at"]
                self.last_run_elapsed_ms = elapsed_ms
                self.last_error = None

            self._save_cache()
            log.info(f"Polygonization: {len(all_features)} polygons in {elapsed_ms}ms, bbox={gr1-gr0}x{gc1-gc0}")

        except Exception as e:
            log.error(f"Polygonization failed: {e}", exc_info=True)
            with self.lock:
                self.last_error = str(e)

    def get_geojson(self) -> dict:
        with self.lock:
            return self.cached_geojson

    def stats(self) -> dict:
        with self.lock:
            return {
                "last_run": self.last_run_ts,
                "last_run_elapsed_ms": self.last_run_elapsed_ms,
                "polygon_count": len(self.cached_geojson.get("features", [])),
                "last_error": self.last_error,
                "cache_file": POLYGONS_CACHE_FILE,
                "consensus_min_score": CONSENSUS_MIN_SCORE,
            }


polygonizer = Polygonizer()


# ── MRMS fetch/decode (generic for any product) ──
def fetch_product(prod_key: str, prod_def: dict) -> Optional[dict]:
    """Fetch and decode one MRMS product. Returns {values: 2D array, timestamp: datetime} or None."""
    url = f"{MRMS_BASE}/{prod_def['dir']}/MRMS_{prod_def['name']}.latest.grib2.gz"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45, allow_redirects=True)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
    except Exception as e:
        log.warning(f"[{prod_key}] fetch failed: {e}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        if not messages:
            return None
        grb = messages[0]
        values = np.asarray(grb.values, dtype=np.float32)
        # Zero out sentinels (missing/no-coverage) using product's missing_below threshold
        values = np.where(values < prod_def["missing_below"], 0.0, values).astype(np.float32)
        try:
            ts = grb.validDate.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        grbs.close()
        return {"values": values, "timestamp": ts}
    except Exception as e:
        log.warning(f"[{prod_key}] decode failed: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def fetch_with_coords(prod_def: dict) -> Optional[dict]:
    """Like fetch_product but also returns lat/lon arrays. Used only by /test endpoints."""
    url = f"{MRMS_BASE}/{prod_def['dir']}/MRMS_{prod_def['name']}.latest.grib2.gz"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45, allow_redirects=True)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
    except Exception as e:
        log.warning(f"fetch_with_coords failed: {e}")
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        if not messages:
            return None
        grb = messages[0]
        values = np.asarray(grb.values, dtype=np.float32)
        lats, lons = grb.latlons()
        values = np.where(values < prod_def["missing_below"], 0.0, values)
        try:
            ts = grb.validDate.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        grbs.close()
        return {"values": values, "lats": lats, "lons": lons, "timestamp": ts}
    except Exception as e:
        log.warning(f"decode with coords failed: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def tick():
    """Fetch all products SERIALLY and merge into accumulator one at a time.

    Serial is mandatory, not a choice — each 25M-pixel float32 product array is ~94MB.
    Fetching 5 in parallel would briefly hold ~470MB of incoming data on top of the
    319MB persistent state, blowing past Railway's 512MB container limit. Serial
    processing keeps peak memory around 250MB at the cost of ~5 extra seconds of
    wall time per tick (well under our 2-minute interval).
    """
    t0 = time.time()
    try:
        # MESH first — it's the primary product and drives everything else
        mesh_result = fetch_product("mesh", PRODUCTS["mesh"])
        if not mesh_result:
            accumulator.last_error = "MESH unavailable"
            log.warning("Tick skipped — MESH fetch failed")
            return
        ts_unix = int(mesh_result["timestamp"].timestamp())
        mesh_vals = mesh_result["values"]
        mesh_result = None  # drop the outer dict wrapper

        # Find the pixel mask from MESH — we only care about these pixels for all products
        hail_mask = mesh_vals >= 0.5
        n_hail = int(hail_mask.sum())

        fetch_summary = {"mesh": "ok"}
        max_summary = {"mesh": round(float(mesh_vals.max()), 2)}

        with accumulator.lock:
            if n_hail > 0:
                rows, cols = np.where(hail_mask)
                hail_mesh_x10 = np.clip(mesh_vals[hail_mask] * SCALE, 0, 32000).astype(np.int16)

                # MESH accumulation
                existing = accumulator.mesh_max_x10[rows, cols]
                bigger = hail_mesh_x10 > existing
                accumulator.mesh_max_x10[rows[bigger], cols[bigger]] = hail_mesh_x10[bigger]

                # last_seen
                active = hail_mesh_x10 >= MIN_HAIL_STORED
                accumulator.last_seen_h[rows[active], cols[active]] = np.int32(ts_unix // 3600)
            else:
                rows = cols = None

            accumulator.update_count += 1
            accumulator.last_update_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()

        # Free the MESH values array before fetching next product
        mesh_vals = None
        hail_mask = None
        gc.collect()

        # If no hail, skip secondary products (no pixels to update anyway)
        if n_hail == 0:
            accumulator.last_tick_products = fetch_summary
            accumulator.last_tick_maxes = max_summary
            accumulator.last_error = None
            elapsed_s = time.time() - t0
            log.info(f"Tick #{accumulator.update_count}: no hail — skipped secondaries ({elapsed_s:.1f}s)")
            return

        # Now fetch each secondary product serially, using the MESH-derived rows/cols
        for prod_key in ("posh", "refl_m20c", "echo_top", "vil_density"):
            prod_def = PRODUCTS[prod_key]
            result = fetch_product(prod_key, prod_def)
            if not result:
                fetch_summary[prod_key] = "fail"
                max_summary[prod_key] = None
                continue

            fetch_summary[prod_key] = "ok"
            vals = result["values"]
            max_summary[prod_key] = round(float(vals.max()), 2)
            # Sample values only at hail pixels (sparse — small array)
            vals_at_hail = vals[rows, cols]
            # Free the full-grid array immediately
            result = None
            vals = None
            gc.collect()

            with accumulator.lock:
                if prod_key == "posh":
                    new_v = np.clip(vals_at_hail, 0, 127).astype(np.int8)
                    existing = accumulator.posh_max[rows, cols]
                    bigger = new_v > existing
                    accumulator.posh_max[rows[bigger], cols[bigger]] = new_v[bigger]
                elif prod_key == "refl_m20c":
                    new_v = np.clip(vals_at_hail * SCALE, 0, 32000).astype(np.int16)
                    existing = accumulator.refl_max_x10[rows, cols]
                    bigger = new_v > existing
                    accumulator.refl_max_x10[rows[bigger], cols[bigger]] = new_v[bigger]
                elif prod_key == "echo_top":
                    new_v = np.clip(vals_at_hail * SCALE, 0, 32000).astype(np.int16)
                    existing = accumulator.echo_top_x10[rows, cols]
                    bigger = new_v > existing
                    accumulator.echo_top_x10[rows[bigger], cols[bigger]] = new_v[bigger]
                elif prod_key == "vil_density":
                    new_v = np.clip(vals_at_hail * SCALE, 0, 32000).astype(np.int16)
                    existing = accumulator.vil_dens_x10[rows, cols]
                    bigger = new_v > existing
                    accumulator.vil_dens_x10[rows[bigger], cols[bigger]] = new_v[bigger]

            vals_at_hail = new_v = None
            gc.collect()

        accumulator.last_tick_products = fetch_summary
        accumulator.last_tick_maxes = max_summary
        accumulator.last_error = None
        elapsed_s = time.time() - t0
        prod_status = ",".join(f"{k}:{'y' if fetch_summary.get(k)=='ok' else 'n'}" for k in ("mesh","posh","refl_m20c","echo_top","vil_density"))
        log.info(
            f"Tick #{accumulator.update_count}: "
            f"pixels={n_hail}, mesh_max={max_summary.get('mesh')}mm, "
            f"posh_max={max_summary.get('posh')}, "
            f"refl_max={max_summary.get('refl_m20c')}dBZ, "
            f"echo_top_max={max_summary.get('echo_top')}km, "
            f"products={prod_status}, "
            f"{elapsed_s:.1f}s"
        )
    except Exception as e:
        accumulator.last_error = str(e)
        log.error(f"Tick failed: {e}", exc_info=True)


def sanity_check_mesh_1440():
    """Hourly: fetch NOAA's own 24h MESH accumulation and compare to ours."""
    try:
        result = fetch_product("mesh_1440min", MESH_1440_PRODUCT)
        if not result:
            log.warning("MESH 1440min fetch failed")
            return
        vals = result["values"]
        max_mm = float(vals.max())
        max_inches = round(max_mm / 25.4, 2)
        with accumulator.lock:
            accumulator.mesh_1440min_max_inches = max_inches
            accumulator.mesh_1440min_last_check = datetime.now(timezone.utc).isoformat()
            our_max_mm = float(accumulator.mesh_max_x10.max()) / SCALE
        diff = max_mm - our_max_mm
        log.info(f"MESH 1440min sanity: NOAA max={max_mm:.1f}mm ({max_inches}\"), ours={our_max_mm:.1f}mm, diff={diff:+.1f}mm")
        # Free the 94MB array
        result = None
        vals = None
        gc.collect()
    except Exception as e:
        log.error(f"MESH 1440 sanity check failed: {e}", exc_info=True)


def periodic_save():
    accumulator.save()


def midnight_reset():
    """Runs at 00:00 UTC every day. Wipes accumulator so the new storm day starts fresh."""
    n_cleared = accumulator.reset_all()
    # Immediately rebuild polygons so the map shows empty state right after reset
    polygonizer.run()
    log.info(f"midnight_reset complete: cleared {n_cleared} pixels, polygonizer re-ran")


def periodic_polygonize():
    polygonizer.run()


# ── Scheduler ──
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    tick, "interval", minutes=2, id="mesh_tick",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
    max_instances=1, coalesce=True,
)
scheduler.add_job(periodic_save, "interval", minutes=10, id="save_state",
                  max_instances=1, coalesce=True)
# Daily reset at midnight UTC — each storm day starts with an empty map
scheduler.add_job(midnight_reset, "cron", hour=0, minute=0, id="midnight_reset",
                  max_instances=1, coalesce=True)
scheduler.add_job(
    periodic_polygonize, "interval", minutes=5, id="polygonize",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=120),
    max_instances=1, coalesce=True,
)
scheduler.add_job(
    sanity_check_mesh_1440, "interval", minutes=60, id="mesh_1440_sanity",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=180),
    max_instances=1, coalesce=True,
)


@app.on_event("startup")
def startup():
    scheduler.start()
    log.info("Phase 5.1 scheduler: tick 2min (5 products), polygonize 5min, midnight reset 00:00 UTC, 1440 sanity hourly")


@app.on_event("shutdown")
def shutdown():
    try:
        accumulator.save()
    except Exception:
        pass
    scheduler.shutdown(wait=False)


# ── Helpers ──
def pixel_to_lonlat(ridx: int, cidx: int) -> tuple:
    lat = GRID_LAT1 - ridx * GRID_DEG
    lon_raw = GRID_LON1_360 + cidx * GRID_DEG
    lon = lon_raw - 360 if lon_raw > 180 else lon_raw
    return lon, lat


def _values_to_points(values, lats, lons, min_mm: float, max_points: int = 500):
    mask = values >= min_mm
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return []
    sizes = values[mask]
    order = np.argsort(-sizes)[:max_points]
    features = []
    for i in order:
        ridx, cidx = indices[i]
        mm = float(values[ridx, cidx])
        lon = float(lons[ridx, cidx])
        if lon > 180:
            lon -= 360
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, float(lats[ridx, cidx])]},
            "properties": {"sizeMM": round(mm, 1), "sizeInches": round(mm / 25.4, 2)},
        })
    return features


# ── Endpoints ──
@app.get("/")
def root():
    return {
        "service": "StormDataPro MESH Processor",
        "version": "0.5.0",
        "phase": "5 — multi-product MRMS consensus",
        "products_fetched": list(PRODUCTS.keys()),
        "consensus_scoring": {
            "mesh_threshold_points": 40,
            "posh_50_points": 30,
            "reflectivity_m20c_55dbz_points": 20,
            "echo_top_40kft_points": 10,
            "min_score_to_draw_polygon": CONSENSUS_MIN_SCORE,
        },
        "endpoints": [
            "GET /health",
            "GET /status",
            "GET /test — latest MESH fetch summary",
            "GET /test-products — latest values for all 5 products",
            "GET /test-points — current hail >=1in (MESH only)",
            "GET /test-all — all detected hail (MESH only)",
            "GET /accumulated?min_inches=1.0",
            "GET /accumulated/stats",
            "GET /swaths — consensus-filtered polygons (the main output)",
            "GET /swaths/stats",
            "POST /admin/force-tick",
            "POST /admin/polygonize",
            "POST /admin/sanity-check",
            "POST /admin/reset",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
def status():
    return {
        "service": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "scheduler_running": scheduler.running,
        "jobs": [
            {"id": j.id, "next_run": str(j.next_run_time) if j.next_run_time else None}
            for j in scheduler.get_jobs()
        ],
        "accumulator": accumulator.snapshot_stats(),
        "polygonizer": polygonizer.stats(),
    }


@app.get("/test")
def test_decode():
    result = fetch_product("mesh", PRODUCTS["mesh"])
    if not result:
        raise HTTPException(500, "MESH fetch failed")
    values = result["values"]
    hail_count = int(np.sum(values >= 25.4))
    max_val = float(values.max())
    return {
        "status": "success",
        "timestamp": result["timestamp"].isoformat(),
        "grid_shape": list(values.shape),
        "hail_pixels_1in_plus": hail_count,
        "max_hail_mm": round(max_val, 2),
        "max_hail_inches": round(max_val / 25.4, 2),
    }


@app.get("/test-products")
def test_products():
    """Fetch all 5 products in parallel and return their current max values — useful for verification."""
    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_product, k, v): k for k, v in PRODUCTS.items()}
        for fut in as_completed(futures, timeout=90):
            key = futures[fut]
            try:
                r = fut.result()
                if r:
                    results[key] = {
                        "status": "ok",
                        "timestamp": r["timestamp"].isoformat(),
                        "unit": PRODUCTS[key]["unit"],
                        "max_value": round(float(r["values"].max()), 2),
                        "nonzero_pixels": int((r["values"] > 0).sum()),
                    }
                else:
                    results[key] = {"status": "fail"}
            except Exception as e:
                results[key] = {"status": "error", "error": str(e)}
    return {"products": results, "fetched_at": datetime.now(timezone.utc).isoformat()}


@app.get("/test-points")
def test_points():
    result = fetch_with_coords(PRODUCTS["mesh"])
    if not result:
        raise HTTPException(500, "fetch failed")
    features = _values_to_points(result["values"], result["lats"], result["lons"], min_mm=25.4, max_points=500)
    return {
        "type": "FeatureCollection",
        "metadata": {
            "timestamp": result["timestamp"].isoformat(),
            "threshold": ">=1 inch",
            "count": len(features),
        },
        "features": features,
    }


@app.get("/test-all")
def test_all():
    result = fetch_with_coords(PRODUCTS["mesh"])
    if not result:
        raise HTTPException(500, "fetch failed")
    features = _values_to_points(result["values"], result["lats"], result["lons"], min_mm=5.0, max_points=2000)
    return {
        "type": "FeatureCollection",
        "metadata": {
            "timestamp": result["timestamp"].isoformat(),
            "threshold": ">=0.2 inch",
            "count": len(features),
        },
        "features": features,
    }


@app.get("/accumulated")
def accumulated(min_inches: float = 1.0, max_points: int = 2000):
    """Accumulated MESH-max points (not consensus-filtered — raw pixel data)."""
    min_mm_x10 = int(min_inches * 25.4 * SCALE)
    with accumulator.lock:
        mask = accumulator.mesh_max_x10 >= min_mm_x10
        n_above = int(mask.sum())
        if n_above == 0:
            return {
                "type": "FeatureCollection",
                "metadata": {
                    "reset_schedule": "00:00 UTC daily",
                    "min_inches": min_inches,
                    "count": 0,
                    "update_count": accumulator.update_count,
                    "last_update": accumulator.last_update_ts,
                },
                "features": [],
            }
        indices = np.argwhere(mask)
        sizes_x10 = accumulator.mesh_max_x10[mask]
        order = np.argsort(-sizes_x10)[:max_points]
        features = []
        for i in order:
            ridx, cidx = indices[i]
            mm = float(accumulator.mesh_max_x10[ridx, cidx]) / SCALE
            posh = int(accumulator.posh_max[ridx, cidx])
            refl = float(accumulator.refl_max_x10[ridx, cidx]) / SCALE
            echo = float(accumulator.echo_top_x10[ridx, cidx]) / SCALE
            last_h = int(accumulator.last_seen_h[ridx, cidx])
            last_unix = last_h * 3600 if last_h > 0 else 0
            lon, lat = pixel_to_lonlat(int(ridx), int(cidx))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                "properties": {
                    "sizeMM": round(mm, 1),
                    "sizeInches": round(mm / 25.4, 2),
                    "posh": posh,
                    "reflectivityM20C_dBZ": round(refl, 1),
                    "echoTop_km": round(echo, 1),
                    "lastSeen": datetime.fromtimestamp(last_unix, tz=timezone.utc).isoformat() if last_unix > 0 else None,
                },
            })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "reset_schedule": "00:00 UTC daily",
            "min_inches": min_inches,
            "count": len(features),
            "total_above_threshold": n_above,
            "update_count": accumulator.update_count,
            "last_update": accumulator.last_update_ts,
        },
        "features": features,
    }


@app.get("/accumulated/stats")
def accumulated_stats():
    return accumulator.snapshot_stats()


@app.get("/swaths")
def get_swaths():
    return polygonizer.get_geojson()


@app.get("/swaths/stats")
def swaths_stats():
    return polygonizer.stats()


@app.post("/admin/polygonize")
def admin_polygonize():
    polygonizer.run()
    return {"status": "done", "stats": polygonizer.stats()}


@app.post("/admin/force-tick")
def admin_force_tick():
    tick()
    return {"status": "done", "stats": accumulator.snapshot_stats()}


@app.post("/admin/sanity-check")
def admin_sanity_check():
    sanity_check_mesh_1440()
    return {"status": "done", "stats": accumulator.snapshot_stats()}


@app.post("/admin/reset")
def admin_reset():
    with accumulator.lock:
        accumulator.mesh_max_x10[:] = 0
        accumulator.posh_max[:] = 0
        accumulator.refl_max_x10[:] = 0
        accumulator.echo_top_x10[:] = 0
        accumulator.vil_dens_x10[:] = 0
        accumulator.last_seen_h[:] = 0
        accumulator.update_count = 0
        accumulator.last_update_ts = None
    accumulator.save()
    with polygonizer.lock:
        polygonizer.cached_geojson = {
            "type": "FeatureCollection",
            "metadata": {"note": "reset"},
            "features": [],
        }
    polygonizer._save_cache()
    return {"status": "reset"}
