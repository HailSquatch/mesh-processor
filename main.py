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
from datetime import datetime, timezone, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo

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

app = FastAPI(title="StormDataPro MESH Processor", version="0.7.0")
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

# Last-seen bucket resolution. Storing the unix-seconds-bucket (not just the hour)
# lets us drive a playback timeline. 10-minute buckets give ~144 frames/day, which
# is plenty for smooth scrubbing without bloating int32 storage. last_seen_h is
# still int32 (sufficient: unix_seconds / 600 fits in int32 until year 42857).
LAST_SEEN_BUCKET_SECONDS = 600  # 10 minutes

# Persistence
STATE_DIR = os.environ.get("STATE_DIR", "/data" if os.path.isdir("/data") else "/tmp/mesh-state")
os.makedirs(STATE_DIR, exist_ok=True)
# v6 = last_seen_h is now 10-min buckets (was 1-hour buckets in v5). Loading a v5
# file under v6 code would treat 1-hour-bucket integers as 10-min buckets, putting
# all timestamps 6× too far in the past (and every pixel would look stale). Forcing
# a fresh start at deploy time is safer than auto-migrating.
STATE_FILE = os.path.join(STATE_DIR, "accumulator_v6.npz")

# ─────────────────────────────────────────────────────────────────────
# Storm Day Boundaries (Phase 5.2)
# ─────────────────────────────────────────────────────────────────────
# Storm days end at midnight Central Time. CT is the dominant time zone for
# US severe weather (Texas Triangle, Tornado Alley, Plains states). Midnight
# CT is 4-6 hours after typical evening storm wrap-up everywhere in CONUS.
#
# DST is handled automatically by ZoneInfo - APScheduler will adjust the actual
# UTC firing time twice a year as the schedule shifts between CDT (UTC-5) and
# CST (UTC-6).
STORM_DAY_TZ = ZoneInfo("America/Chicago")

# Archive directory holds daily snapshots: /data/archive/YYYY-MM-DD.json
# Each file contains both polygons and ground-truth reports for that storm day.
# Archives auto-prune at >42 days (6 weeks) to bound disk usage.
ARCHIVE_DIR = os.path.join(STATE_DIR, "archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)
ARCHIVE_RETENTION_DAYS = 42
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
                self.last_seen_h[rows[active], cols[active]] = np.int32(ts_unix // LAST_SEEN_BUCKET_SECONDS)
            self.update_count += 1
            self.last_update_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            return n_hail

    def prune_old(self, now_unix: int):
        """DEPRECATED. Previously used for rolling-window pruning. Replaced by reset_all()
        (called at 00:00 UTC daily). Kept for reference; no longer scheduled.
        """
        cutoff_h = np.int32((now_unix - (ROLLING_WINDOW_HOURS * 3600)) // LAST_SEEN_BUCKET_SECONDS)
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

# Consensus scoring thresholds — recalibrated 2026-04-26 against RadarScope ground-truth.
# Prior tuning was missing fast-moving spring squall lines; the gates below match
# RadarScope footprint within ~5-10% on the Apr 26 KS/MO MCS event.
CONSENSUS_POSH_MIN = 40            # POSH ≥ 40% gets full credit
CONSENSUS_REFL_DBZ_MIN = 50        # Reflectivity@-20C ≥ 50 dBZ gets full credit
CONSENSUS_ECHO_TOP_KM_MIN = 9.0    # ~30 kft echo top gets full credit
CONSENSUS_MIN_SCORE = 40           # pixel needs ≥ this score to be drawn

# Pre-scoring spatial dilation. MRMS MESH undercounts swath WIDTH for fast cells —
# a hail core sweeps through a pixel for less than the 2-min MRMS scan window, so
# adjacent pixels see lower MESH even though hail actually fell there. RadarScope/
# HailTrace pad the core slightly. 1 = ~1 km dilation each side, 2 = ~2 km.
PRESCORE_DILATION_PIXELS = 1

# Polygon smoothing parameters (HailTrace-style rounded look).
# We do simplify -> buffer -> negative-buffer. The buffer/de-buffer pair rounds
# corners and merges fingers. Tune SMOOTH_RADIUS to taste; bigger = blobbier.
SIMPLIFY_TOLERANCE_DEG = 0.008    # was 0.015 — preserves more detail before smoothing
SMOOTH_RADIUS_DEG = 0.012         # ~1.3 km — gives the soft-edge HailTrace blob look
MIN_POLYGON_AREA_DEG2 = 0.00003   # ~0.4 km² — was 0.0001, killed too many real cells

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

                # Pre-scoring spatial dilation. Pads the MESH-meets mask so storm
                # cells get scored at their true (radar-observed) width instead of
                # the per-pixel-instantaneous footprint MRMS reports. This is the
                # single biggest cause of our maps under-painting fast squall lines
                # vs RadarScope.
                if PRESCORE_DILATION_PIXELS > 0:
                    crop_mesh_meets = ndimage.binary_dilation(
                        crop_mesh_meets,
                        iterations=PRESCORE_DILATION_PIXELS,
                    )

                # Consensus scoring — give half-credit for missing data on BOTH
                # POSH and Reflectivity. POSH product flakes out on fast cells and
                # we shouldn't dock real hail just because the secondary product
                # was lagging. NOAA documentation acknowledges POSH is computed on
                # a slower cadence than MESH for some cell types.
                # Scoring: MESH(40) + POSH(0/15/30) + Refl(0/10/20) + EchoTop(0/10) max 100
                crop_score = np.full(crop_mesh_meets.shape, 40, dtype=np.uint8)
                crop_score[~crop_mesh_meets] = 0

                # POSH: 30 if ≥ threshold, 15 if missing (sentinel 0), 0 if low
                posh_strong = (crop_posh >= CONSENSUS_POSH_MIN) & crop_mesh_meets
                posh_missing = (crop_posh == 0) & crop_mesh_meets
                crop_score += posh_strong.astype(np.uint8) * 30
                crop_score += (posh_missing & ~posh_strong).astype(np.uint8) * 15

                # Reflectivity: 20 if ≥ threshold, 10 if missing (sentinel 0), 0 if low
                refl_missing = (crop_refl == 0) & crop_mesh_meets
                refl_strong = (crop_refl >= refl_dbz_x10_thresh) & crop_mesh_meets
                crop_score += refl_strong.astype(np.uint8) * 20
                crop_score += (refl_missing & ~refl_strong).astype(np.uint8) * 10

                # EchoTop: 10 if ≥ threshold, 0 otherwise (no missing-data half credit
                # — EchoTop_50 is highly reliable and "missing" usually means storm
                # tops are below 50 dBZ, which is genuinely a low-confidence signal)
                crop_score += ((crop_echo >= echo_km_x10_thresh) & crop_mesh_meets).astype(np.uint8) * 10

                pass_mask = crop_score >= CONSENSUS_MIN_SCORE
                crop_mesh_meets = crop_score = None
                refl_missing = refl_strong = posh_missing = posh_strong = None
                gc.collect()

                if not pass_mask.any():
                    pass_mask = None
                    continue

                # Morphology — tuned to AVOID amputating thin storm tracks.
                # Prior config used `binary_opening(iter=1)` which used a default
                # 3×3 plus-cross structuring element and erased any feature narrower
                # than ~3 km. Fast squall lines drop 1-2 pixel-wide tracks; opening
                # was killing them entirely (matched our Apr 26 KS missing-line bug).
                # New strategy: closing only. Rely on the consensus score itself
                # plus area filtering to reject noise. Closing iterations bridge
                # gaps in the accumulated track without removing real cells.
                if band["min_inches"] >= 2.0:
                    cleaned = ndimage.binary_closing(pass_mask, iterations=2)
                elif band["min_inches"] >= 1.5:
                    cleaned = ndimage.binary_closing(pass_mask, iterations=2)
                else:
                    # Light hail: extra closing to bridge fragmented tracks
                    cleaned = ndimage.binary_closing(pass_mask, iterations=3)
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

                # Rounded HailTrace-style smoothing:
                #   1. Light Douglas-Peucker simplify
                #   2. Outward buffer to round corners
                #   3. Inward buffer of equal distance to restore size
                # The buffer pair acts like a low-pass filter on the polygon
                # outline and merges thin gaps between adjacent storm cells.
                simplified = []
                for p in raw_polys:
                    sp = p.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
                    if sp.is_empty or sp.area < MIN_POLYGON_AREA_DEG2:
                        continue
                    # Buffer-smooth. resolution=8 = 8 segments per quarter-circle = nice round corners.
                    try:
                        smoothed = sp.buffer(SMOOTH_RADIUS_DEG, resolution=8, join_style=1).buffer(
                            -SMOOTH_RADIUS_DEG, resolution=8, join_style=1
                        )
                    except Exception:
                        smoothed = sp  # fall back to un-smoothed if buffer fails (rare)
                    if smoothed.is_empty:
                        continue
                    # buffer(...) can return MultiPolygon — handle both
                    geoms = [smoothed] if smoothed.geom_type == "Polygon" else list(smoothed.geoms)
                    for g in geoms:
                        if g.area >= MIN_POLYGON_AREA_DEG2 and not g.is_empty:
                            simplified.append(g)

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
                            # POSH: full / half / zero
                            if peak_posh >= CONSENSUS_POSH_MIN:
                                sc += 30
                            elif peak_posh == 0:
                                sc += 15  # half credit for missing POSH
                            # Reflectivity: full / half / zero
                            if peak_refl_x10 >= refl_dbz_x10_thresh:
                                sc += 20
                            elif peak_refl_x10 == 0:
                                sc += 10  # half credit for missing refl
                            if peak_echo_x10 >= echo_km_x10_thresh: sc += 10
                            peak_score = sc

                            sub_last = crop_last_seen[sub_r1:sub_r0, sub_c0:sub_c1]
                            sub_last_nz = sub_last[sub_last > 0]
                            if len(sub_last_nz) > 0:
                                last_unix = int(sub_last_nz.max()) * LAST_SEEN_BUCKET_SECONDS
                                first_unix = int(sub_last_nz.min()) * LAST_SEEN_BUCKET_SECONDS
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
                        "mesh_in_band_points": 40,
                        "posh_full_points": 30,
                        "posh_missing_half_credit": 15,
                        "reflectivity_full_points": 20,
                        "reflectivity_missing_half_credit": 10,
                        "echo_top_points": 10,
                        "prescore_dilation_pixels": PRESCORE_DILATION_PIXELS,
                        "smooth_radius_deg": SMOOTH_RADIUS_DEG,
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

    def build_playback(self, step_minutes: int = 15) -> dict:
        """Build a series of cumulative GeoJSON frames for timeline scrubbing.

        Strategy: for each timestep, polygonize only the pixels whose last_seen_h
        bucket falls AT OR BEFORE that timestep. This produces a "growing storm"
        loop you can scrub. step_minutes is clamped to multiples of LAST_SEEN_BUCKET
        (10 min), so 10/20/30/60 are natural choices. Default 15 → 96 frames/day,
        but we round to the bucket so 15 effectively becomes 20.

        Memory: we bbox-crop just like the live polygonizer. For each frame we
        do consensus + morphology + polygonize + smooth on the crop. Cost is
        roughly N_frames × normal-polygonize-time, so ~30-90 sec for a full
        24h day. Result is JSON, not cached on disk — re-runs on demand.
        """
        t0 = datetime.now(timezone.utc)
        # Round step to bucket multiple
        bucket_min = LAST_SEEN_BUCKET_SECONDS // 60
        step_buckets = max(1, round(step_minutes / bucket_min))
        effective_step_min = step_buckets * bucket_min

        try:
            min_thresh_x10 = int(POLYGON_BANDS[0]["min_mm"] * SCALE)
            with accumulator.lock:
                rows_any = np.any(accumulator.mesh_max_x10 >= min_thresh_x10, axis=1)
                cols_any = np.any(accumulator.mesh_max_x10 >= min_thresh_x10, axis=0)
                update_count = accumulator.update_count
                last_update_ts = accumulator.last_update_ts

            if not rows_any.any():
                return {
                    "frames": [],
                    "metadata": {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "step_minutes": effective_step_min,
                        "frame_count": 0,
                        "note": "No hail accumulated since last reset",
                    },
                }

            gr0 = max(0, int(np.argmax(rows_any)) - 5)
            gr1 = min(GRID_ROWS, GRID_ROWS - int(np.argmax(rows_any[::-1])) + 5)
            gc0 = max(0, int(np.argmax(cols_any)) - 5)
            gc1 = min(GRID_COLS, GRID_COLS - int(np.argmax(cols_any[::-1])) + 5)
            rows_any = cols_any = None

            with accumulator.lock:
                crop_mesh = accumulator.mesh_max_x10[gr0:gr1, gc0:gc1].copy()
                crop_posh = accumulator.posh_max[gr0:gr1, gc0:gc1].copy()
                crop_refl = accumulator.refl_max_x10[gr0:gr1, gc0:gc1].copy()
                crop_echo = accumulator.echo_top_x10[gr0:gr1, gc0:gc1].copy()
                crop_last_seen = accumulator.last_seen_h[gr0:gr1, gc0:gc1].copy()

            # Find min and max bucket indices in the crop
            nz_buckets = crop_last_seen[crop_last_seen > 0]
            if len(nz_buckets) == 0:
                return {
                    "frames": [],
                    "metadata": {
                        "step_minutes": effective_step_min,
                        "frame_count": 0,
                        "note": "Accumulator has hail but no last_seen timestamps",
                    },
                }
            min_bucket = int(nz_buckets.min())
            max_bucket = int(nz_buckets.max())
            nz_buckets = None

            # Generate frame timestamps
            frame_buckets = list(range(min_bucket, max_bucket + 1, step_buckets))
            if frame_buckets[-1] < max_bucket:
                frame_buckets.append(max_bucket)

            refl_dbz_x10_thresh = int(CONSENSUS_REFL_DBZ_MIN * SCALE)
            echo_km_x10_thresh = int(CONSENSUS_ECHO_TOP_KM_MIN * SCALE)

            crop_transform = Affine(
                GRID_DEG, 0, (GRID_LON1_360 - 360) + gc0 * GRID_DEG,
                0, -GRID_DEG, GRID_LAT1 - gr0 * GRID_DEG
            )

            frames = []
            for frame_idx, bucket_cap in enumerate(frame_buckets):
                # "Cumulative up to and including this bucket"
                time_mask = (crop_last_seen > 0) & (crop_last_seen <= bucket_cap)
                if not time_mask.any():
                    continue

                frame_features = []
                for band in POLYGON_BANDS:
                    band_min_x10 = int(band["min_mm"] * SCALE)
                    band_max_x10 = int(band["max_mm"] * SCALE) if band["max_mm"] is not None else None

                    if band_max_x10 is None:
                        crop_mesh_meets = (crop_mesh >= band_min_x10) & time_mask
                    else:
                        crop_mesh_meets = (crop_mesh >= band_min_x10) & (crop_mesh < band_max_x10) & time_mask

                    if not crop_mesh_meets.any():
                        continue

                    if PRESCORE_DILATION_PIXELS > 0:
                        crop_mesh_meets = ndimage.binary_dilation(
                            crop_mesh_meets, iterations=PRESCORE_DILATION_PIXELS
                        )

                    crop_score = np.full(crop_mesh_meets.shape, 40, dtype=np.uint8)
                    crop_score[~crop_mesh_meets] = 0
                    posh_strong = (crop_posh >= CONSENSUS_POSH_MIN) & crop_mesh_meets
                    posh_missing = (crop_posh == 0) & crop_mesh_meets
                    crop_score += posh_strong.astype(np.uint8) * 30
                    crop_score += (posh_missing & ~posh_strong).astype(np.uint8) * 15
                    refl_strong = (crop_refl >= refl_dbz_x10_thresh) & crop_mesh_meets
                    refl_missing = (crop_refl == 0) & crop_mesh_meets
                    crop_score += refl_strong.astype(np.uint8) * 20
                    crop_score += (refl_missing & ~refl_strong).astype(np.uint8) * 10
                    crop_score += ((crop_echo >= echo_km_x10_thresh) & crop_mesh_meets).astype(np.uint8) * 10

                    pass_mask = crop_score >= CONSENSUS_MIN_SCORE
                    if not pass_mask.any():
                        continue

                    cleaned = ndimage.binary_closing(pass_mask, iterations=2)
                    if not cleaned.any():
                        continue

                    uint8_mask = cleaned.astype(np.uint8)
                    for geom, val in rio_features.shapes(uint8_mask, mask=cleaned, transform=crop_transform):
                        if val != 1:
                            continue
                        p = shp_shape(geom).simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
                        if p.is_empty or p.area < MIN_POLYGON_AREA_DEG2:
                            continue
                        try:
                            sm = p.buffer(SMOOTH_RADIUS_DEG, resolution=8, join_style=1).buffer(
                                -SMOOTH_RADIUS_DEG, resolution=8, join_style=1
                            )
                        except Exception:
                            sm = p
                        if sm.is_empty:
                            continue
                        geoms = [sm] if sm.geom_type == "Polygon" else list(sm.geoms)
                        for g in geoms:
                            if g.area < MIN_POLYGON_AREA_DEG2 or g.is_empty:
                                continue
                            frame_features.append({
                                "type": "Feature",
                                "geometry": shp_mapping(g),
                                "properties": {
                                    "bandMinInches": band["min_inches"],
                                    "bandMaxInches": band["max_inches"],
                                    "bandLabel": band["desc"],
                                    "color": band["color"],
                                    "thresholdInches": band["min_inches"],
                                    "thresholdLabel": band["desc"],
                                },
                            })

                frames.append({
                    "timestamp": datetime.fromtimestamp(
                        bucket_cap * LAST_SEEN_BUCKET_SECONDS, tz=timezone.utc
                    ).isoformat(),
                    "bucket": bucket_cap,
                    "frame_index": frame_idx,
                    "polygon_count": len(frame_features),
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": frame_features,
                    },
                })

            crop_mesh = crop_posh = crop_refl = crop_echo = crop_last_seen = None
            gc.collect()

            elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return {
                "frames": frames,
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "step_minutes": effective_step_min,
                    "frame_count": len(frames),
                    "first_timestamp": frames[0]["timestamp"] if frames else None,
                    "last_timestamp": frames[-1]["timestamp"] if frames else None,
                    "accumulator_update_count": update_count,
                    "accumulator_last_update": last_update_ts,
                },
            }
        except Exception as e:
            log.error(f"Playback build failed: {e}", exc_info=True)
            return {"frames": [], "metadata": {"error": str(e)}}


polygonizer = Polygonizer()


# ─────────────────────────────────────────────────────────────────────
# HRRR Forecast — 6h hail forecast pipeline (Phase 7)
# ─────────────────────────────────────────────────────────────────────
# HRRR (High-Resolution Rapid Refresh) is NOAA's 3km convection-allowing model.
# It runs hourly with 18h forecasts (and 48h on 00/06/12/18z runs). We pull just
# the HAIL field at the surface, which is the HAILCAST max hail diameter forecast
# in METERS. We convert to mm and apply the same band cutoffs we use for MRMS
# observations (1.0", 1.5", 2.0", 2.75").
#
# Strategy:
#   - Every 15 min: probe the latest available HRRR run via .idx file
#   - Range-fetch ONLY the HAIL records for forecast hours +1 through +6
#   - Decode with pygrib, polygonize each forecast hour, save GeoJSON
#
# Each HAIL record range-fetch is ~150KB-1MB (HAIL is small/sparse vs full grib2)
# so 6 hours × ~500KB ≈ 3MB per run cycle. Cheap.
#
# HRRR runs are typically posted ~50 min after init. Logic: try the run from
# (current_hour - 1) first; if .idx not yet posted, fall back to (current_hour - 2).
HRRR_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
HRRR_FORECAST_HOURS = 6  # +1 to +6 inclusive
HRRR_REFRESH_MINUTES = 15  # how often to check for a new run
HRRR_CACHE_FILE = os.path.join(STATE_DIR, "hrrr_forecast.json")

# HRRR HAIL is in meters of max diameter. Same size bands as observed MRMS, in mm.
# We can reuse POLYGON_BANDS directly since both products are "hail size."
# Forecast confidence has its own visual-distinction in the frontend (dashed,
# striped fill), so we don't need to filter further on the backend — output every
# pixel that meets the 1.0" threshold.

class HRRRForecast:
    """Fetches HRRR HAIL forecast and produces 1-6h hail polygons.
    
    The cached_geojson layout is:
        {
            "type": "FeatureCollection",
            "metadata": {
                "init_time":  "2026-04-26T18:00:00Z",  # HRRR run cycle
                "generated_at": "...",
                "forecast_hours": [1,2,3,4,5,6],
                "polygon_count": N,
            },
            "features": [
                {... "properties": {"forecastHour": 1, "validTime": "...",
                                    "color": "...", ...}}
            ]
        }
    Frontend filters by forecastHour to render time-stepped overlays.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.cached_geojson: dict = {
            "type": "FeatureCollection",
            "metadata": {"note": "HRRR forecast not yet fetched"},
            "features": [],
        }
        self.last_run_init: Optional[str] = None
        self.last_attempt_ts: Optional[str] = None
        self.last_error: Optional[str] = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(HRRR_CACHE_FILE):
            try:
                with open(HRRR_CACHE_FILE, "r") as f:
                    self.cached_geojson = json.load(f)
                meta = self.cached_geojson.get("metadata", {})
                self.last_run_init = meta.get("init_time")
                n = len(self.cached_geojson.get("features", []))
                log.info(f"Loaded {n} cached HRRR polygons (init={self.last_run_init})")
            except Exception as e:
                log.warning(f"HRRR cache load failed: {e}")

    def _save_cache(self):
        tmp = HRRR_CACHE_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.cached_geojson, f, separators=(",", ":"))
            os.replace(tmp, HRRR_CACHE_FILE)
        except Exception as e:
            log.error(f"HRRR cache save failed: {e}")

    def _idx_url(self, run_dt: datetime, fxx: int) -> str:
        return (f"{HRRR_NOMADS_BASE}/hrrr.{run_dt.strftime('%Y%m%d')}/conus/"
                f"hrrr.t{run_dt.strftime('%H')}z.wrfsfcf{fxx:02d}.grib2.idx")

    def _grib_url(self, run_dt: datetime, fxx: int) -> str:
        return (f"{HRRR_NOMADS_BASE}/hrrr.{run_dt.strftime('%Y%m%d')}/conus/"
                f"hrrr.t{run_dt.strftime('%H')}z.wrfsfcf{fxx:02d}.grib2")

    def _find_hail_byte_range(self, idx_text: str) -> Optional[tuple]:
        """Parse .idx text to find the byte range for HAIL:surface.

        idx files look like:
            71:34884036:d=2026042618:HAIL:surface:6 hour fcst:
            72:36136433:d=2026042618:NEXT_VAR:...
        We need the start byte from the HAIL line, and end byte = start of NEXT line - 1.
        """
        lines = idx_text.strip().split("\n")
        hail_idx = None
        for i, line in enumerate(lines):
            # ":HAIL:surface:" is the canonical match. We avoid matching MAXHAIL or
            # other fields by requiring the exact "HAIL:" substring with surroundings.
            parts = line.split(":")
            if len(parts) >= 5 and parts[3] == "HAIL" and parts[4] == "surface":
                hail_idx = i
                break
        if hail_idx is None:
            return None
        # Start byte from the HAIL line
        try:
            start = int(lines[hail_idx].split(":")[1])
        except (IndexError, ValueError):
            return None
        # End byte = start of next line - 1; if HAIL is the last record, omit end (open-ended fetch)
        if hail_idx + 1 < len(lines) and lines[hail_idx + 1].strip():
            try:
                end = int(lines[hail_idx + 1].split(":")[1]) - 1
            except (IndexError, ValueError):
                end = None
        else:
            end = None
        return (start, end)

    def _fetch_hail_for_hour(self, run_dt: datetime, fxx: int) -> Optional[dict]:
        """Returns {'values': np.ndarray (mm), 'lats': np.ndarray, 'lons': np.ndarray, 'valid_time': datetime} or None."""
        idx_url = self._idx_url(run_dt, fxx)
        grib_url = self._grib_url(run_dt, fxx)
        try:
            r_idx = requests.get(idx_url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if r_idx.status_code != 200:
                return None
            byte_range = self._find_hail_byte_range(r_idx.text)
            if byte_range is None:
                log.warning(f"HRRR f{fxx:02d}: HAIL not in idx")
                return None
            start, end = byte_range
            range_header = f"bytes={start}-" + (f"{end}" if end is not None else "")
            r_grib = requests.get(
                grib_url,
                headers={"User-Agent": USER_AGENT, "Range": range_header},
                timeout=60,
            )
            if r_grib.status_code not in (200, 206):
                log.warning(f"HRRR f{fxx:02d} byte fetch failed: {r_grib.status_code}")
                return None

            # Write to temp file, decode with pygrib (it doesn't accept BytesIO well across versions)
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
                tmp.write(r_grib.content)
                tmp_path = tmp.name
            try:
                grbs = pygrib.open(tmp_path)
                grb = grbs.message(1)  # first (and only) record in this range
                vals_m = np.array(grb.values, dtype=np.float32)  # meters
                lats, lons = grb.latlons()
                valid_time = grb.validDate
                grbs.close()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Convert m → mm (HAILCAST output convention)
            vals_mm = vals_m * 1000.0
            # Make sure lons are in [-180, 180]
            lons = np.where(lons > 180, lons - 360, lons)
            return {
                "values_mm": vals_mm,
                "lats": lats.astype(np.float32),
                "lons": lons.astype(np.float32),
                "valid_time": valid_time if valid_time.tzinfo else valid_time.replace(tzinfo=timezone.utc),
            }
        except Exception as e:
            log.warning(f"HRRR f{fxx:02d} fetch failed: {e}")
            return None

    def _polygonize_hour(self, values_mm: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                        forecast_hour: int, valid_time: datetime, init_time: datetime) -> list:
        """Convert one hour's HAIL grid into GeoJSON features (one per band).
        
        We don't have an Affine transform (HRRR is Lambert Conformal, not regular
        lat/lon), so we can't use rasterio.features.shapes directly with geo
        coordinates. Approach: run shapes() in pixel coordinates, then map polygon
        vertices through the lats/lons arrays via bilinear interp.
        """
        features = []
        nrows, ncols = values_mm.shape
        # Identity affine — vertices come out in (col, row) pixel space
        pixel_transform = Affine(1.0, 0, 0, 0, 1.0, 0)

        for band in POLYGON_BANDS:
            band_min_x10 = int(band["min_mm"] * SCALE)
            band_max_x10 = int(band["max_mm"] * SCALE) if band["max_mm"] is not None else None
            vals_x10 = (values_mm * SCALE).astype(np.int32)

            if band_max_x10 is None:
                mask = vals_x10 >= band_min_x10
            else:
                mask = (vals_x10 >= band_min_x10) & (vals_x10 < band_max_x10)

            if not mask.any():
                continue

            # Light closing to bridge fragmented model output
            mask = ndimage.binary_closing(mask, iterations=1)
            if not mask.any():
                continue

            uint8_mask = mask.astype(np.uint8)
            for geom, val in rio_features.shapes(uint8_mask, mask=mask, transform=pixel_transform):
                if val != 1:
                    continue
                # geom["coordinates"] is a list of rings, each a list of [col, row] pairs
                # Map each vertex to (lon, lat) via bilinear interp in lats/lons
                geo_rings = []
                for ring in geom["coordinates"]:
                    geo_ring = []
                    for col, row in ring:
                        # Clamp to grid bounds
                        c0 = int(np.clip(np.floor(col), 0, ncols - 1))
                        c1 = int(np.clip(c0 + 1, 0, ncols - 1))
                        r0 = int(np.clip(np.floor(row), 0, nrows - 1))
                        r1 = int(np.clip(r0 + 1, 0, nrows - 1))
                        fc = col - c0
                        fr = row - r0
                        # Bilinear
                        lat_v = (lats[r0, c0] * (1 - fr) * (1 - fc) +
                                 lats[r0, c1] * (1 - fr) * fc +
                                 lats[r1, c0] * fr * (1 - fc) +
                                 lats[r1, c1] * fr * fc)
                        lon_v = (lons[r0, c0] * (1 - fr) * (1 - fc) +
                                 lons[r0, c1] * (1 - fr) * fc +
                                 lons[r1, c0] * fr * (1 - fc) +
                                 lons[r1, c1] * fr * fc)
                        geo_ring.append([float(lon_v), float(lat_v)])
                    geo_rings.append(geo_ring)
                if not geo_rings or len(geo_rings[0]) < 4:
                    continue
                try:
                    poly = shp_shape({"type": "Polygon", "coordinates": geo_rings})
                except Exception:
                    continue
                if poly.is_empty or poly.area < MIN_POLYGON_AREA_DEG2:
                    continue
                # Smooth with the same buffer trick as observed swaths
                try:
                    sp = poly.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
                    smoothed = sp.buffer(SMOOTH_RADIUS_DEG, resolution=8, join_style=1).buffer(
                        -SMOOTH_RADIUS_DEG, resolution=8, join_style=1
                    )
                except Exception:
                    smoothed = poly
                if smoothed.is_empty:
                    continue
                geoms = [smoothed] if smoothed.geom_type == "Polygon" else list(smoothed.geoms)
                for g in geoms:
                    if g.area < MIN_POLYGON_AREA_DEG2 or g.is_empty:
                        continue
                    # Find peak hail value within polygon footprint (cheap approx:
                    # use band_min as floor; for accurate peak we'd re-sample, skipping
                    # for performance — band already gives size range)
                    peak_mm = band["min_mm"]
                    if band["max_mm"]:
                        peak_mm = (band["min_mm"] + band["max_mm"]) / 2

                    features.append({
                        "type": "Feature",
                        "geometry": shp_mapping(g),
                        "properties": {
                            "forecastHour": forecast_hour,
                            "validTime": valid_time.isoformat(),
                            "initTime": init_time.isoformat(),
                            "bandMinInches": band["min_inches"],
                            "bandMaxInches": band["max_inches"],
                            "bandLabel": band["desc"],
                            "color": band["color"],
                            "peakSizeInches": round(peak_mm / 25.4, 2),
                            "isForecast": True,
                        },
                    })
        return features

    def _try_run(self, run_dt: datetime) -> Optional[dict]:
        """Attempt to fetch and polygonize an entire HRRR run for forecast hours 1..N.
        Returns full GeoJSON dict or None if the run isn't ready."""
        # Probe f01 first — if its idx doesn't exist yet, the run isn't posted
        probe_url = self._idx_url(run_dt, 1)
        try:
            probe = requests.head(probe_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        except Exception as e:
            log.warning(f"HRRR probe failed for {run_dt}: {e}")
            return None
        if probe.status_code != 200:
            return None

        log.info(f"HRRR run {run_dt.isoformat()} found, fetching {HRRR_FORECAST_HOURS}h")
        all_features = []
        for fxx in range(1, HRRR_FORECAST_HOURS + 1):
            data = self._fetch_hail_for_hour(run_dt, fxx)
            if data is None:
                continue
            features = self._polygonize_hour(
                data["values_mm"], data["lats"], data["lons"],
                forecast_hour=fxx, valid_time=data["valid_time"], init_time=run_dt
            )
            all_features.extend(features)
            # Free arrays before next hour
            data = None
            gc.collect()

        return {
            "type": "FeatureCollection",
            "metadata": {
                "init_time": run_dt.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "forecast_hours": list(range(1, HRRR_FORECAST_HOURS + 1)),
                "polygon_count": len(all_features),
                "source": "HRRR HAILCAST (HAIL field, surface)",
            },
            "features": all_features,
        }

    def refresh(self):
        """Find the freshest available HRRR run and update the cache.
        
        HRRR runs are posted ~50 min after init. We try (now-1h), (now-2h), (now-3h)
        in order, stopping at the first run that exists AND is newer than what we
        have cached.
        """
        self.last_attempt_ts = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc)
        # Truncate to top of hour
        candidate_hours = [now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=h)
                           for h in (1, 2, 3, 4)]
        for run_dt in candidate_hours:
            run_iso = run_dt.isoformat()
            if self.last_run_init == run_iso:
                # We already have this run cached
                log.info(f"HRRR refresh: already on run {run_iso}, skipping")
                return
            try:
                geojson = self._try_run(run_dt)
            except Exception as e:
                log.error(f"HRRR run {run_iso} failed: {e}", exc_info=True)
                self.last_error = str(e)
                continue
            if geojson is None:
                continue
            # Got it!
            with self.lock:
                self.cached_geojson = geojson
                self.last_run_init = run_iso
                self.last_error = None
            self._save_cache()
            log.info(f"HRRR refreshed: run {run_iso} → {len(geojson['features'])} polygons")
            return
        # All attempts failed
        log.warning("HRRR refresh: no runs available in last 4 hours")

    def get_geojson(self) -> dict:
        with self.lock:
            return self.cached_geojson

    def stats(self) -> dict:
        with self.lock:
            return {
                "last_run_init": self.last_run_init,
                "last_attempt": self.last_attempt_ts,
                "polygon_count": len(self.cached_geojson.get("features", [])),
                "forecast_hours": HRRR_FORECAST_HOURS,
                "last_error": self.last_error,
                "cache_file": HRRR_CACHE_FILE,
            }


hrrr_forecast = HRRRForecast()


# ─────────────────────────────────────────────────────────────────────
# Archiver — daily snapshot storage
# ─────────────────────────────────────────────────────────────────────
class Archiver:
    """Manages daily archives of polygons + ground-truth reports.

    File layout: /data/archive/YYYY-MM-DD.json (one file per storm day, CT-based)
    Each file contains:
      {
        "date": "2026-04-25",
        "archived_at": "2026-04-26T05:00:00+00:00",
        "swaths": {<full /swaths GeoJSON>},
        "ground_truth": {<full SPC reports GeoJSON>}
      }
    """

    def __init__(self):
        self.lock = threading.Lock()

    def archive_date(self) -> str:
        """The date this archive represents (yesterday in storm-day CT terms).
        Called at reset-time, this returns yesterday in Central Time."""
        now_ct = datetime.now(STORM_DAY_TZ)
        # Reset fires at 00:00 CT, so "today CT" at that moment is the new day -
        # the data we're archiving belongs to "yesterday CT"
        yesterday_ct = (now_ct - timedelta(hours=1)).date()
        return yesterday_ct.isoformat()

    def save_snapshot(self, archive_date_str: str) -> dict:
        """Snapshot today's polygons and ground-truth, save to disk."""
        try:
            # Snapshot polygons (will get current /swaths content)
            polygons_geojson = polygonizer.get_geojson()
            poly_count = len(polygons_geojson.get("features", []))

            # Snapshot ground-truth (fetch fresh from SPC for today's reports)
            ground_truth_geojson = self._fetch_spc_today()
            gt_count = len(ground_truth_geojson.get("features", []))

            archive = {
                "date": archive_date_str,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "polygon_count": poly_count,
                "ground_truth_count": gt_count,
                "swaths": polygons_geojson,
                "ground_truth": ground_truth_geojson,
            }

            path = os.path.join(ARCHIVE_DIR, f"{archive_date_str}.json")
            with self.lock:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(archive, f, default=str)
                os.replace(tmp_path, path)

            log.info(f"Archived {archive_date_str}: {poly_count} polygons, {gt_count} ground-truth reports → {path}")
            return {"date": archive_date_str, "polygons": poly_count, "reports": gt_count, "path": path}
        except Exception as e:
            log.exception(f"Archive failed for {archive_date_str}: {e}")
            return {"date": archive_date_str, "error": str(e)}

    def _fetch_spc_today(self) -> dict:
        """Pull SPC's filtered hail reports CSV for today and convert to GeoJSON."""
        try:
            url = "https://www.spc.noaa.gov/climo/reports/today_filtered_hail.csv"
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if not r.ok:
                return {"type": "FeatureCollection", "features": [], "error": f"HTTP {r.status_code}"}
            # SPC's "today" file is the convective day starting at 12:00 UTC today
            convective_start = datetime.now(timezone.utc).date()
            return self._parse_spc_csv(r.text, convective_day_start=convective_start)
        except Exception as e:
            log.warning(f"SPC fetch failed: {e}")
            return {"type": "FeatureCollection", "features": [], "error": str(e)}

    def _parse_spc_csv(self, csv_text: str, convective_day_start: Optional[date] = None) -> dict:
        """Convert SPC filtered_hail.csv text into GeoJSON FeatureCollection.
        CSV columns: Time, Size, Location, County, State, Lat, Lon, Comments...
        
        SPC's "convective day" runs from 12:00 UTC to 12:00 UTC the next calendar day.
        Time column is HHMM in UTC only — no date. We compute the actual UTC timestamp
        by combining the row's HHMM with the convective day's start date:
          - If HH >= 12: report is on convective_day_start
          - If HH <  12: report is on convective_day_start + 1
        
        Without convective_day_start, we still parse but skip timestamp generation.
        """
        features = []
        for line in csv_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("Time"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                time_s, size_s, loc, county, state, lat_s, lon_s = parts[:7]
                comments = ",".join(parts[7:]).strip() if len(parts) > 7 else ""
                size = int(size_s) if size_s.strip().isdigit() else 0
                lat = float(lat_s)
                lon = float(lon_s)
                if size < 75:  # ignore sub-3/4" reports
                    continue

                # Compute real UTC timestamp if we know the convective day
                timestamp_iso = None
                if convective_day_start and len(time_s.strip()) == 4 and time_s.strip().isdigit():
                    hhmm = time_s.strip()
                    h = int(hhmm[:2])
                    m = int(hhmm[2:4])
                    base = convective_day_start
                    if h < 12:
                        base = base + timedelta(days=1)
                    try:
                        report_dt = datetime(base.year, base.month, base.day, h, m, 0, tzinfo=timezone.utc)
                        timestamp_iso = report_dt.isoformat()
                    except ValueError:
                        pass

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "size": size,
                        "sizeInches": round(size / 100.0, 2),
                        "time": time_s,
                        "timestamp": timestamp_iso,
                        "location": loc,
                        "county": county,
                        "state": state,
                        "comments": comments,
                    },
                })
            except (ValueError, IndexError):
                continue
        return {"type": "FeatureCollection", "features": features}

    def fetch_spc_for_date(self, date_str: str) -> dict:
        """Fetch SPC archived reports for a specific date (YYYY-MM-DD).
        Used for hydrating historical view with fresh data when available.
        SPC URL pattern: https://www.spc.noaa.gov/climo/reports/{YYMMDD}_rpts_filtered_hail.csv"""
        try:
            d = date.fromisoformat(date_str)
            # SPC archive URL uses YYMMDD format
            yymmdd = d.strftime("%y%m%d")
            url = f"https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_filtered_hail.csv"
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if not r.ok:
                return {"type": "FeatureCollection", "features": [], "note": f"SPC archive returned {r.status_code}"}
            # The SPC archive file represents the convective day starting at 12 UTC on `d`
            return self._parse_spc_csv(r.text, convective_day_start=d)
        except Exception as e:
            log.warning(f"SPC archive fetch failed for {date_str}: {e}")
            return {"type": "FeatureCollection", "features": [], "error": str(e)}

    def load_archive(self, date_str: str) -> Optional[dict]:
        """Read an archived day's snapshot from disk. Returns None if not found."""
        try:
            # Validate date format - prevents path traversal
            date.fromisoformat(date_str)
        except ValueError:
            return None
        path = os.path.join(ARCHIVE_DIR, f"{date_str}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load archive {date_str}: {e}")
            return None

    def list_dates(self) -> list:
        """Return sorted list of all archived dates (newest first)."""
        try:
            files = [f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".json")]
            dates = sorted([f[:-5] for f in files], reverse=True)
            # Filter out anything that doesn't parse as a valid date
            valid = []
            for d in dates:
                try:
                    date.fromisoformat(d)
                    valid.append(d)
                except ValueError:
                    pass
            return valid
        except Exception:
            return []

    def prune_old(self) -> int:
        """Delete archives older than ARCHIVE_RETENTION_DAYS. Returns count deleted."""
        cutoff = date.today() - timedelta(days=ARCHIVE_RETENTION_DAYS)
        deleted = 0
        try:
            for f in os.listdir(ARCHIVE_DIR):
                if not f.endswith(".json"):
                    continue
                try:
                    file_date = date.fromisoformat(f[:-5])
                    if file_date < cutoff:
                        os.remove(os.path.join(ARCHIVE_DIR, f))
                        deleted += 1
                except (ValueError, OSError):
                    continue
            if deleted:
                log.info(f"Pruned {deleted} archive files older than {cutoff.isoformat()}")
        except Exception as e:
            log.warning(f"Archive prune failed: {e}")
        return deleted


archiver = Archiver()


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

                # last_seen — using 10-min buckets (LAST_SEEN_BUCKET_SECONDS) for playback
                active = hail_mesh_x10 >= MIN_HAIL_STORED
                accumulator.last_seen_h[rows[active], cols[active]] = np.int32(ts_unix // LAST_SEEN_BUCKET_SECONDS)
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
    """Runs at 00:00 America/Chicago every day. 
    
    Sequence:
      1. Archive yesterday's polygons + ground-truth to /data/archive/YYYY-MM-DD.json
      2. Prune any archives older than 6 weeks
      3. Wipe accumulator so today starts fresh
      4. Run polygonizer once so /swaths returns empty FeatureCollection immediately
    """
    archive_date_str = archiver.archive_date()
    log.info(f"midnight_reset starting — archiving {archive_date_str}")
    
    # Step 1: Archive yesterday's data BEFORE we wipe anything
    archive_result = archiver.save_snapshot(archive_date_str)
    log.info(f"midnight_reset archive: {archive_result}")
    
    # Step 2: Prune old archives (>6 weeks)
    archiver.prune_old()
    
    # Step 3: Wipe accumulator
    n_cleared = accumulator.reset_all()
    
    # Step 4: Re-run polygonizer so /swaths returns empty immediately
    polygonizer.run()
    log.info(f"midnight_reset complete: cleared {n_cleared} pixels, polygonizer re-ran")


def periodic_polygonize():
    polygonizer.run()


def periodic_hrrr_refresh():
    """Check for a new HRRR run and refresh forecast cache if found."""
    hrrr_forecast.refresh()


# ── Scheduler ──
# Scheduler runs in UTC for interval jobs (consistent regardless of DST), but
# the daily reset job uses America/Chicago timezone so it always fires at
# midnight Central Time, automatically adjusting for CDT vs CST.
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    tick, "interval", minutes=2, id="mesh_tick",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
    max_instances=1, coalesce=True,
)
scheduler.add_job(periodic_save, "interval", minutes=10, id="save_state",
                  max_instances=1, coalesce=True)
# Daily reset at midnight Central Time — covers Texas Triangle/Tornado Alley/Plains
# storm windows. Using ZoneInfo so DST transitions are handled automatically:
# 00:00 CT = 05:00 UTC during CDT (March-November), 06:00 UTC during CST (Nov-March)
scheduler.add_job(midnight_reset, "cron", hour=0, minute=0, id="midnight_reset",
                  timezone=STORM_DAY_TZ,
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
# HRRR forecast refresh — every 15 min, checks if a newer model run is available
# and re-pulls + re-polygonizes the 1-6h hail forecast if so.
scheduler.add_job(
    periodic_hrrr_refresh, "interval", minutes=HRRR_REFRESH_MINUTES, id="hrrr_refresh",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=240),
    max_instances=1, coalesce=True,
)


@app.on_event("startup")
def startup():
    scheduler.start()
    log.info("Phase 7 scheduler: tick 2min (5 products), polygonize 5min, midnight CT reset, 1440 sanity hourly, HRRR refresh 15min, archive retention 6 weeks")


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
        "version": "0.7.0",
        "phase": "7.0 — RadarScope-parity + playback + HRRR 6h forecast",
        "products_fetched": list(PRODUCTS.keys()),
        "storm_day_timezone": "America/Chicago",
        "archive_retention_days": ARCHIVE_RETENTION_DAYS,
        "consensus_scoring": {
            "mesh_in_band_points": 40,
            "posh_40_pct_points": 30,
            "reflectivity_50dbz_points": 20,
            "reflectivity_missing_half_credit": 10,
            "echo_top_30kft_points": 10,
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
            "GET /swaths — consensus-filtered polygons (today, live)",
            "GET /swaths/stats",
            "GET /swaths/playback?step_minutes=20 — time-lapse frames",
            "GET /forecast/hail — HRRR 1-6h hail forecast polygons",
            "GET /forecast/hail?forecast_hour=N — single forecast hour (1-6)",
            "GET /forecast/stats — HRRR pipeline status",
            "GET /history/dates — list of archived storm days",
            "GET /history/{YYYY-MM-DD} — archived swaths + ground-truth for one day",
            "POST /admin/force-tick",
            "POST /admin/polygonize",
            "POST /admin/sanity-check",
            "POST /admin/reset",
            "POST /admin/archive-now",
            "POST /admin/prune-archives",
            "POST /admin/refresh-forecast",
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
            last_unix = last_h * LAST_SEEN_BUCKET_SECONDS if last_h > 0 else 0
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


@app.get("/swaths/playback")
def get_swaths_playback(step_minutes: int = 20):
    """Time-lapse frames showing how today's swaths grew over the day.
    
    Each frame is a cumulative GeoJSON FeatureCollection — frame N contains all
    polygons formed by hail that fell at or before frame N's timestamp. The
    frontend renders frames in sequence to play back the storm day.
    
    step_minutes is rounded to the bucket size (10 min). Defaults to 20 → ~72
    frames/day for a smooth scrub. Try 10 for max detail, 30 for speed.
    """
    if step_minutes < 10:
        step_minutes = 10
    if step_minutes > 120:
        step_minutes = 120
    return polygonizer.build_playback(step_minutes=step_minutes)


@app.get("/forecast/hail")
def get_forecast_hail(forecast_hour: Optional[int] = None):
    """HRRR HAILCAST 1-6h hail forecast polygons.
    
    Optional `forecast_hour` param filters to a specific hour (1-6). Without it,
    returns all hours combined; the frontend filters via the forecastHour property.
    """
    geojson = hrrr_forecast.get_geojson()
    if forecast_hour is not None and 1 <= forecast_hour <= HRRR_FORECAST_HOURS:
        filtered = [f for f in geojson.get("features", [])
                    if f.get("properties", {}).get("forecastHour") == forecast_hour]
        return {
            "type": "FeatureCollection",
            "metadata": {**geojson.get("metadata", {}), "filtered_to_hour": forecast_hour},
            "features": filtered,
        }
    return geojson


@app.get("/forecast/stats")
def forecast_stats():
    return hrrr_forecast.stats()


@app.post("/admin/refresh-forecast")
def admin_refresh_forecast():
    """Manually trigger a HRRR refresh (useful for testing without waiting 15 min)."""
    hrrr_forecast.refresh()
    return hrrr_forecast.stats()


# ─────────────────────────────────────────────────────────────────────
# History endpoints — daily archives
# ─────────────────────────────────────────────────────────────────────

@app.get("/history/dates")
def history_dates():
    """List all archived storm-day dates (newest first).
    Frontend uses this to enable/disable dates in the calendar picker."""
    return {
        "dates": archiver.list_dates(),
        "retention_days": ARCHIVE_RETENTION_DAYS,
        "today": datetime.now(STORM_DAY_TZ).date().isoformat(),
        "timezone": "America/Chicago",
    }


@app.get("/history/{date_str}")
def history_for_date(date_str: str):
    """Return archived snapshot for one storm day.
    
    Response shape:
      {
        "date": "2026-04-25",
        "swaths": {<GeoJSON FeatureCollection of polygons>},
        "ground_truth": {<GeoJSON FeatureCollection of SPC reports>},
        "ground_truth_source": "archive" | "spc-fresh",
        "archived_at": "..."
      }
    
    Strategy: load the on-disk archive for swaths (those can never be
    re-fetched). For ground-truth, prefer SPC's fresh archive URL when
    available - SPC sometimes adds late-arriving reports overnight, so
    fresh-fetch can be more complete than what we archived at reset time.
    Fall back to archive if SPC fetch fails."""
    
    # Validate date format - prevents path traversal
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Refuse future dates
    today_ct = datetime.now(STORM_DAY_TZ).date()
    if d > today_ct:
        raise HTTPException(status_code=400, detail="Cannot request future dates")
    
    # Refuse dates older than retention
    cutoff = today_ct - timedelta(days=ARCHIVE_RETENTION_DAYS)
    if d < cutoff:
        raise HTTPException(status_code=410, detail=f"Archive retention is {ARCHIVE_RETENTION_DAYS} days. Older data is not stored.")
    
    archive = archiver.load_archive(date_str)
    if archive is None:
        # No archive on disk - return empty result with explanation
        return {
            "date": date_str,
            "swaths": {"type": "FeatureCollection", "features": [], "metadata": {"note": "No archive exists for this date"}},
            "ground_truth": archiver.fetch_spc_for_date(date_str),
            "ground_truth_source": "spc-fresh",
            "archived_at": None,
            "note": "No swath archive — pipeline may not have been running that day",
        }
    
    # Try to upgrade ground-truth with fresh SPC fetch (it often has late additions)
    fresh_gt = archiver.fetch_spc_for_date(date_str)
    if fresh_gt and len(fresh_gt.get("features", [])) > 0:
        ground_truth = fresh_gt
        gt_source = "spc-fresh"
    else:
        ground_truth = archive.get("ground_truth", {"type": "FeatureCollection", "features": []})
        gt_source = "archive"
    
    return {
        "date": archive["date"],
        "swaths": archive.get("swaths", {"type": "FeatureCollection", "features": []}),
        "ground_truth": ground_truth,
        "ground_truth_source": gt_source,
        "polygon_count": archive.get("polygon_count", 0),
        "ground_truth_count": len(ground_truth.get("features", [])),
        "archived_at": archive.get("archived_at"),
    }


@app.post("/admin/archive-now")
def admin_archive_now():
    """Manually trigger an archive of today's data without resetting.
    Useful for testing the archive flow without waiting for midnight CT."""
    archive_date_str = datetime.now(STORM_DAY_TZ).date().isoformat()
    return archiver.save_snapshot(archive_date_str)


@app.post("/admin/prune-archives")
def admin_prune_archives():
    """Manually trigger archive pruning."""
    deleted = archiver.prune_old()
    return {"deleted": deleted, "remaining": archiver.list_dates()}


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
