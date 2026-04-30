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
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from apscheduler.schedulers.background import BackgroundScheduler

from scipy import ndimage
from rasterio import features as rio_features
from rasterio.transform import Affine
from shapely.geometry import shape as shp_shape, mapping as shp_mapping

# ── Setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="StormDataPro MESH Processor", version="0.9.3")
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
HEATMAP_PNG_FILE = os.path.join(STATE_DIR, "heatmap.png")
HEATMAP_META_FILE = os.path.join(STATE_DIR, "heatmap_meta.json")

# Heatmap configuration — HailPoint-style nested topographic look.
# We generate a continuous color ramp from the MESH grid directly. Bands match
# HailPoint's published color scale starting at 1.0" (we drop the 0.75" halo
# because it visually swallowed small storm cells in tropical-air environments
# where MESH is noisy at the low end).
# Alpha values target ~55-75% so basemap labels and roads stay visible underneath
# — this is what makes the overlay readable when zoomed in on a town.
HEATMAP_COLOR_STOPS = [
    (1.00, 255, 220, 100, 140),    # yellow ~55% opaque
    (1.25, 255, 184, 60,  155),    # gold
    (1.50, 255, 140, 40,  170),    # light orange
    (1.75, 255, 100, 30,  180),    # orange
    (2.00, 230, 50,  30,  185),    # red-orange
    (2.25, 200, 25,  35,  190),    # red
    (2.50, 160, 15,  35,  195),    # dark red
    (2.75, 220, 60,  220, 195),    # magenta
    (3.00, 240, 130, 240, 200),    # bright pink
    (3.50, 230, 230, 230, 205),    # near-white
    (4.00, 255, 255, 255, 210),    # white core (still semi-transparent so map shows through)
]
HEATMAP_MIN_INCHES = 1.00   # below 1" = transparent. Quarter-size is the actual damage threshold.
HEATMAP_DOWNSAMPLE = 1      # full 1km resolution


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
# Legacy alias — preserved so /admin and stats endpoints don't 500.
# Real gating now uses CONSENSUS_MIN_SCORE_OUTER / CONSENSUS_MIN_SCORE_DEFAULT.
CONSENSUS_MIN_SCORE = 40           # legacy single-threshold value (use band-aware vars below)

# Pre-scoring spatial dilation. MRMS MESH undercounts swath WIDTH for fast cells —
# a hail core sweeps through a pixel for less than the 2-min MRMS scan window, so
# adjacent pixels see lower MESH even though hail actually fell there. RadarScope/
# HailTrace pad the core slightly. 1 = ~1 km dilation each side, 2 = ~2 km.
# Bumped to 2 on 2026-04-27 — 1 was still under-painting fast squall lines vs
# RadarScope's MESH contour overlay.
PRESCORE_DILATION_PIXELS = 2

# Two-tier consensus thresholds.
# RadarScope's MESH contour overlay just draws a contour at each MESH threshold
# with no other product gating. We can't fully match that without losing our false-
# positive defenses, but we can split the difference: trust MESH more for the
# OUTER halo (1.0-1.5") where it's the most reliable size estimate, and apply the
# full 4-product consensus only for the larger bands where false positives in the
# claims pipeline have higher business cost.
CONSENSUS_MIN_SCORE_OUTER = 25     # 1.0-1.5" band — MESH alone (40) qualifies
CONSENSUS_MIN_SCORE_DEFAULT = 40   # 1.5"+ bands — needs MESH + at least half-credit support

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

    def _generate_heatmap(self, crop_mesh: np.ndarray, gr0: int, gr1: int, gc0: int, gc1: int):
        """Generate a HailPoint-style topographic heatmap PNG from the MESH crop.

        crop_mesh is in tenths of millimeters (int16). We:
          1. Downsample (avg-pool) by HEATMAP_DOWNSAMPLE if >1 to reduce file size.
          2. Map each pixel through HEATMAP_COLOR_STOPS (linear interpolation between stops).
          3. Apply small Gaussian blur for the smooth topographic gradient look.
          4. Write PNG with transparent background everywhere MESH < HEATMAP_MIN_INCHES.

        The PNG's geographic bounds are saved separately to HEATMAP_META_FILE so the
        frontend knows where to overlay it.
        """
        try:
            from PIL import Image
        except ImportError:
            log.warning("Heatmap skipped: Pillow not installed")
            return

        try:
            # Crop dimensions
            crop_h, crop_w = crop_mesh.shape
            if crop_h == 0 or crop_w == 0:
                # No data in bbox — write empty meta and skip image
                self._save_heatmap_meta(None, None)
                return

            # Step 1: optionally downsample by averaging blocks. We use peak (max), not mean,
            # because hail size is a max-of-a-region quantity — averaging would smooth away cores.
            ds = max(1, int(HEATMAP_DOWNSAMPLE))
            if ds > 1:
                # Trim so dimensions are divisible by ds
                use_h = (crop_h // ds) * ds
                use_w = (crop_w // ds) * ds
                if use_h == 0 or use_w == 0:
                    self._save_heatmap_meta(None, None)
                    return
                trimmed = crop_mesh[:use_h, :use_w]
                # Reshape and take max over each ds×ds block
                pooled = trimmed.reshape(use_h // ds, ds, use_w // ds, ds).max(axis=(1, 3))
                values_x10 = pooled.astype(np.float32)
                out_h, out_w = pooled.shape
            else:
                values_x10 = crop_mesh.astype(np.float32)
                out_h, out_w = crop_h, crop_w

            # Step 2: convert tenths-of-mm to inches
            values_inches = values_x10 / SCALE / 25.4

            # Step 3: build RGBA image array (out_h, out_w, 4) uint8
            # Default fully transparent
            rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)

            # Build lookup arrays from HEATMAP_COLOR_STOPS
            stops = np.array([s[0] for s in HEATMAP_COLOR_STOPS], dtype=np.float32)  # sizes
            r_stops = np.array([s[1] for s in HEATMAP_COLOR_STOPS], dtype=np.float32)
            g_stops = np.array([s[2] for s in HEATMAP_COLOR_STOPS], dtype=np.float32)
            b_stops = np.array([s[3] for s in HEATMAP_COLOR_STOPS], dtype=np.float32)
            a_stops = np.array([s[4] for s in HEATMAP_COLOR_STOPS], dtype=np.float32)

            # Mask of pixels above threshold
            visible_mask = values_inches >= HEATMAP_MIN_INCHES
            if not visible_mask.any():
                # No visible hail — empty PNG (transparent everywhere)
                img = Image.fromarray(rgba, mode="RGBA")
                tmp = HEATMAP_PNG_FILE + ".tmp"
                img.save(tmp, "PNG", optimize=True)
                os.replace(tmp, HEATMAP_PNG_FILE)
                self._save_heatmap_meta(None, None)
                return

            # Vectorized linear interpolation: for each visible pixel, find which stop
            # interval it falls into and interpolate the RGBA components.
            v = values_inches[visible_mask]
            # np.interp is what we need — works component-by-component
            r = np.interp(v, stops, r_stops).astype(np.uint8)
            g = np.interp(v, stops, g_stops).astype(np.uint8)
            b = np.interp(v, stops, b_stops).astype(np.uint8)
            a = np.interp(v, stops, a_stops).astype(np.uint8)

            rgba[..., 0][visible_mask] = r
            rgba[..., 1][visible_mask] = g
            rgba[..., 2][visible_mask] = b
            rgba[..., 3][visible_mask] = a

            # NO blur — HailPoint shows crisp nested band contours, not soft glow.
            # The earlier blur made small storm cells look like smudges. The natural
            # MESH gradient (continuous values) provides the topographic feel without
            # any post-processing. Every pixel is its true color.

            img = Image.fromarray(rgba, mode="RGBA")
            tmp = HEATMAP_PNG_FILE + ".tmp"
            img.save(tmp, "PNG", optimize=True)
            os.replace(tmp, HEATMAP_PNG_FILE)

            # Step 5: compute geographic bounds of the image. Use the original bbox
            # (gr0..gr1, gc0..gc1) — the downsample block-trim might shrink slightly
            # but we use the trimmed dimensions to compute exact bounds.
            actual_h_pixels = out_h * ds
            actual_w_pixels = out_w * ds
            lat_north = GRID_LAT1 - gr0 * GRID_DEG
            lat_south = GRID_LAT1 - (gr0 + actual_h_pixels) * GRID_DEG
            lon_west = (GRID_LON1_360 - 360) + gc0 * GRID_DEG
            lon_east = (GRID_LON1_360 - 360) + (gc0 + actual_w_pixels) * GRID_DEG

            self._save_heatmap_meta(
                bounds=[lon_west, lat_south, lon_east, lat_north],
                size=[out_w, out_h],
            )
            log.info(f"Heatmap: {out_w}×{out_h}px PNG, bounds W={lon_west:.2f} S={lat_south:.2f} E={lon_east:.2f} N={lat_north:.2f}")

        except Exception as e:
            log.error(f"Heatmap generation failed: {e}", exc_info=True)

    def _save_heatmap_meta(self, bounds: Optional[list], size: Optional[list]):
        """Persist heatmap geographic bounds + image dimensions for the frontend."""
        try:
            meta = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bounds": bounds,   # [west, south, east, north] or None
                "size": size,        # [width, height] in pixels or None
                "color_stops": [
                    {"size_inches": s[0], "rgba": [s[1], s[2], s[3], s[4]]}
                    for s in HEATMAP_COLOR_STOPS
                ],
                "min_inches": HEATMAP_MIN_INCHES,
                "downsample": HEATMAP_DOWNSAMPLE,
            }
            tmp = HEATMAP_META_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f)
            os.replace(tmp, HEATMAP_META_FILE)
        except Exception as e:
            log.warning(f"Heatmap meta save failed: {e}")

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

                # Two-tier consensus gate: lower bar for the outer 1.0-1.5" halo
                # (which RadarScope draws aggressively), strict bar for larger sizes.
                band_min_score = CONSENSUS_MIN_SCORE_OUTER if band["min_inches"] < 1.5 else CONSENSUS_MIN_SCORE_DEFAULT
                pass_mask = crop_score >= band_min_score
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
                    # Score baseline is the band-aware minimum, so feature properties
                    # don't claim a higher confidence than the gate that admitted it.
                    band_min_score = CONSENSUS_MIN_SCORE_OUTER if band["min_inches"] < 1.5 else CONSENSUS_MIN_SCORE_DEFAULT
                    peak_score = band_min_score
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

                    # Patch B: peak MESH anywhere within the polygon's bounding box
                    # (ignoring band constraint). Useful for the popup — even on a
                    # 1-1.5" donut polygon, this surfaces the actual max hail observed
                    # in or near the polygon, which is what the user usually cares about.
                    peak_nearby_x10 = peak_x10
                    if sub_c1 > sub_c0 and sub_r0 > sub_r1:
                        sub_mesh_full = crop_mesh[sub_r1:sub_r0, sub_c0:sub_c1]
                        if sub_mesh_full.size > 0:
                            peak_nearby_x10 = int(max(peak_x10, sub_mesh_full.max()))

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
                            # Patch B: peak MESH anywhere in/near this polygon (no band constraint).
                            # Often >= peakSizeInches; for the outer 1-1.5" halo polygon,
                            # peakNearbyInches will reveal the larger core's true size.
                            "peakNearbyMM": round(peak_nearby_x10 / SCALE, 1),
                            "peakNearbyInches": round(peak_nearby_x10 / SCALE / 25.4, 2),
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

            # Free the secondary crops now (we still need crop_mesh for heatmap)
            crop_posh = crop_refl = crop_echo = crop_last_seen = None
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
                    "consensus_min_score_outer": CONSENSUS_MIN_SCORE_OUTER,
                    "consensus_min_score_default": CONSENSUS_MIN_SCORE_DEFAULT,
                    "scoring": {
                        "mesh_in_band_points": 40,
                        "posh_full_points": 30,
                        "posh_missing_half_credit": 15,
                        "reflectivity_full_points": 20,
                        "reflectivity_missing_half_credit": 10,
                        "echo_top_points": 10,
                        "prescore_dilation_pixels": PRESCORE_DILATION_PIXELS,
                        "smooth_radius_deg": SMOOTH_RADIUS_DEG,
                        "outer_band_threshold": CONSENSUS_MIN_SCORE_OUTER,
                        "default_band_threshold": CONSENSUS_MIN_SCORE_DEFAULT,
                        "outer_band_definition": "min_inches < 1.5",
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

            # Generate the HailPoint-style topographic heatmap PNG from the same crop.
            # crop_mesh was deliberately kept alive after the polygon loop for this step.
            # This is a separate visual product — failure here doesn't break polygons.
            try:
                self._generate_heatmap(crop_mesh, gr0, gr1, gc0, gc1)
            except Exception as he:
                log.error(f"Heatmap generation failed (non-fatal): {he}", exc_info=True)
            finally:
                # Now free crop_mesh
                crop_mesh = None
                gc.collect()

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

                    band_min_score = CONSENSUS_MIN_SCORE_OUTER if band["min_inches"] < 1.5 else CONSENSUS_MIN_SCORE_DEFAULT
                    pass_mask = crop_score >= band_min_score
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

            # Snapshot heatmap PNG + meta so archive dates show their own heatmap
            heatmap_archived = False
            try:
                import shutil
                png_dest  = os.path.join(ARCHIVE_DIR, f"{archive_date_str}_heatmap.png")
                meta_dest = os.path.join(ARCHIVE_DIR, f"{archive_date_str}_heatmap_meta.json")
                if os.path.exists(HEATMAP_PNG_FILE):
                    shutil.copy2(HEATMAP_PNG_FILE, png_dest)
                if os.path.exists(HEATMAP_META_FILE):
                    shutil.copy2(HEATMAP_META_FILE, meta_dest)
                heatmap_archived = True
                # Back up to Supabase so a fresh Railway deploy can restore them
                backup_heatmap_to_supabase(archive_date_str, png_dest, meta_dest)
            except Exception as he:
                log.warning(f"Heatmap snapshot failed (non-fatal): {he}")

            log.info(f"Archived {archive_date_str}: {poly_count} polygons, {gt_count} reports, heatmap={heatmap_archived} → {path}")
            return {"date": archive_date_str, "polygons": poly_count, "reports": gt_count, "path": path, "heatmap": heatmap_archived}
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

    def compute_summary(self, date_str: str) -> Optional[dict]:
        """Quick stats for one archive: hit count, peak size, total reports.
        
        Returns: {"date": "...", "swath_count": N, "peak_inches": F,
                  "ground_truth_count": M, "max_band": "..."} or None if no archive.
        Used by the calendar picker to color-code each day by intensity.
        """
        archive = self.load_archive(date_str)
        if archive is None:
            return None
        try:
            swaths = archive.get("swaths", {}).get("features", []) or []
            ground = archive.get("ground_truth", {}).get("features", []) or []
            # Walk swaths once to find peak and count
            peak_inches = 0.0
            for f in swaths:
                p = (f.get("properties") or {})
                # Prefer peakNearbyInches (absolute peak in polygon footprint),
                # fall back to peakSizeInches (peak within the band's range).
                v = p.get("peakNearbyInches") or p.get("peakSizeInches") or 0
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = 0
                if v > peak_inches:
                    peak_inches = v
            return {
                "date": date_str,
                "swath_count": len(swaths),
                "peak_inches": round(peak_inches, 2),
                "ground_truth_count": len(ground),
            }
        except Exception as e:
            log.warning(f"Failed to compute summary for {date_str}: {e}")
            return None

    def all_summaries(self) -> list:
        """Compute summaries for every archived date. Cached on disk and refreshed
        only when the cached set is missing dates. Most days the cache is hit and
        we return without touching disk."""
        try:
            dates = self.list_dates()
            cached = {}
            cache_path = os.path.join(STATE_DIR, "archive_summaries.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path) as f:
                        for entry in json.load(f).get("summaries", []):
                            cached[entry["date"]] = entry
                except Exception:
                    cached = {}
            # For each date NOT in the cache (or for today/yesterday which may have
            # been re-archived), compute fresh
            today_ct = datetime.now(STORM_DAY_TZ).date().isoformat()
            yesterday_ct = (datetime.now(STORM_DAY_TZ).date() - timedelta(days=1)).isoformat()
            volatile = {today_ct, yesterday_ct}
            updated = False
            for d in dates:
                if d not in cached or d in volatile:
                    summary = self.compute_summary(d)
                    if summary:
                        cached[d] = summary
                        updated = True
            # Drop cached entries for dates that no longer have archives (got pruned)
            for d in list(cached.keys()):
                if d not in dates:
                    del cached[d]
                    updated = True
            if updated:
                try:
                    tmp = cache_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"summaries": list(cached.values())}, f)
                    os.replace(tmp, cache_path)
                except Exception as e:
                    log.warning(f"Summary cache write failed: {e}")
            # Return sorted newest first
            return sorted(cached.values(), key=lambda x: x["date"], reverse=True)
        except Exception as e:
            log.error(f"all_summaries failed: {e}", exc_info=True)
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


# ─────────────────────────────────────────────────────────────────────
# Daily Hit-Town Report — Phase 8
# ─────────────────────────────────────────────────────────────────────
# Generates a private HTML report once per day showing every US town
# (≥5,000 population) that was hit with ≥1.0" hail and ≥5 sq mi of coverage.
# Data sources, all free:
#   - US Census ACS 5-year API: population, demographics, median income
#     (https://api.census.gov/data/2023/acs/acs5)
#   - US Census Gazetteer "Places" file: town centroids + FIPS codes
#     (https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/)
#   - OpenStreetMap Overpass API: Tractor Supply locations
#     (https://overpass-api.de/api/interpreter)
#
# Caching policy:
#   - Gazetteer downloaded once at first generation, kept in /data/gazetteer.tsv
#   - TSC locations refreshed weekly to /data/tsc_locations.json
#   - Per-town demographics cached 1 year in /data/town_demographics.json
# Report URL: /report/{REPORT_URL_TOKEN}/YYYY-MM-DD.html  + index page
# Token is generated once per deploy in env var REPORT_URL_TOKEN; if not set,
# we generate one and persist it in /data/report_token.txt.

REPORT_DIR = os.path.join(STATE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
GAZETTEER_FILE = os.path.join(STATE_DIR, "gazetteer_places.tsv")
TSC_LOCATIONS_FILE = os.path.join(STATE_DIR, "tsc_locations.json")
TOWN_DEMOGRAPHICS_FILE = os.path.join(STATE_DIR, "town_demographics.json")
REPORT_TOKEN_FILE = os.path.join(STATE_DIR, "report_token.txt")

# Gazetteer URL — Census publishes annually. The 2024 file is current as of
# this deploy. URL format is stable; only year changes annually.
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip"
ACS_BASE_URL = "https://api.census.gov/data/2023/acs/acs5"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Hit-criteria
# Population threshold lowered from 5000 to 2500 on 2026-04-27 — small towns
# in the 2.5k-5k range (e.g. Warsaw NY village ~3,600 pop) are real opportunities
# that the higher threshold was missing.
# Coverage threshold lowered from 5.0 to 1.0 sq mi — a 2,500-pop town's TOTAL
# land area can be 4-5 sq mi (Warsaw village = 4.1 sq mi), so 5 sq mi was
# mathematically excluding small towns even on a perfect direct hit.
REPORT_MIN_HAIL_INCHES = 1.0
REPORT_MIN_TOWN_POP = 2500
REPORT_MIN_COVERAGE_SQMI = 1.0

# Census ACS variable codes — these are stable across years
ACS_VARS = {
    "B01003_001E": "total_pop",
    "B19013_001E": "median_household_income",
    "B02001_002E": "race_white_alone",
    "B02001_003E": "race_black_alone",
    "B02001_004E": "race_aian_alone",         # American Indian / Alaska Native
    "B02001_005E": "race_asian_alone",
    "B02001_006E": "race_nhpi_alone",         # Native Hawaiian / Pacific Islander
    "B02001_007E": "race_other_alone",
    "B02001_008E": "race_two_or_more",
    "B03003_003E": "hispanic_or_latino",      # of any race
}


class DailyReport:
    """Generates daily town-impact reports.
    
    Pipeline:
      1. Load (or download) the Census places gazetteer
      2. Load (or refresh) the Tractor Supply locations
      3. For a given archive date, find polygons with peak ≥1.0"
      4. Find every gazetteer place with centroid inside any polygon
         AND polygon-coverage-inside-place-bbox ≥ 5 sq mi
      5. For each candidate town: pull/cache ACS demographics
      6. For each candidate town: compute distance to nearest TSC
      7. Render HTML report → /data/reports/YYYY-MM-DD.html
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.gazetteer: Optional[list] = None  # list of dicts
        self.tsc_locations: Optional[list] = None  # list of (lat, lon, name) tuples
        self.demographics_cache: dict = {}     # geoid -> {data, cached_at}
        self.token = self._load_or_create_token()
        self._load_demographics_cache()

    def _load_or_create_token(self) -> str:
        """Stable obscure URL token — kept across deploys via /data."""
        env_token = os.environ.get("REPORT_URL_TOKEN")
        if env_token:
            return env_token
        if os.path.exists(REPORT_TOKEN_FILE):
            with open(REPORT_TOKEN_FILE, "r") as f:
                t = f.read().strip()
                if t:
                    return t
        # Generate ~13-char base32 token (78 bits of entropy)
        import secrets
        import base64
        t = base64.b32encode(secrets.token_bytes(10)).decode().rstrip("=").lower()
        with open(REPORT_TOKEN_FILE, "w") as f:
            f.write(t)
        log.info(f"Generated report URL token: {t} (saved to {REPORT_TOKEN_FILE})")
        return t

    def _load_demographics_cache(self):
        if os.path.exists(TOWN_DEMOGRAPHICS_FILE):
            try:
                with open(TOWN_DEMOGRAPHICS_FILE, "r") as f:
                    self.demographics_cache = json.load(f)
                log.info(f"Loaded {len(self.demographics_cache)} cached town demographics")
            except Exception as e:
                log.warning(f"Demographics cache load failed: {e}")
                self.demographics_cache = {}

    def _save_demographics_cache(self):
        try:
            tmp = TOWN_DEMOGRAPHICS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.demographics_cache, f)
            os.replace(tmp, TOWN_DEMOGRAPHICS_FILE)
        except Exception as e:
            log.warning(f"Demographics cache save failed: {e}")

    def ensure_gazetteer(self) -> bool:
        """Download + parse Census places gazetteer if not already loaded.
        TSV columns: USPS GEOID ANSICODE NAME LSAD FUNCSTAT ALAND AWATER ALAND_SQMI AWATER_SQMI INTPTLAT INTPTLONG
        We only need GEOID (state+place fips), NAME, INTPTLAT, INTPTLONG, ALAND_SQMI.
        """
        if self.gazetteer is not None:
            return True
        if not os.path.exists(GAZETTEER_FILE):
            try:
                log.info(f"Downloading Census gazetteer from {GAZETTEER_URL}")
                r = requests.get(GAZETTEER_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
                if r.status_code != 200:
                    log.error(f"Gazetteer download HTTP {r.status_code}")
                    return False
                # The download is a zip file containing one .txt
                import zipfile
                import io
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    txt_names = [n for n in z.namelist() if n.endswith(".txt")]
                    if not txt_names:
                        log.error("Gazetteer zip has no .txt member")
                        return False
                    with z.open(txt_names[0]) as src, open(GAZETTEER_FILE, "wb") as dst:
                        dst.write(src.read())
                log.info(f"Gazetteer saved to {GAZETTEER_FILE}")
            except Exception as e:
                log.error(f"Gazetteer download failed: {e}")
                return False

        # Parse — the file is tab-separated with a header row
        try:
            places = []
            with open(GAZETTEER_FILE, "r", encoding="latin-1") as f:
                header = f.readline().strip().split("\t")
                # Find columns of interest
                idx_usps = header.index("USPS")
                idx_geoid = header.index("GEOID")
                idx_name = header.index("NAME")
                idx_aland_sqmi = header.index("ALAND_SQMI")
                idx_lat = header.index("INTPTLAT")
                idx_lon = header.index("INTPTLONG")
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < len(header):
                        continue
                    try:
                        places.append({
                            "state": parts[idx_usps],
                            "geoid": parts[idx_geoid],
                            "name": parts[idx_name],
                            "land_sqmi": float(parts[idx_aland_sqmi]),
                            "lat": float(parts[idx_lat]),
                            "lon": float(parts[idx_lon]),
                        })
                    except (ValueError, IndexError):
                        continue
            self.gazetteer = places
            log.info(f"Loaded {len(places)} places from gazetteer")
            return True
        except Exception as e:
            log.error(f"Gazetteer parse failed: {e}", exc_info=True)
            return False

    def ensure_tsc_locations(self) -> bool:
        """Load Tractor Supply locations from cache or fetch from OSM Overpass.
        Refreshes if file is older than 14 days."""
        if self.tsc_locations is not None:
            return True
        # Check cache freshness
        fresh = False
        if os.path.exists(TSC_LOCATIONS_FILE):
            try:
                mtime = os.path.getmtime(TSC_LOCATIONS_FILE)
                age_days = (time.time() - mtime) / 86400
                if age_days < 14:
                    fresh = True
                    with open(TSC_LOCATIONS_FILE, "r") as f:
                        data = json.load(f)
                    self.tsc_locations = [(d["lat"], d["lon"], d.get("name", "Tractor Supply")) for d in data]
                    log.info(f"Loaded {len(self.tsc_locations)} TSC locations (cache age {age_days:.1f}d)")
                    return True
            except Exception as e:
                log.warning(f"TSC cache load failed: {e}")

        if not fresh:
            return self._refresh_tsc_locations()
        return False

    def _refresh_tsc_locations(self) -> bool:
        """Fetch all Tractor Supply locations in CONUS from OSM via Overpass."""
        # Overpass QL query: nodes/ways tagged as Tractor Supply within US bounding box.
        # OSM tags vary — TSC stores show up under shop=farm, brand=Tractor Supply,
        # operator=Tractor Supply, or name~"Tractor Supply". Cast a wide net.
        query = """
        [out:json][timeout:90];
        (
          node["brand"~"Tractor Supply",i](24,-125,50,-66);
          way["brand"~"Tractor Supply",i](24,-125,50,-66);
          node["name"~"Tractor Supply",i](24,-125,50,-66);
          way["name"~"Tractor Supply",i](24,-125,50,-66);
        );
        out center;
        """
        try:
            log.info("Refreshing TSC locations from OSM Overpass…")
            r = requests.post(OVERPASS_URL, data={"data": query},
                              headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code != 200:
                log.error(f"Overpass HTTP {r.status_code}")
                return False
            elements = r.json().get("elements", [])
            results = []
            seen_coords = set()
            for el in elements:
                if el.get("type") == "node":
                    lat, lon = el.get("lat"), el.get("lon")
                elif el.get("type") == "way" and "center" in el:
                    lat, lon = el["center"]["lat"], el["center"]["lon"]
                else:
                    continue
                if lat is None or lon is None:
                    continue
                # Dedupe by ~100m grid bucket (OSM sometimes has overlapping nodes/ways for same store)
                key = (round(lat, 3), round(lon, 3))
                if key in seen_coords:
                    continue
                seen_coords.add(key)
                tags = el.get("tags", {})
                name = tags.get("name", "Tractor Supply")
                results.append({"lat": lat, "lon": lon, "name": name})

            with open(TSC_LOCATIONS_FILE, "w") as f:
                json.dump(results, f)
            self.tsc_locations = [(r["lat"], r["lon"], r["name"]) for r in results]
            log.info(f"Saved {len(results)} TSC locations to {TSC_LOCATIONS_FILE}")
            return True
        except Exception as e:
            log.error(f"TSC refresh failed: {e}", exc_info=True)
            return False

    def _haversine_miles(self, lat1, lon1, lat2, lon2) -> float:
        """Great-circle distance in miles."""
        R = 3958.8  # Earth radius in miles
        rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def _nearest_tsc(self, lat: float, lon: float) -> tuple:
        """Returns (distance_miles, name, lat, lon) of nearest TSC, or (None, None, None, None)."""
        if not self.tsc_locations:
            return (None, None, None, None)
        best_d = float("inf")
        best = None
        for s_lat, s_lon, s_name in self.tsc_locations:
            d = self._haversine_miles(lat, lon, s_lat, s_lon)
            if d < best_d:
                best_d = d
                best = (s_lat, s_lon, s_name)
        if best is None:
            return (None, None, None, None)
        return (round(best_d, 1), best[2], best[0], best[1])

    def _fetch_demographics(self, geoid: str) -> Optional[dict]:
        """Fetch ACS data for a single place. geoid is 7-digit state(2)+place(5)."""
        # Cache hit?
        cached = self.demographics_cache.get(geoid)
        if cached:
            cached_at = cached.get("cached_at", 0)
            if (time.time() - cached_at) < (365 * 86400):
                return cached.get("data")

        # ACS API: place lookups need state and place separately
        if len(geoid) < 7:
            return None
        state_fips = geoid[:2]
        place_fips = geoid[2:]
        try:
            var_list = ",".join(ACS_VARS.keys())
            url = f"{ACS_BASE_URL}?get=NAME,{var_list}&for=place:{place_fips}&in=state:{state_fips}"
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code != 200:
                log.warning(f"ACS HTTP {r.status_code} for geoid {geoid}")
                return None
            arr = r.json()
            if len(arr) < 2:
                return None
            header_row = arr[0]
            data_row = arr[1]
            # Build dict: {acs_var: int_or_none}
            data = {"name_with_state": data_row[0]}
            for var, friendly in ACS_VARS.items():
                idx = header_row.index(var)
                v = data_row[idx]
                try:
                    data[friendly] = int(v) if v not in (None, "", "null") else None
                except (ValueError, TypeError):
                    data[friendly] = None
            self.demographics_cache[geoid] = {
                "cached_at": time.time(),
                "data": data,
            }
            return data
        except Exception as e:
            log.warning(f"ACS fetch failed for geoid {geoid}: {e}")
            return None

    @staticmethod
    def _point_in_polygon(lat: float, lon: float, polygon_geom: dict) -> bool:
        """Point-in-polygon test for a GeoJSON Polygon or MultiPolygon."""
        try:
            from shapely.geometry import shape, Point
            poly = shape(polygon_geom)
            return poly.contains(Point(lon, lat))
        except Exception:
            return False

    def _polygon_coverage_within_box(self, polygon_geom: dict, center_lat: float, center_lon: float,
                                     box_radius_miles: float = 8.0) -> float:
        """Compute the area of polygon intersected with a small box around the town centroid.
        Used to enforce the 'min coverage area inside town' criterion. Box radius defaults to
        8 miles (~roughly the size of a small to mid-size city footprint)."""
        try:
            from shapely.geometry import shape, box
            # 1 deg latitude ≈ 69 miles. 1 deg longitude ≈ 69 * cos(lat) miles
            dlat = box_radius_miles / 69.0
            dlon = box_radius_miles / (69.0 * max(0.1, math.cos(math.radians(center_lat))))
            town_box = box(center_lon - dlon, center_lat - dlat,
                          center_lon + dlon, center_lat + dlat)
            poly = shape(polygon_geom)
            inter = poly.intersection(town_box)
            if inter.is_empty:
                return 0.0
            # Convert area in deg² to sq miles. Use centroid latitude for the longitude term.
            # Area_deg² × (69 miles)² × cos(lat) ≈ Area_sqmi
            sqmi = inter.area * (69.0 * 69.0) * math.cos(math.radians(center_lat))
            return max(0.0, sqmi)
        except Exception:
            return 0.0

    def generate(self, archive_date_str: str, archive_data: Optional[dict] = None) -> dict:
        """Generate the report for a given date. archive_data (if passed) avoids
        re-reading from disk. Returns metadata including saved file path.
        
        The report has TWO sections:
          1. Swath-hit towns — found by polygon coverage; augmented with any
             ground reports inside their bbox
          2. Ground-only towns — towns with a spotter report ≥1.0" but no
             qualifying swath polygon (catches MRMS misses, radar gaps)
        """
        log.info(f"DailyReport.generate({archive_date_str}) starting")
        t0 = datetime.now(timezone.utc)
        try:
            # Step 1: Get the archived polygons + ground reports for this date
            if archive_data is None:
                archive_path = os.path.join(ARCHIVE_DIR, f"{archive_date_str}.json")
                if not os.path.exists(archive_path):
                    return {"error": f"No archive for {archive_date_str}"}
                with open(archive_path, "r") as f:
                    archive_data = json.load(f)

            features = archive_data.get("swaths", {}).get("features", [])
            ground_reports = archive_data.get("ground_truth", {}).get("features", [])

            # Filter swath features to those with peak ≥ REPORT_MIN_HAIL_INCHES
            qualifying = [
                f for f in features
                if (f.get("properties", {}).get("peakSizeInches") or 0) >= REPORT_MIN_HAIL_INCHES
                or (f.get("properties", {}).get("bandMinInches") or 0) >= REPORT_MIN_HAIL_INCHES
            ]
            # Filter ground reports to ≥ REPORT_MIN_HAIL_INCHES
            qualifying_ground = [
                gr for gr in ground_reports
                if (gr.get("properties", {}).get("sizeInches") or 0) >= REPORT_MIN_HAIL_INCHES
            ]

            log.info(f"Filtered: {len(qualifying)} swath features, {len(qualifying_ground)} ground reports")

            if not qualifying and not qualifying_ground:
                return self._render_and_save(archive_date_str, swath_hits=[], ground_only_hits=[],
                                             total_features=len(features), total_ground=len(ground_reports))

            # Step 2: Make sure dependencies are loaded
            if not self.ensure_gazetteer():
                return {"error": "Could not load Census gazetteer"}
            if not self.ensure_tsc_locations():
                log.warning("TSC locations unavailable; report will skip TSC distance")

            # Step 3: Spatial join — find places whose centroid is in any polygon
            #         AND that have ≥REPORT_MIN_COVERAGE_SQMI of polygon inside their bbox.
            from shapely.geometry import shape, Point
            poly_with_bbox = []
            for f in qualifying:
                try:
                    geom = shape(f["geometry"])
                    minx, miny, maxx, maxy = geom.bounds
                    poly_with_bbox.append((f, geom, (minx, miny, maxx, maxy)))
                except Exception:
                    continue

            swath_hits = []  # towns hit by swaths
            swath_hit_geoids = set()  # for dedup against ground-only
            for place in self.gazetteer:
                lat, lon = place["lat"], place["lon"]
                # Quick skip: if no polygon's bbox contains this point, no hit possible
                candidate_polys = [
                    (f, g) for (f, g, (minx, miny, maxx, maxy)) in poly_with_bbox
                    if minx <= lon <= maxx and miny <= lat <= maxy
                ]
                if not candidate_polys:
                    continue

                hitting_features = []
                max_peak = 0.0
                total_coverage = 0.0
                for f, geom in candidate_polys:
                    try:
                        if geom.contains(Point(lon, lat)) or geom.touches(Point(lon, lat)):
                            cov = self._polygon_coverage_within_box(f["geometry"], lat, lon)
                            total_coverage += cov
                            peak = f.get("properties", {}).get("peakSizeInches") or 0
                            if peak > max_peak:
                                max_peak = peak
                            hitting_features.append({"feature": f, "coverage_sqmi": cov})
                    except Exception:
                        continue

                if not hitting_features or total_coverage < REPORT_MIN_COVERAGE_SQMI:
                    continue

                # Step 3b: For this swath-hit town, also count ground reports inside the box
                ground_in_town = self._ground_reports_in_box(qualifying_ground, lat, lon)

                swath_hits.append({
                    "place": place,
                    "max_peak_inches": max_peak,
                    "total_coverage_sqmi": round(total_coverage, 1),
                    "polygon_count": len(hitting_features),
                    "ground_reports": ground_in_town,
                    "ground_max_inches": max((gr["properties"].get("sizeInches") or 0)
                                             for gr in ground_in_town) if ground_in_town else 0,
                })
                swath_hit_geoids.add(place["geoid"])

            log.info(f"Swath spatial join: {len(swath_hits)} candidate towns hit (before pop filter)")

            # Step 3c: Find GROUND-ONLY hits — ground reports near towns NOT covered by a swath.
            # For each ground report, find the closest gazetteer place within 5 miles.
            # If that place isn't already swath-hit, it's a ground-only candidate.
            ground_only_candidates = {}  # geoid -> {place, ground_reports[]}
            GROUND_NEAREST_TOWN_RADIUS_MI = 5.0
            for gr in qualifying_ground:
                try:
                    gr_lon, gr_lat = gr["geometry"]["coordinates"]
                except (KeyError, ValueError, TypeError):
                    continue
                # Find nearest place (cheap brute-force; gazetteer is ~30k entries
                # but we can short-circuit on bbox)
                best = None
                best_d = GROUND_NEAREST_TOWN_RADIUS_MI
                for place in self.gazetteer:
                    # Quick reject: skip if too far on lat alone (1 deg ≈ 69 mi)
                    if abs(place["lat"] - gr_lat) > (GROUND_NEAREST_TOWN_RADIUS_MI / 69.0):
                        continue
                    d = self._haversine_miles(gr_lat, gr_lon, place["lat"], place["lon"])
                    if d < best_d:
                        best_d = d
                        best = place
                if not best:
                    continue
                if best["geoid"] in swath_hit_geoids:
                    continue  # already counted in swath_hits, augmented there
                # Add to ground-only candidates
                gid = best["geoid"]
                if gid not in ground_only_candidates:
                    ground_only_candidates[gid] = {
                        "place": best,
                        "ground_reports": [],
                    }
                ground_only_candidates[gid]["ground_reports"].append(gr)

            log.info(f"Ground-only candidates: {len(ground_only_candidates)} towns with reports but no qualifying swath")

            # Step 4: Filter both lists by population (Census ACS lookup)
            enriched_swath = self._enrich_with_demographics(swath_hits)
            ground_only_list = [
                {
                    "place": v["place"],
                    "ground_reports": v["ground_reports"],
                    "ground_max_inches": max((gr["properties"].get("sizeInches") or 0)
                                             for gr in v["ground_reports"]),
                }
                for v in ground_only_candidates.values()
            ]
            enriched_ground = self._enrich_with_demographics(ground_only_list)

            # Save updated demographics cache
            self._save_demographics_cache()

            log.info(f"After population filter: {len(enriched_swath)} swath towns, {len(enriched_ground)} ground-only towns")

            # Sort: swath hits by hail size desc, ground-only by ground size desc
            enriched_swath.sort(
                key=lambda h: (-h["max_peak_inches"],
                               -(h["demographics"].get("total_pop") or 0))
            )
            enriched_ground.sort(
                key=lambda h: (-h["ground_max_inches"],
                               -(h["demographics"].get("total_pop") or 0))
            )

            return self._render_and_save(
                archive_date_str,
                swath_hits=enriched_swath,
                ground_only_hits=enriched_ground,
                total_features=len(features),
                total_ground=len(ground_reports),
            )

        except Exception as e:
            log.error(f"DailyReport.generate failed: {e}", exc_info=True)
            return {"error": str(e), "elapsed_ms": int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)}

    def _ground_reports_in_box(self, reports: list, center_lat: float, center_lon: float,
                               box_radius_miles: float = 8.0) -> list:
        """Return ground reports whose Point falls inside an N-mile box around centroid.
        Uses the same 8-mile bbox as the swath coverage check for consistency."""
        dlat = box_radius_miles / 69.0
        dlon = box_radius_miles / (69.0 * max(0.1, math.cos(math.radians(center_lat))))
        in_box = []
        for gr in reports:
            try:
                gr_lon, gr_lat = gr["geometry"]["coordinates"]
            except (KeyError, ValueError, TypeError):
                continue
            if (center_lat - dlat) <= gr_lat <= (center_lat + dlat) and \
               (center_lon - dlon) <= gr_lon <= (center_lon + dlon):
                in_box.append(gr)
        return in_box

    def _enrich_with_demographics(self, hits: list) -> list:
        """For each hit, fetch ACS demographics and add TSC distance.
        Drops any town that fails the population filter."""
        enriched = []
        for hit in hits:
            geoid = hit["place"]["geoid"]
            demo = self._fetch_demographics(geoid)
            if not demo:
                continue
            pop = demo.get("total_pop") or 0
            if pop < REPORT_MIN_TOWN_POP:
                continue
            tsc_dist, tsc_name, tsc_lat, tsc_lon = self._nearest_tsc(
                hit["place"]["lat"], hit["place"]["lon"]
            )
            enriched.append({
                **hit,
                "demographics": demo,
                "tsc_distance_miles": tsc_dist,
                "tsc_name": tsc_name,
            })
        return enriched

    def _render_and_save(self, archive_date_str: str, swath_hits: list, ground_only_hits: list,
                        total_features: int, total_ground: int) -> dict:
        """Render the HTML and save to /data/reports/YYYY-MM-DD.html."""
        html = self._render_html(archive_date_str, swath_hits, ground_only_hits, total_features, total_ground)
        path = os.path.join(REPORT_DIR, f"{archive_date_str}.html")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(html)
            os.replace(tmp, path)

            # Update index page
            self._update_index()

            log.info(f"Report saved: {path} ({len(swath_hits)} swath hits, {len(ground_only_hits)} ground-only, {len(html)} bytes)")
            return {
                "date": archive_date_str,
                "swath_hits": len(swath_hits),
                "ground_only_hits": len(ground_only_hits),
                "total_features": total_features,
                "total_ground": total_ground,
                "path": path,
                "url": f"/report/{self.token}/{archive_date_str}.html",
            }
        except Exception as e:
            log.error(f"Report save failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _update_index(self):
        """Build a simple index page listing all available reports, newest first."""
        try:
            files = sorted([f for f in os.listdir(REPORT_DIR) if f.endswith(".html") and f != "index.html"],
                          reverse=True)
            rows = []
            for fname in files:
                d = fname.replace(".html", "")
                rows.append(f'<li><a href="{d}.html">{d}</a></li>')
            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>StormDataPro — Daily Hit-Town Reports</title>
<style>body{{font-family:system-ui,sans-serif;max-width:600px;margin:40px auto;padding:0 20px;color:#1e293b}}
h1{{font-size:18px;color:#0a7a4e}} ul{{padding-left:20px;line-height:1.8}}
a{{color:#0a7a4e;text-decoration:none}} a:hover{{text-decoration:underline}}
.note{{font-size:12px;color:#64748b;margin-top:30px;line-height:1.5}}</style></head>
<body><h1>StormDataPro — Daily Hit-Town Reports</h1>
<p>Towns ≥{REPORT_MIN_TOWN_POP:,} population hit by ≥{REPORT_MIN_HAIL_INCHES}" hail with ≥{REPORT_MIN_COVERAGE_SQMI} sq mi of coverage.</p>
<ul>{"".join(rows) if rows else "<li>No reports available yet.</li>"}</ul>
<div class="note">Reports auto-generate at 00:00 CT for the prior storm day.<br>This URL is intentionally obscure — share carefully.</div>
</body></html>"""
            with open(os.path.join(REPORT_DIR, "index.html"), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            log.warning(f"Index update failed: {e}")

    def _render_html(self, date_str: str, swath_hits: list, ground_only_hits: list,
                    total_features: int, total_ground: int) -> str:
        """Render the full report as a single self-contained HTML page.

        Two sections:
          1. Swath-hit towns — sortable by hail size; with confirmation badge if
             ground reports also fell inside
          2. Ground-only towns — spotter-confirmed but no qualifying swath
        """
        # Friendly date for header
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            friendly_date = dt.strftime("%A, %B %-d, %Y")
        except Exception:
            friendly_date = date_str

        # Helper to render a single demographics column for a hit
        def render_demo_cells(h):
            d = h["demographics"]
            pop = d.get("total_pop") or 0
            income = d.get("median_household_income")
            white = d.get("race_white_alone") or 0
            black = d.get("race_black_alone") or 0
            aian = d.get("race_aian_alone") or 0
            asian = d.get("race_asian_alone") or 0
            hispanic = d.get("hispanic_or_latino") or 0
            pct = lambda n: f"{(n / pop * 100):.0f}%" if pop > 0 else "—"
            income_str = f"${income:,}" if income and income > 0 else "—"
            tsc_str = (f"{h['tsc_distance_miles']} mi" if h.get("tsc_distance_miles") is not None else "—")

            race_html = f"""<div><span class="bar w" style="width:{pct(white)}"></span> White {pct(white)}</div>
                  <div><span class="bar b" style="width:{pct(black)}"></span> Black {pct(black)}</div>
                  <div><span class="bar h" style="width:{pct(hispanic)}"></span> Hispanic {pct(hispanic)}</div>"""
            if asian / max(pop, 1) > 0.02:
                race_html += f'<div><span class="bar a" style="width:{pct(asian)}"></span> Asian {pct(asian)}</div>'
            if aian / max(pop, 1) > 0.02:
                race_html += f'<div><span class="bar n" style="width:{pct(aian)}"></span> Native {pct(aian)}</div>'

            return pop, income_str, tsc_str, race_html

        # Build swath-hit rows
        swath_rows = []
        for h in swath_hits:
            p = h["place"]
            pop, income_str, tsc_str, race_html = render_demo_cells(h)

            hail_str = f'{h["max_peak_inches"]:.1f}"'
            hail_class = ("h-large" if h["max_peak_inches"] >= 2.0 else
                         "h-mid" if h["max_peak_inches"] >= 1.5 else "h-small")

            # Confirmation badge — town has both swath AND ground reports
            n_ground = len(h.get("ground_reports", []))
            ground_max = h.get("ground_max_inches") or 0
            if n_ground > 0:
                confirm_html = f'<span class="confirm-badge">✓ {n_ground} report{"s" if n_ground > 1 else ""}'
                if ground_max > 0:
                    confirm_html += f' · max {ground_max:.1f}"'
                confirm_html += '</span>'
            else:
                confirm_html = '<span class="unconfirm-badge">radar only</span>'

            swath_rows.append(f"""
            <tr>
              <td><b>{p["name"]}</b><br><span class="state">{p["state"]}</span><br>{confirm_html}</td>
              <td class="num"><span class="hail {hail_class}">{hail_str}</span></td>
              <td class="num">{h["total_coverage_sqmi"]:.1f}</td>
              <td class="num">{pop:,}</td>
              <td class="num">{income_str}</td>
              <td class="race">{race_html}</td>
              <td class="num">{tsc_str}</td>
            </tr>""")

        # Build ground-only rows
        ground_rows = []
        for h in ground_only_hits:
            p = h["place"]
            pop, income_str, tsc_str, race_html = render_demo_cells(h)
            n_ground = len(h.get("ground_reports", []))
            ground_max = h["ground_max_inches"]
            hail_class = ("h-large" if ground_max >= 2.0 else
                         "h-mid" if ground_max >= 1.5 else "h-small")
            hail_str = f'{ground_max:.1f}"'

            # Build a comma-list of locations for the report cell
            locs = []
            for gr in h["ground_reports"][:5]:  # cap at 5 per row to avoid clutter
                gp = gr.get("properties", {})
                loc = gp.get("location", "")
                size = gp.get("sizeInches", 0)
                if loc:
                    locs.append(f'{loc} ({size}")')
            if len(h["ground_reports"]) > 5:
                locs.append(f'+ {len(h["ground_reports"]) - 5} more')
            locs_str = "; ".join(locs)

            ground_rows.append(f"""
            <tr>
              <td><b>{p["name"]}</b><br><span class="state">{p["state"]}</span><br><span class="confirm-badge">✓ {n_ground} report{"s" if n_ground > 1 else ""}</span></td>
              <td class="num"><span class="hail {hail_class}">{hail_str}</span></td>
              <td class="locs">{locs_str}</td>
              <td class="num">{pop:,}</td>
              <td class="num">{income_str}</td>
              <td class="race">{race_html}</td>
              <td class="num">{tsc_str}</td>
            </tr>""")

        # Build the body sections conditionally
        body_parts = []

        if swath_hits:
            body_parts.append(f"""
            <h2 class="section-h">Radar-Detected Swath Hits ({len(swath_hits)})</h2>
            <p class="section-note">Towns where hail polygons covered ≥{REPORT_MIN_COVERAGE_SQMI} sq mi inside the town footprint. Towns marked <b>✓ N reports</b> were confirmed by spotter ground reports — these are the highest-confidence leads.</p>
            <table>
              <thead><tr>
                <th>Town</th>
                <th class="num">Peak Hail<br>(radar)</th>
                <th class="num">Coverage<br>(sq mi)</th>
                <th class="num">Population</th>
                <th class="num">Median<br>HH Income</th>
                <th>Race / Ethnicity</th>
                <th class="num">Nearest<br>TSC</th>
              </tr></thead>
              <tbody>{"".join(swath_rows)}</tbody></table>""")
        elif total_features > 0:
            body_parts.append(f"""
            <h2 class="section-h">Radar-Detected Swath Hits</h2>
            <div class="empty-section">
              {total_features} swath polygon{"s" if total_features != 1 else ""} were on the map, but none impacted a town with population ≥{REPORT_MIN_TOWN_POP:,} and ≥{REPORT_MIN_COVERAGE_SQMI} sq mi of coverage.
            </div>""")

        if ground_only_hits:
            body_parts.append(f"""
            <h2 class="section-h" style="margin-top:32px">Ground-Only Confirmed Hits ({len(ground_only_hits)})</h2>
            <p class="section-note">Towns with spotter reports of ≥{REPORT_MIN_HAIL_INCHES}" hail but no qualifying radar swath. These typically indicate radar gaps, beam blockage, or fast cells the MRMS pipeline didn't paint. Worth investigating — the hail definitely fell.</p>
            <table>
              <thead><tr>
                <th>Town</th>
                <th class="num">Max Reported</th>
                <th>Report Locations</th>
                <th class="num">Population</th>
                <th class="num">Median<br>HH Income</th>
                <th>Race / Ethnicity</th>
                <th class="num">Nearest<br>TSC</th>
              </tr></thead>
              <tbody>{"".join(ground_rows)}</tbody></table>""")

        if not swath_hits and not ground_only_hits:
            body_parts.append(f"""<div class="empty">
                <h2>No qualifying towns hit on {friendly_date}</h2>
                <p>{total_features} swath polygon{"s" if total_features != 1 else ""} and {total_ground} ground report{"s" if total_ground != 1 else ""} were processed, but none combined to meet the hit criteria.</p>
            </div>""")

        body = "".join(body_parts)

        # Combined population for stats (swath + ground, dedup not needed since they're disjoint)
        total_pop = (sum(h["demographics"].get("total_pop") or 0 for h in swath_hits)
                     + sum(h["demographics"].get("total_pop") or 0 for h in ground_only_hits))
        confirmed_count = sum(1 for h in swath_hits if h.get("ground_reports"))

        styles = """
        :root { --green:#0a7a4e; --bg:#f8fafc; --border:#e2e8f0; --txt:#1e293b; --muted:#64748b; }
        * { box-sizing:border-box; }
        body { font-family:-apple-system,system-ui,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--txt); margin:0; padding:0; }
        header { background:var(--green); color:white; padding:18px 20px; }
        header h1 { margin:0 0 4px 0; font-size:18px; font-weight:700; }
        header .sub { font-size:12px; opacity:0.85; }
        .container { max-width:1300px; margin:0 auto; padding:20px; }
        .meta { background:white; border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:13px; }
        .meta .stat { display:inline-block; margin-right:24px; }
        .meta .stat b { font-size:18px; color:var(--green); display:block; }
        .section-h { font-size:15px; font-weight:700; color:var(--txt); margin:0 0 4px 0; padding-bottom:6px; border-bottom:2px solid var(--green); }
        .section-note { font-size:12px; color:var(--muted); margin:6px 0 14px 0; line-height:1.5; }
        table { width:100%; border-collapse:collapse; background:white; border:1px solid var(--border); border-radius:8px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
        th { background:#f1f5f9; text-align:left; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); padding:10px 12px; border-bottom:1px solid var(--border); }
        td { padding:14px 12px; border-bottom:1px solid var(--border); vertical-align:top; font-size:13px; }
        tr:last-child td { border-bottom:none; }
        tr:hover { background:#fafbfc; }
        .num { text-align:right; font-variant-numeric:tabular-nums; }
        .state { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }
        .hail { font-weight:700; padding:2px 7px; border-radius:4px; font-size:13px; }
        .hail.h-small { background:#fef3c7; color:#92400e; }
        .hail.h-mid { background:#fed7aa; color:#9a3412; }
        .hail.h-large { background:#fecaca; color:#991b1b; }
        .race { font-size:11px; line-height:1.55; min-width:200px; }
        .race .bar { display:inline-block; height:8px; vertical-align:middle; margin-right:6px; border-radius:2px; }
        .race .bar.w { background:#94a3b8; } .race .bar.b { background:#7c3aed; }
        .race .bar.h { background:#0ea5e9; } .race .bar.a { background:#f59e0b; }
        .race .bar.n { background:#10b981; }
        .confirm-badge { display:inline-block; margin-top:4px; font-size:10px; font-weight:600; color:#15803d; background:#dcfce7; padding:2px 6px; border-radius:3px; }
        .unconfirm-badge { display:inline-block; margin-top:4px; font-size:10px; font-weight:500; color:var(--muted); background:#f1f5f9; padding:2px 6px; border-radius:3px; }
        .locs { font-size:11px; color:#475569; line-height:1.4; max-width:240px; }
        .empty { background:white; border:1px solid var(--border); border-radius:8px; padding:40px; text-align:center; color:var(--muted); }
        .empty-section { background:white; border:1px solid var(--border); border-radius:8px; padding:20px; text-align:center; color:var(--muted); font-size:13px; }
        footer { padding:30px 20px; text-align:center; font-size:11px; color:var(--muted); }
        @media (max-width:700px) { .race { display:none; } .locs { max-width:120px; } th, td { padding:8px 6px; font-size:11px; } }
        """

        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StormDataPro Daily Report — {friendly_date}</title>
<meta name="robots" content="noindex,nofollow">
<style>{styles}</style></head>
<body>
<header>
  <h1>StormDataPro Daily Hit-Town Report</h1>
  <div class="sub">{friendly_date} · Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</div>
</header>
<div class="container">
  <div class="meta">
    <div class="stat"><b>{len(swath_hits)}</b>Swath Hits</div>
    <div class="stat"><b>{confirmed_count}</b>Spotter-Confirmed</div>
    <div class="stat"><b>{len(ground_only_hits)}</b>Ground-Only Hits</div>
    <div class="stat"><b>{total_pop:,}</b>Total Population</div>
    <div class="stat"><b>{total_features}</b>Total Polygons</div>
    <div class="stat"><b>{total_ground}</b>Ground Reports</div>
  </div>
  {body}
</div>
<footer>
  Sources: NOAA MRMS · NOAA SPC filtered hail reports · US Census ACS 2023 5-yr · OpenStreetMap (Tractor Supply locations)<br>
  Hail threshold: ≥{REPORT_MIN_HAIL_INCHES}" · Population threshold: ≥{REPORT_MIN_TOWN_POP:,} · Coverage threshold: ≥{REPORT_MIN_COVERAGE_SQMI} sq mi (swath section only)<br>
  Private report — do not share URL.
</footer></body></html>"""


daily_report = DailyReport()


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
    
    # Step 1.5: Generate the daily hit-town report from this archive
    try:
        report_result = daily_report.generate(archive_date_str)
        log.info(f"midnight_reset daily report: {report_result}")
    except Exception as e:
        log.error(f"midnight_reset daily report failed (non-fatal): {e}", exc_info=True)
    
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
        "version": "0.9.3",
        "phase": "9.3 — Heatmap: 1.0\" minimum, semi-transparent so basemap shows through",
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
            "GET /swaths/heatmap.png — HailPoint-style topographic heatmap (PNG)",
            "GET /swaths/heatmap/meta — heatmap geographic bounds + color scale",
            "GET /swaths/playback?step_minutes=20 — time-lapse frames",
            "GET /forecast/hail — HRRR 1-6h hail forecast polygons",
            "GET /forecast/hail?forecast_hour=N — single forecast hour (1-6)",
            "GET /forecast/stats — HRRR pipeline status",
            "GET /report/{token}/ — daily hit-town report index (private URL)",
            "GET /report/{token}/{date}.html — specific dated report",
            "GET /history/dates — list of archived storm days",
            "GET /history/summaries — per-day intensity summaries for calendar UI",
            "GET /history/{YYYY-MM-DD} — archived swaths + ground-truth for one day",
            "POST /admin/force-tick",
            "POST /admin/polygonize",
            "POST /admin/sanity-check",
            "POST /admin/reset",
            "POST /admin/archive-now",
            "POST /admin/prune-archives",
            "POST /admin/refresh-forecast",
            "POST /admin/generate-report?date=YYYY-MM-DD",
            "POST /admin/refresh-tsc — re-pull Tractor Supply locations from OSM",
            "GET /admin/report-token — reveal the private report URL token",
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


@app.get("/swaths/heatmap.png")
def get_swaths_heatmap_png():
    """Returns the latest hail heatmap as PNG (HailPoint-style topographic colors).
    The PNG is RGBA; transparent everywhere MESH < 0.75". Refreshed every polygon cycle.
    Use /swaths/heatmap/meta to get the geographic bounds."""
    if not os.path.exists(HEATMAP_PNG_FILE):
        raise HTTPException(status_code=404, detail="Heatmap not generated yet")
    return FileResponse(HEATMAP_PNG_FILE, media_type="image/png", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
    })


@app.get("/swaths/heatmap/meta")
def get_swaths_heatmap_meta():
    """Returns geographic bounds, image size, and color scale for the heatmap PNG.
    Frontend uses this to position the raster overlay correctly on the map."""
    if not os.path.exists(HEATMAP_META_FILE):
        return {"bounds": None, "size": None, "note": "Heatmap not generated yet"}
    try:
        with open(HEATMAP_META_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@app.get("/history/{date_str}/heatmap.png")
def history_heatmap_png(date_str: str):
    """Archived heatmap PNG for a storm day. Auto-restores from Supabase on a fresh deploy."""
    path = os.path.join(ARCHIVE_DIR, f"{date_str}_heatmap.png")
    if not os.path.exists(path):
        restore_heatmap_from_supabase(date_str)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No archived heatmap for {date_str}")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/history/{date_str}/heatmap/meta")
def history_heatmap_meta(date_str: str):
    """Archived heatmap bounds + color scale for a storm day."""
    path = os.path.join(ARCHIVE_DIR, f"{date_str}_heatmap_meta.json")
    if not os.path.exists(path):
        restore_heatmap_from_supabase(date_str)
    if not os.path.exists(path):
        return {"bounds": None, "size": None, "note": f"No archived heatmap for {date_str}"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


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


# ─────────────────────────────────────────────────────────────
# Daily Report endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/report/{token}/", response_class=HTMLResponse)
@app.get("/report/{token}", response_class=HTMLResponse)
def report_index(token: str):
    """Index page listing all available reports. Requires obscure token in URL."""
    if token != daily_report.token:
        raise HTTPException(status_code=404, detail="Not found")
    index_path = os.path.join(REPORT_DIR, "index.html")
    if not os.path.exists(index_path):
        # Generate an empty index on first access
        daily_report._update_index()
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>No reports available yet</h1>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/report/{token}/{filename}", response_class=HTMLResponse)
def report_file(token: str, filename: str):
    """Serve a specific dated report. Filename must be YYYY-MM-DD.html."""
    if token != daily_report.token:
        raise HTTPException(status_code=404, detail="Not found")
    # Sanitize filename — only allow YYYY-MM-DD.html or index.html
    if filename != "index.html":
        if not (len(filename) == 15 and filename.endswith(".html")):
            raise HTTPException(status_code=404, detail="Not found")
        try:
            datetime.strptime(filename[:10], "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/admin/report-token")
def admin_report_token():
    """Reveal the report URL token. Useful to look up the share URL after deploy."""
    return {
        "token": daily_report.token,
        "index_url": f"/report/{daily_report.token}/",
        "note": "Share this URL only with people who should access the daily report.",
    }


@app.post("/admin/generate-report")
def admin_generate_report(date: Optional[str] = None):
    """Manually generate a report for a specific archive date (default: yesterday CT).
    Useful for backfilling reports for past archive dates or testing."""
    if date is None:
        date = archiver.archive_date()
    return daily_report.generate(date)


@app.post("/admin/refresh-tsc")
def admin_refresh_tsc():
    """Force-refresh Tractor Supply locations from OSM Overpass."""
    daily_report.tsc_locations = None  # invalidate
    if os.path.exists(TSC_LOCATIONS_FILE):
        os.remove(TSC_LOCATIONS_FILE)
    ok = daily_report.ensure_tsc_locations()
    return {
        "ok": ok,
        "count": len(daily_report.tsc_locations) if daily_report.tsc_locations else 0,
    }


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


@app.get("/history/summaries")
def history_summaries():
    """Per-day summary stats for every archived date. Cached on disk; only
    today/yesterday are recomputed each call. Frontend calendar uses this to
    color-code days by intensity (hit count + peak hail size).
    
    Response shape:
      {
        "summaries": [
          {"date": "2026-04-29", "swath_count": 47, "peak_inches": 4.5, "ground_truth_count": 89},
          ...
        ],
        "today": "2026-04-30",
        "retention_days": 42
      }
    """
    return {
        "summaries": archiver.all_summaries(),
        "today": datetime.now(STORM_DAY_TZ).date().isoformat(),
        "retention_days": ARCHIVE_RETENTION_DAYS,
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
