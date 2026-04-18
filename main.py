"""
StormDataPro MESH Processor — Phase 3 (polygonization)

Phase 1: MESH decode (done)
Phase 2: Rolling-window accumulator (done)
Phase 3: Contour the accumulator into swath polygons at size thresholds
         and serve them as GeoJSON for map rendering.

Memory budget breakdown:
  max_mm_x10   int16  3500*7000*2 =  49 MB
  last_seen_h  int32  3500*7000*4 =  98 MB
  TOTAL persistent state:               147 MB

Polygonization adds ~100 MB peak during the contour job but only runs every 5 min
and results are cached to disk and served from memory between runs.
"""
import os
import gzip
import tempfile
import logging
import threading
import gc
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

# Phase 3 imports
from scipy import ndimage
from rasterio import features as rio_features
from rasterio.transform import Affine
from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.geometry import Polygon, MultiPolygon

# ── Setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="StormDataPro MESH Processor", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MESH_URL = "https://mrms.ncep.noaa.gov/data/2D/MESH/MRMS_MESH.latest.grib2.gz"
USER_AGENT = "StormDataPro/0.2 (colton@transcendentpdr.com)"

# MRMS CONUS grid at 0.01° resolution
GRID_ROWS = 3500
GRID_COLS = 7000
GRID_LAT1 = 54.995       # top-left lat (row 0)
GRID_LON1_360 = 230.005  # top-left lon in 0-360 convention (col 0)
GRID_DEG = 0.01

# Hail size storage: store as int16 where value = mm * 10
# Max representable: 3276.7 mm (far beyond any real hail ~200mm softball)
# Min resolution: 0.1mm which is better than the 1mm resolution of the source data
MM_SCALE = 10
MIN_HAIL_STORED_MM = 5.0   # only store pixels with ≥5mm (0.2")
MIN_HAIL_STORED = int(MIN_HAIL_STORED_MM * MM_SCALE)  # 50

# Rolling 24h window
ROLLING_WINDOW_HOURS = 24

# Persistence
STATE_DIR = os.environ.get("STATE_DIR", "/data" if os.path.isdir("/data") else "/tmp/mesh-state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "accumulator.npz")


# ── Accumulator ──
class Accumulator:
    def __init__(self):
        self.lock = threading.Lock()
        # Hail size as int16 (mm * 10). Zero means "no hail seen here".
        self.max_mm_x10 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16)
        # Hours since unix epoch as int32 (comfortably fits until year ~250,000)
        self.last_seen_h = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        # Track first-seen via a single timestamp per pixel only when we upgrade max
        # Use a secondary sparse structure — we'll build it on demand from history
        # For now, we drop first_seen to save memory. Duration is computed at Phase 3.
        self.update_count = 0
        self.last_update_ts: Optional[str] = None
        self.last_update_maxmm: float = 0.0
        self.last_update_pixels: int = 0
        self.last_error: Optional[str] = None
        self.load()

    def load(self):
        if os.path.exists(STATE_FILE):
            try:
                data = np.load(STATE_FILE)
                self.max_mm_x10 = data['max_mm_x10']
                self.last_seen_h = data['last_seen_h']
                self.update_count = int(data['update_count'][0])
                log.info(f"Loaded accumulator state (updates={self.update_count})")
            except Exception as e:
                log.warning(f"Could not load state, starting fresh: {e}")

    def save(self):
        tmp = STATE_FILE + ".tmp"
        try:
            np.savez_compressed(
                tmp,
                max_mm_x10=self.max_mm_x10,
                last_seen_h=self.last_seen_h,
                update_count=np.array([self.update_count]),
            )
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            log.error(f"Save failed: {e}")

    def apply_grid(self, values_mm: np.ndarray, ts_unix: int):
        """Merge new MESH grid. values_mm is float32 in millimeters.
        
        Memory-lean path: avoid creating full-grid temp float arrays.
        Approach: find hail pixels via a sparse mask first (most of grid is -1),
        then update only those locations in the persistent arrays.
        """
        with self.lock:
            # First, find pixels with any hail (>=0.5mm) — skip the rest
            # This creates a bool mask (24 MB) but we'll free it fast
            hail_mask = values_mm >= 0.5
            n_hail = int(hail_mask.sum())
            
            if n_hail > 0:
                # Extract only the hail values and their positions
                hail_vals = values_mm[hail_mask]  # small 1D array
                # Convert to int16 * scale
                hail_x10 = np.clip(hail_vals * MM_SCALE, 0, 32000).astype(np.int16)
                
                # Get 2D indices of hail pixels
                rows, cols = np.where(hail_mask)
                
                # Update max_mm_x10 only where new > existing
                existing = self.max_mm_x10[rows, cols]
                bigger = hail_x10 > existing
                update_rows = rows[bigger]
                update_cols = cols[bigger]
                update_vals = hail_x10[bigger]
                self.max_mm_x10[update_rows, update_cols] = update_vals
                
                # Update last_seen for all active pixels (meaningful hail >= MIN_HAIL_STORED)
                active = hail_x10 >= MIN_HAIL_STORED
                active_rows = rows[active]
                active_cols = cols[active]
                self.last_seen_h[active_rows, active_cols] = np.int32(ts_unix // 3600)
                
                max_this_tick = float(hail_x10.max()) / MM_SCALE if len(hail_x10) > 0 else 0.0
                active_count = int(active.sum())
            else:
                max_this_tick = 0.0
                active_count = 0
            
            self.update_count += 1
            self.last_update_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            self.last_update_maxmm = max_this_tick
            self.last_update_pixels = active_count

    def prune_old(self, now_unix: int):
        """Zero out pixels whose last_seen is older than the rolling window."""
        cutoff_h = np.int32((now_unix - (ROLLING_WINDOW_HOURS * 3600)) // 3600)
        with self.lock:
            # Pixels that were seen but are now too old
            has_data = self.last_seen_h > 0
            too_old = self.last_seen_h < cutoff_h
            stale = has_data & too_old
            n_stale = int(stale.sum())
            if n_stale > 0:
                self.max_mm_x10[stale] = 0
                self.last_seen_h[stale] = 0
                log.info(f"Pruned {n_stale} stale pixels")
            return n_stale

    def snapshot_stats(self) -> dict:
        with self.lock:
            hail_1in = self.max_mm_x10 >= int(25.4 * MM_SCALE)
            return {
                "update_count": self.update_count,
                "last_update": self.last_update_ts,
                "last_update_max_mm": round(self.last_update_maxmm, 2),
                "last_update_max_inches": round(self.last_update_maxmm / 25.4, 2),
                "last_update_active_pixels": self.last_update_pixels,
                "accumulated_hail_pixels_1in": int(hail_1in.sum()),
                "accumulated_max_mm": round(float(self.max_mm_x10.max()) / MM_SCALE, 2),
                "accumulated_max_inches": round(float(self.max_mm_x10.max()) / MM_SCALE / 25.4, 2),
                "state_file": STATE_FILE,
                "last_error": self.last_error,
            }


accumulator = Accumulator()


# ── Phase 3: Polygonizer ──
# Size thresholds in mm * 10 (matching int16 storage in accumulator)
# IHM/HailTrace-style color ramp
POLYGON_THRESHOLDS = [
    {"min_mm": 25.4, "inches": 1.0,  "desc": "1.0\" (quarter)",  "color": "#eab308"},
    {"min_mm": 38.1, "inches": 1.5,  "desc": "1.5\" (walnut)",   "color": "#f97316"},
    {"min_mm": 50.8, "inches": 2.0,  "desc": "2.0\" (golf ball)","color": "#ef4444"},
    {"min_mm": 69.85,"inches": 2.75, "desc": "2.75\"+ (baseball)","color": "#a855f7"},
]

# Rasterio affine transform for the MRMS grid (maps pixel → lon/lat in -180/180)
GRID_TRANSFORM = Affine(
    GRID_DEG, 0.0, GRID_LON1_360 - 360,   # px width, 0, left edge lon in -180/180
    0.0, -GRID_DEG, GRID_LAT1              # 0, -px height (lat decreasing south), top edge lat
)

# Polygon cache file (survives restarts via the same volume)
POLYGONS_CACHE_FILE = os.path.join(STATE_DIR, "polygons.json")


class Polygonizer:
    """Converts accumulator raster into size-thresholded swath polygons."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_run_ts: Optional[str] = None
        self.last_run_elapsed_ms: int = 0
        self.last_error: Optional[str] = None
        # Cached GeoJSON feature collection
        self.cached_geojson: dict = {
            "type": "FeatureCollection",
            "metadata": {"note": "polygons not yet generated"},
            "features": [],
        }
        self._load_cache()

    def _load_cache(self):
        """Load polygons from disk on startup."""
        if os.path.exists(POLYGONS_CACHE_FILE):
            try:
                with open(POLYGONS_CACHE_FILE, "r") as f:
                    self.cached_geojson = json.load(f)
                n = len(self.cached_geojson.get("features", []))
                log.info(f"Loaded {n} cached polygons from disk")
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
        """
        Rebuild polygons from the current accumulator state.
        Called on a schedule (every 5 min) or manually via admin endpoint.
        """
        t0 = datetime.now(timezone.utc)
        try:
            # Snapshot the accumulator arrays under lock, then release
            with accumulator.lock:
                max_snapshot = accumulator.max_mm_x10.copy()
                last_seen_snapshot = accumulator.last_seen_h.copy()
                update_count = accumulator.update_count
                last_update_ts = accumulator.last_update_ts

            all_features = []

            for thresh in POLYGON_THRESHOLDS:
                thresh_x10 = int(thresh["min_mm"] * MM_SCALE)

                # 1. Threshold
                mask = max_snapshot >= thresh_x10
                raw_count = int(mask.sum())
                if raw_count == 0:
                    continue

                # 2. Morphological cleanup
                # Opening: kills isolated single-pixel noise
                # Closing: fills tiny gaps so polygons don't have swiss cheese holes
                cleaned = ndimage.binary_opening(mask, iterations=1)
                cleaned = ndimage.binary_closing(cleaned, iterations=2)
                cleaned_count = int(cleaned.sum())
                if cleaned_count == 0:
                    continue

                # 3. Polygonize
                uint8_mask = cleaned.astype(np.uint8)
                raw_polys = []
                for geom, val in rio_features.shapes(uint8_mask, mask=cleaned, transform=GRID_TRANSFORM):
                    if val == 1:
                        raw_polys.append(shp_shape(geom))

                # 4. Simplify and filter
                # simplify tolerance 0.015° ≈ 1 mile — keeps shape, trims vertices
                # area filter 0.0001 sq-deg ≈ ~0.4 sq mi — drops tiny false polys
                simplified = []
                for p in raw_polys:
                    sp = p.simplify(0.015, preserve_topology=True)
                    if sp.area >= 0.0001 and not sp.is_empty:
                        simplified.append(sp)

                # 5. Attribute each polygon with size stats from underlying accumulator
                for poly in simplified:
                    # Compute a bounding box in pixel space to get peak size within the polygon
                    minx, miny, maxx, maxy = poly.bounds
                    # Reverse-project bbox to grid indices
                    c0 = max(0, int((minx - (GRID_LON1_360 - 360)) / GRID_DEG))
                    c1 = min(GRID_COLS, int((maxx - (GRID_LON1_360 - 360)) / GRID_DEG) + 1)
                    r1 = max(0, int((GRID_LAT1 - maxy) / GRID_DEG))
                    r0 = min(GRID_ROWS, int((GRID_LAT1 - miny) / GRID_DEG) + 1)
                    # Peak size inside bbox (simple, fast; precise in-polygon stats would need per-pixel test)
                    if c1 > c0 and r0 > r1:
                        sub = max_snapshot[r1:r0, c0:c1]
                        sub_mask = sub >= thresh_x10
                        peak_x10 = int(sub[sub_mask].max()) if sub_mask.any() else thresh_x10
                        last_hours_sub = last_seen_snapshot[r1:r0, c0:c1]
                        last_hours_sub = last_hours_sub[last_hours_sub > 0]
                        last_unix = int(last_hours_sub.max()) * 3600 if len(last_hours_sub) > 0 else 0
                        first_unix = int(last_hours_sub.min()) * 3600 if len(last_hours_sub) > 0 else 0
                        duration_min = round((last_unix - first_unix) / 60.0, 0) if last_unix > 0 and first_unix > 0 else 0
                    else:
                        peak_x10 = thresh_x10
                        last_unix = 0
                        first_unix = 0
                        duration_min = 0

                    # Area in sq miles (rough: 1 sq deg ≈ 3800 sq mi at ~40°N)
                    area_sqmi = round(poly.area * 3800, 1)

                    # Convert polygon to GeoJSON
                    geom_json = shp_mapping(poly)

                    feature = {
                        "type": "Feature",
                        "geometry": geom_json,
                        "properties": {
                            "thresholdInches": thresh["inches"],
                            "thresholdLabel": thresh["desc"],
                            "color": thresh["color"],
                            "peakSizeMM": round(peak_x10 / MM_SCALE, 1),
                            "peakSizeInches": round(peak_x10 / MM_SCALE / 25.4, 2),
                            "areaSquareMiles": area_sqmi,
                            "firstSeen": datetime.fromtimestamp(first_unix, tz=timezone.utc).isoformat() if first_unix > 0 else None,
                            "lastSeen": datetime.fromtimestamp(last_unix, tz=timezone.utc).isoformat() if last_unix > 0 else None,
                            "durationMinutes": duration_min,
                        },
                    }
                    all_features.append(feature)

                # Free per-threshold temps
                mask = cleaned = uint8_mask = raw_polys = simplified = None
                gc.collect()

            # Free the snapshot
            max_snapshot = last_seen_snapshot = None
            gc.collect()

            # Sort features so larger thresholds render on top (painters algorithm)
            all_features.sort(key=lambda f: f["properties"]["thresholdInches"])

            elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

            new_geojson = {
                "type": "FeatureCollection",
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "polygon_count": len(all_features),
                    "accumulator_update_count": update_count,
                    "accumulator_last_update": last_update_ts,
                    "window_hours": ROLLING_WINDOW_HOURS,
                    "thresholds": [t["inches"] for t in POLYGON_THRESHOLDS],
                },
                "features": all_features,
            }

            with self.lock:
                self.cached_geojson = new_geojson
                self.last_run_ts = new_geojson["metadata"]["generated_at"]
                self.last_run_elapsed_ms = elapsed_ms
                self.last_error = None

            self._save_cache()
            log.info(f"Polygonization: {len(all_features)} polygons in {elapsed_ms}ms")

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
            }


polygonizer = Polygonizer()


# ── MESH fetch + decode (memory-lean) ──
def download_mesh() -> Optional[bytes]:
    log.info(f"Fetching {MESH_URL}")
    try:
        r = requests.get(MESH_URL, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        return gzip.decompress(r.content)
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def decode_mesh_values_only(grib_bytes: bytes) -> Optional[dict]:
    """
    Memory-lean decode: only return the values array + timestamp.
    Does NOT compute lats/lons arrays (saves ~200 MB).
    Lat/lon can be derived from pixel indices on demand.
    """
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name
    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        if not messages:
            return None
        grb = messages[0]
        values = np.asarray(grb.values, dtype=np.float32)  # single allocation
        try:
            ts = grb.validDate.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        grbs.close()
        return {"values": values, "timestamp": ts}
    except Exception as e:
        log.error(f"Decode failed: {e}", exc_info=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def decode_mesh_with_coords(grib_bytes: bytes) -> Optional[dict]:
    """
    Full decode including lat/lon arrays. Only used by /test endpoints for verification.
    Should not be used in the scheduled tick loop.
    """
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name
    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        if not messages:
            return None
        grb = messages[0]
        values = np.asarray(grb.values, dtype=np.float32)
        lats, lons = grb.latlons()
        values = np.where(values < 0, np.nan, values)
        try:
            ts = grb.validDate.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        grbs.close()
        return {"values": values, "lats": lats, "lons": lons, "timestamp": ts}
    except Exception as e:
        log.error(f"Decode failed: {e}", exc_info=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def tick():
    """Scheduled job: pull latest MESH and merge into accumulator. Aggressive gc."""
    try:
        grib_bytes = download_mesh()
        if not grib_bytes:
            accumulator.last_error = "download failed"
            return
        result = decode_mesh_values_only(grib_bytes)
        grib_bytes = None  # free immediately
        if not result:
            accumulator.last_error = "decode failed"
            return
        ts_unix = int(result["timestamp"].timestamp())
        accumulator.apply_grid(result["values"], ts_unix)
        result = None  # free values array
        gc.collect()
        accumulator.last_error = None
        log.info(
            f"Tick #{accumulator.update_count}: "
            f"{accumulator.last_update_pixels} active px, "
            f"max {accumulator.last_update_maxmm:.1f}mm"
        )
    except Exception as e:
        accumulator.last_error = str(e)
        log.error(f"Tick failed: {e}", exc_info=True)


def periodic_save():
    accumulator.save()


def periodic_prune():
    accumulator.prune_old(int(datetime.now(timezone.utc).timestamp()))


def periodic_polygonize():
    """Rebuild the swath polygons from the current accumulator state."""
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
scheduler.add_job(periodic_prune, "interval", hours=1, id="prune_old",
                  max_instances=1, coalesce=True)
scheduler.add_job(
    periodic_polygonize, "interval", minutes=5, id="polygonize",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=90),
    max_instances=1, coalesce=True,
)


@app.on_event("startup")
def startup():
    scheduler.start()
    log.info("Scheduler started — tick 2min, save 10min, prune hourly, polygonize 5min")


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


def _values_to_points(values: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                     min_mm: float, max_points: int) -> list:
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
        "version": "0.3.0",
        "phase": "3 — polygonization",
        "endpoints": [
            "GET /health",
            "GET /status — scheduler + accumulator + polygonizer summary",
            "GET /test — live fetch+decode test",
            "GET /test-points — current hail >=1in",
            "GET /test-all — all detected hail (any size)",
            "GET /accumulated?min_inches=1.0 — accumulated points over 24h",
            "GET /accumulated/stats — accumulator summary",
            "GET /swaths — swath POLYGONS as GeoJSON (the main map data)",
            "GET /swaths/stats — polygonizer summary",
            "POST /admin/force-tick — trigger a MESH fetch now",
            "POST /admin/polygonize — rebuild polygons now",
            "POST /admin/reset — wipe accumulator state",
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
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "fetch failed")
    result = decode_mesh_values_only(grib_bytes)
    if not result:
        raise HTTPException(500, "decode failed")
    values = result["values"]
    valid_count = int(np.sum(values >= 0))  # pygrib returns -1 or -999 for missing
    hail_count = int(np.sum(values >= 25.4))
    max_val = float(values.max())
    return {
        "status": "success",
        "timestamp": result["timestamp"].isoformat(),
        "grid_shape": list(values.shape),
        "valid_pixels": valid_count,
        "hail_pixels_1in_plus": hail_count,
        "max_hail_mm": round(max_val, 2),
        "max_hail_inches": round(max_val / 25.4, 2),
        "fetched_bytes": len(grib_bytes),
    }


@app.get("/test-points")
def test_points():
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "fetch failed")
    result = decode_mesh_with_coords(grib_bytes)
    if not result:
        raise HTTPException(500, "decode failed")
    features = _values_to_points(result["values"], result["lats"], result["lons"],
                                  min_mm=25.4, max_points=500)
    return {
        "type": "FeatureCollection",
        "metadata": {"timestamp": result["timestamp"].isoformat(), "threshold": ">=1 inch", "count": len(features)},
        "features": features,
    }


@app.get("/test-all")
def test_all():
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "fetch failed")
    result = decode_mesh_with_coords(grib_bytes)
    if not result:
        raise HTTPException(500, "decode failed")
    features = _values_to_points(result["values"], result["lats"], result["lons"],
                                  min_mm=5.0, max_points=2000)
    return {
        "type": "FeatureCollection",
        "metadata": {"timestamp": result["timestamp"].isoformat(), "threshold": ">=0.2 inch", "count": len(features)},
        "features": features,
    }


@app.get("/accumulated")
def accumulated(min_inches: float = 1.0, max_points: int = 2000):
    """Accumulated max-per-pixel hail from rolling 24h window as GeoJSON."""
    min_mm_x10 = int(min_inches * 25.4 * MM_SCALE)
    with accumulator.lock:
        mask = accumulator.max_mm_x10 >= min_mm_x10
        n_above = int(mask.sum())
        if n_above == 0:
            return {
                "type": "FeatureCollection",
                "metadata": {
                    "window_hours": ROLLING_WINDOW_HOURS,
                    "min_inches": min_inches,
                    "count": 0,
                    "update_count": accumulator.update_count,
                    "last_update": accumulator.last_update_ts,
                    "note": "No hail in window at this threshold",
                },
                "features": [],
            }
        indices = np.argwhere(mask)
        sizes_x10 = accumulator.max_mm_x10[mask]
        last_hours = accumulator.last_seen_h[mask]
        order = np.argsort(-sizes_x10)[:max_points]

        features = []
        for i in order:
            ridx, cidx = indices[i]
            mm = float(accumulator.max_mm_x10[ridx, cidx]) / MM_SCALE
            last_h = int(accumulator.last_seen_h[ridx, cidx])
            last_unix = last_h * 3600 if last_h > 0 else 0
            lon, lat = pixel_to_lonlat(int(ridx), int(cidx))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                "properties": {
                    "sizeMM": round(mm, 1),
                    "sizeInches": round(mm / 25.4, 2),
                    "lastSeen": datetime.fromtimestamp(last_unix, tz=timezone.utc).isoformat() if last_unix > 0 else None,
                },
            })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "window_hours": ROLLING_WINDOW_HOURS,
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


# ── Phase 3: Swath polygon endpoints ──
@app.get("/swaths")
def get_swaths():
    """
    The MAIN MAP DATA endpoint.
    Returns swath polygons at size thresholds (1", 1.5", 2", 2.75"+) as GeoJSON.
    Cached in memory; rebuilt every 5 minutes by the polygonizer job.
    """
    return polygonizer.get_geojson()


@app.get("/swaths/stats")
def swaths_stats():
    return polygonizer.stats()


@app.post("/admin/polygonize")
def admin_polygonize():
    """Force a polygon rebuild immediately instead of waiting for the 5-min scheduled run."""
    polygonizer.run()
    return {"status": "done", "stats": polygonizer.stats()}


@app.post("/admin/force-tick")
def admin_force_tick():
    tick()
    return {"status": "done", "stats": accumulator.snapshot_stats()}


@app.post("/admin/reset")
def admin_reset():
    with accumulator.lock:
        accumulator.max_mm_x10[:] = 0
        accumulator.last_seen_h[:] = 0
        accumulator.update_count = 0
        accumulator.last_update_ts = None
    accumulator.save()
    # Also clear the polygon cache
    with polygonizer.lock:
        polygonizer.cached_geojson = {
            "type": "FeatureCollection",
            "metadata": {"note": "reset — polygons not yet generated"},
            "features": [],
        }
    polygonizer._save_cache()
    return {"status": "reset"}
