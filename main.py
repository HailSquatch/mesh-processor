"""
StormDataPro MESH Processor — Phase 2
Decodes real MRMS MESH GRIB2 files AND accumulates max hail per pixel over time
to enable swath polygon generation in a later phase.

Key concepts:
- Background scheduler pulls MESH every 2 minutes
- Per-pixel max value accumulated into a persistent grid
- First-seen and last-seen timestamps tracked per pixel (for duration)
- State saved to disk every 5 minutes so restarts don't lose data
- Rolling 24-hour window — pixels that haven't been hit in 24h are zeroed
"""
import os
import gzip
import tempfile
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

# ── Setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="StormDataPro MESH Processor", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MESH_URL = "https://mrms.ncep.noaa.gov/data/2D/MESH/MRMS_MESH.latest.grib2.gz"
USER_AGENT = "StormDataPro/0.2 (colton@transcendentpdr.com)"

# Grid constants (MRMS CONUS 0.01° resolution)
GRID_ROWS = 3500
GRID_COLS = 7000
GRID_LAT1 = 54.995   # top-left lat (row 0)
GRID_LON1 = 230.005  # top-left lon in 0-360 convention (col 0)
GRID_DLAT = 0.01     # lat step per row (decreasing southward in scan=0)
GRID_DLON = 0.01

# Size thresholds (mm)
SIZE_THRESHOLDS_MM = [25.4, 38.1, 50.8, 69.85]  # 1.0", 1.5", 2.0", 2.75"

# Rolling window — pixels older than this are cleared out
ROLLING_WINDOW_HOURS = 24

# Persistence — Railway volume mounts at /data; fall back to tmp for dev
STATE_DIR = os.environ.get("STATE_DIR", "/data" if os.path.isdir("/data") else "/tmp/mesh-state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "accumulator.npz")


# ── Accumulator — thread-safe global state ──
class Accumulator:
    def __init__(self):
        self.lock = threading.Lock()
        self.max_mm = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        self.first_seen = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        self.last_seen = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
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
                self.max_mm = data['max_mm']
                self.first_seen = data['first_seen']
                self.last_seen = data['last_seen']
                self.update_count = int(data['update_count'][0])
                log.info(f"Loaded accumulator state from {STATE_FILE}")
            except Exception as e:
                log.warning(f"Could not load state (starting fresh): {e}")

    def save(self):
        tmp = STATE_FILE + ".tmp"
        try:
            np.savez_compressed(
                tmp,
                max_mm=self.max_mm,
                first_seen=self.first_seen,
                last_seen=self.last_seen,
                update_count=np.array([self.update_count]),
            )
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            log.error(f"Save failed: {e}")

    def apply_grid(self, values: np.ndarray, ts_unix: int):
        with self.lock:
            clean = np.where(np.isnan(values), -1.0, values).astype(np.float32)
            active = clean >= 5.0
            better = clean > self.max_mm
            self.max_mm = np.where(better, clean, self.max_mm)
            self.last_seen = np.where(active, ts_unix, self.last_seen)
            newly_active = active & (self.first_seen == 0)
            self.first_seen = np.where(newly_active, ts_unix, self.first_seen)
            self.update_count += 1
            self.last_update_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            self.last_update_maxmm = float(clean.max())
            self.last_update_pixels = int(active.sum())

    def prune_old(self, now_unix: int):
        cutoff = now_unix - (ROLLING_WINDOW_HOURS * 3600)
        with self.lock:
            stale = (self.last_seen > 0) & (self.last_seen < cutoff)
            n_stale = int(stale.sum())
            if n_stale > 0:
                self.max_mm[stale] = 0
                self.first_seen[stale] = 0
                self.last_seen[stale] = 0
                log.info(f"Pruned {n_stale} stale pixels older than {ROLLING_WINDOW_HOURS}h")

    def snapshot_stats(self) -> dict:
        with self.lock:
            hail_mask = self.max_mm >= 25.4
            return {
                "update_count": self.update_count,
                "last_update": self.last_update_ts,
                "last_update_max_mm": round(self.last_update_maxmm, 2),
                "last_update_max_inches": round(self.last_update_maxmm / 25.4, 2),
                "last_update_active_pixels": self.last_update_pixels,
                "accumulated_hail_pixels_1in": int(hail_mask.sum()),
                "accumulated_max_mm": round(float(self.max_mm.max()), 2),
                "accumulated_max_inches": round(float(self.max_mm.max()) / 25.4, 2),
                "state_file": STATE_FILE,
                "last_error": self.last_error,
            }


accumulator = Accumulator()


# ── MESH fetch + decode ──
def download_mesh() -> Optional[bytes]:
    log.info(f"Fetching {MESH_URL}")
    try:
        r = requests.get(MESH_URL, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        return gzip.decompress(r.content)
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def decode_mesh(grib_bytes: bytes) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name
    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        if not messages:
            return None
        grb = messages[0]
        values = grb.values
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
    try:
        grib_bytes = download_mesh()
        if not grib_bytes:
            accumulator.last_error = "download failed"
            return
        result = decode_mesh(grib_bytes)
        if not result:
            accumulator.last_error = "decode failed"
            return
        ts_unix = int(result["timestamp"].timestamp())
        accumulator.apply_grid(result["values"], ts_unix)
        accumulator.last_error = None
        log.info(
            f"Tick #{accumulator.update_count}: "
            f"{accumulator.last_update_pixels} active px, "
            f"max {accumulator.last_update_maxmm:.1f}mm "
            f"({accumulator.last_update_maxmm / 25.4:.2f}\")"
        )
    except Exception as e:
        accumulator.last_error = str(e)
        log.error(f"Tick failed: {e}", exc_info=True)


def periodic_save():
    accumulator.save()
    log.info(f"State saved to {STATE_FILE}")


def periodic_prune():
    accumulator.prune_old(int(datetime.now(timezone.utc).timestamp()))


# ── Scheduler ──
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    tick, "interval", minutes=2, id="mesh_tick",
    next_run_time=datetime.now(timezone.utc) + timedelta(seconds=15)
)
scheduler.add_job(periodic_save, "interval", minutes=5, id="save_state")
scheduler.add_job(periodic_prune, "interval", hours=1, id="prune_old")


@app.on_event("startup")
def startup():
    scheduler.start()
    log.info("Scheduler started — MESH tick every 2 min, save every 5 min, prune hourly")


@app.on_event("shutdown")
def shutdown():
    log.info("Shutting down — saving final state")
    try:
        accumulator.save()
    except Exception:
        pass
    scheduler.shutdown(wait=False)


# ── Helpers ──
def pixel_to_lonlat(ridx: int, cidx: int) -> tuple:
    lat = GRID_LAT1 - ridx * GRID_DLAT
    lon_raw = GRID_LON1 + cidx * GRID_DLON
    lon = lon_raw - 360 if lon_raw > 180 else lon_raw
    return lon, lat


def _points_to_geojson(values, lats, lons, min_mm: float, max_points: int = 500):
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
        "version": "0.2.0",
        "phase": "2 — accumulation",
        "endpoints": [
            "GET /health",
            "GET /status — scheduler + accumulator summary",
            "GET /test — live fetch+decode test",
            "GET /test-points — current hail >=1in as points",
            "GET /test-all — all detected hail (any size)",
            "GET /accumulated?min_inches=1.0 — accumulated max-hail over rolling 24h",
            "GET /accumulated/stats — accumulator summary",
            "POST /admin/force-tick — manually trigger one MESH fetch",
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
    }


@app.get("/test")
def test_decode():
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "Could not fetch MESH file from NOAA")
    result = decode_mesh(grib_bytes)
    if not result:
        raise HTTPException(500, "Could not decode MESH GRIB2")
    values = result["values"]
    valid_count = int(np.sum(~np.isnan(values)))
    hail_count = int(np.sum(values >= 25.4))
    max_val = float(np.nanmax(values)) if valid_count > 0 else 0.0
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
        raise HTTPException(500, "Could not fetch")
    result = decode_mesh(grib_bytes)
    if not result:
        raise HTTPException(500, "Could not decode")
    features = _points_to_geojson(result["values"], result["lats"], result["lons"], min_mm=25.4, max_points=500)
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
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "Could not fetch")
    result = decode_mesh(grib_bytes)
    if not result:
        raise HTTPException(500, "Could not decode")
    features = _points_to_geojson(result["values"], result["lats"], result["lons"], min_mm=5.0, max_points=2000)
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
    """
    GeoJSON of the accumulated max-per-pixel hail over the rolling 24h window.
    This is the data polygons are built from in Phase 3.
    """
    min_mm = min_inches * 25.4
    with accumulator.lock:
        mask = accumulator.max_mm >= min_mm
        n_above = int(mask.sum())
        if n_above == 0:
            return {
                "type": "FeatureCollection",
                "metadata": {
                    "window_hours": ROLLING_WINDOW_HOURS,
                    "min_inches": min_inches,
                    "count": 0,
                    "total_above_threshold": 0,
                    "update_count": accumulator.update_count,
                    "last_update": accumulator.last_update_ts,
                    "note": "No hail in rolling window at this threshold",
                },
                "features": [],
            }
        indices = np.argwhere(mask)
        sizes = accumulator.max_mm[mask]
        order = np.argsort(-sizes)[:max_points]

        features = []
        for i in order:
            ridx, cidx = indices[i]
            mm = float(accumulator.max_mm[ridx, cidx])
            fs = int(accumulator.first_seen[ridx, cidx])
            ls = int(accumulator.last_seen[ridx, cidx])
            duration_min = round((ls - fs) / 60.0, 1) if (fs > 0 and ls > 0) else 0
            lon, lat = pixel_to_lonlat(int(ridx), int(cidx))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                "properties": {
                    "sizeMM": round(mm, 1),
                    "sizeInches": round(mm / 25.4, 2),
                    "firstSeen": datetime.fromtimestamp(fs, tz=timezone.utc).isoformat() if fs > 0 else None,
                    "lastSeen": datetime.fromtimestamp(ls, tz=timezone.utc).isoformat() if ls > 0 else None,
                    "durationMinutes": duration_min,
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


@app.post("/admin/force-tick")
def admin_force_tick():
    """Force a MESH fetch right now instead of waiting for next scheduled tick."""
    tick()
    return {"status": "done", "stats": accumulator.snapshot_stats()}


@app.post("/admin/reset")
def admin_reset():
    """Wipe accumulator state. Use carefully."""
    global accumulator
    with accumulator.lock:
        accumulator.max_mm[:] = 0
        accumulator.first_seen[:] = 0
        accumulator.last_seen[:] = 0
        accumulator.update_count = 0
        accumulator.last_update_ts = None
    accumulator.save()
    return {"status": "reset"}
