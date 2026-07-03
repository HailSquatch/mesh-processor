"""
StormDataPro Hail Maps — Backend v2.0.0
========================================
Complete rewrite. Design principles:

1. STATELESS. We no longer accumulate 2-minute MESH snapshots ourselves.
   NSSL/MRMS publishes rolling-max accumulation products (MESH_Max_1440min,
   MESH_Max_60min) built from every radar scan with no gaps. We fetch those.
   A fresh deploy is fully repopulated ~60 seconds after boot. Missed fetches
   cost nothing — the next fetch contains everything.

2. RECALIBRATED. Raw MRMS MESH uses the Witt et al. (1998) fit, which
   overestimates larger hail (the "Memphis problem"). We invert to SHI and
   apply a smooth logistic blend into the Murillo & Homeyer (2021) 75th
   percentile fit at high SHI (formulas verified against pyhail).
   An optional regional bias multiplier (REGIONAL_BIAS env) stacks on top.

3. GROUND-TRUTH FUSED. SPC + IEM LSR hail reports are not just dots — swath
   components whose ground reports consistently disagree with radar are
   capped to (max ground report x GT_CAP_MARGIN) and flagged "verified".

4. LOCAL-MIDNIGHT ARCHIVE. Snapshots at midnight America/Chicago (DST-aware),
   so storm days no longer split at 7 PM CT. Archives back up to Supabase
   Storage and auto-restore on fresh deploys.

Deploy: Railway. Persistent volume at /data (optional — Supabase covers loss).
Env vars (all optional):
  SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_BUCKET   archive backup
  REGIONAL_BIAS      float multiplier on calibrated size (default 1.0)
  MIN_INCH           smallest displayed hail, inches (default 0.75)
  FETCH_INTERVAL     seconds between cycles (default 240)
  LOCAL_TZ           archive timezone (default America/Chicago)
  DATA_DIR           storage root (default /data)
  MESH_1440_URL, MESH_60_URL, SPC_URL, LSR_URL   source overrides (testing)
  PORT               server port (default 8000)
"""

import gzip
import io
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hailmaps")

VERSION = "2.1.0"

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
LIVE_DIR = os.path.join(DATA_DIR, "live")
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)

MESH_1440_URL = os.environ.get(
    "MESH_1440_URL",
    "https://mrms.ncep.noaa.gov/2D/MESH_Max_1440min/MRMS_MESH_Max_1440min.latest.grib2.gz",
)
MESH_60_URL = os.environ.get(
    "MESH_60_URL",
    "https://mrms.ncep.noaa.gov/2D/MESH_Max_60min/MRMS_MESH_Max_60min.latest.grib2.gz",
)
SPC_URL = os.environ.get(
    "SPC_URL", "https://www.spc.noaa.gov/climo/reports/today_raw_hail.csv"
)
SPC_URL_FALLBACK = "https://www.spc.noaa.gov/climo/reports/today_hail.csv"
LSR_URL = os.environ.get(
    "LSR_URL",
    "https://mesonet.agron.iastate.edu/geojson/lsr.geojson?hours=24",
)
IEM_MESH_BASE = os.environ.get(
    "IEM_MESH_BASE", "https://mtarchive.geol.iastate.edu").rstrip("/")
SPC_DATED_BASE = os.environ.get(
    "SPC_DATED_BASE", "https://www.spc.noaa.gov/climo/reports").rstrip("/")

FETCH_INTERVAL = int(os.environ.get("FETCH_INTERVAL", "240"))
MIN_INCH = float(os.environ.get("MIN_INCH", "0.75"))
REGIONAL_BIAS = float(os.environ.get("REGIONAL_BIAS", "1.0"))
LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "America/Chicago"))
GT_CAP_MARGIN = float(os.environ.get("GT_CAP_MARGIN", "1.25"))  # cap = max report * margin
GT_MIN_REPORTS = int(os.environ.get("GT_MIN_REPORTS", "2"))     # reports needed to cap

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "heatmaps")

MM_PER_INCH = 25.4

# Hail size bands (inches). Non-overlapping; label -> [min, max) -> RGBA.
# Alpha is baked light (map stays readable underneath); frontend adds a
# second zoom-interpolated opacity dial.
BANDS = [
    # (min_in, label,        (R,   G,   B,   A))  — alpha baked bold per user request
    (0.75, "0.75\u20131\"",  (96, 190, 110, 150)),
    (1.00, "1\u20131.25\"",  (255, 222, 66, 160)),
    (1.25, "1.25\u20131.5\"",(255, 180, 40, 170)),
    (1.50, "1.5\u20131.75\"",(255, 128, 24, 180)),
    (1.75, "1.75\u20132\"",  (250, 76, 28, 190)),
    (2.00, "2\u20132.5\"",   (222, 28, 28, 198)),
    (2.50, "2.5\u20133\"",   (168, 12, 60, 205)),
    (3.00, "3\u20134\"",     (128, 16, 128, 210)),
    (4.00, "4\"+",           (72, 8, 96, 215)),
]
CONTOUR_ALPHA = 245
SMOOTH_SIGMA = float(os.environ.get("SMOOTH_SIGMA", "2.2"))  # px; display smoothing only


# ----------------------------------------------------------------------------
# MESH recalibration (formulas cross-checked against pyhail / MH 2021)
# ----------------------------------------------------------------------------
WITT_A, WITT_B = 2.54, 0.5          # Witt et al. 1998 (what MRMS publishes)
MH_A, MH_B = 15.096, 0.206          # Murillo & Homeyer 2021, 75th percentile
SHI_INTERCEPT = (MH_A / WITT_A) ** (1.0 / (WITT_B - MH_B))  # ≈ 429.3
BLEND_WIDTH = 200.0


def recalibrate_mesh_mm(mesh_witt_mm: np.ndarray) -> np.ndarray:
    """MRMS MESH (Witt-calibrated, mm) -> blended Witt/MH2021-75 MESH (mm).

    Invert Witt to SHI, then logistic-blend: Witt at low SHI (it behaves
    fine for small hail), MH2021-75 at high SHI (fixes the upper-tail
    overestimation). Continuous and monotonic.
    """
    m = np.asarray(mesh_witt_mm, dtype=np.float32)
    flat_in = np.atleast_2d(m)
    out = np.empty_like(flat_in)
    k = np.float32(2.0 * math.log(9.0) / BLEND_WIDTH)
    step = 256  # row blocks: keeps temporaries tiny on CONUS grids
    for i in range(0, flat_in.shape[0], step):
        mm = flat_in[i:i + step]
        shi = np.square(mm / np.float32(WITT_A))          # SHI = (MESH/2.54)^2
        z = np.clip(-k * (shi - np.float32(SHI_INTERCEPT)), -50.0, 50.0)
        w = 1.0 / (1.0 + np.exp(z))
        mesh_mh = np.float32(MH_A) * np.power(shi, np.float32(MH_B),
                                              where=shi > 0, out=np.zeros_like(shi))
        blk = (1.0 - w) * mm + w * mesh_mh
        blk *= np.float32(REGIONAL_BIAS)
        out[i:i + step] = np.where(mm > 0.0, blk, 0.0)
    return out.reshape(m.shape).astype(np.float32)


def recal_scalar_inches(witt_in: float) -> float:
    """Recalibrate a single Witt-MESH value given in inches."""
    return float(recalibrate_mesh_mm(np.array([witt_in * MM_PER_INCH]))[0]) / MM_PER_INCH


# ----------------------------------------------------------------------------
# GRIB fetch + decode
# ----------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 90) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "StormDataPro/2.0"})
    r.raise_for_status()
    return r.content


def fetch_mesh_grid(url: str):
    """Fetch a MESH GRIB2(.gz) product. Returns (mm_grid, lat_max, lat_min,
    lon_min, lon_max, valid_time_iso). Grid is float32 mm, row 0 = north."""
    raw = _http_get(url)
    if url.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        grbs = pygrib.open(path)
        msg = grbs.message(1)
        v64 = msg.values  # float64 (possibly masked)
        vals = np.empty(v64.shape, dtype=np.float32)
        mask = np.ma.getmaskarray(v64) if np.ma.isMaskedArray(v64) else None
        data64 = v64.data if np.ma.isMaskedArray(v64) else v64
        step = 256  # convert in row blocks to avoid a full float64 copy
        for i in range(0, data64.shape[0], step):
            blk = data64[i:i + step].astype(np.float32)
            if mask is not None:
                blk[mask[i:i + step]] = 0.0
            blk[blk < 0] = 0.0  # -3 / -999 = no data
            vals[i:i + step] = blk
        del v64, data64, mask
        lat1 = float(msg["latitudeOfFirstGridPointInDegrees"])
        lat2 = float(msg["latitudeOfLastGridPointInDegrees"])
        lon1 = float(msg["longitudeOfFirstGridPointInDegrees"])
        lon2 = float(msg["longitudeOfLastGridPointInDegrees"])
        try:
            vt = msg.validDate.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            vt = datetime.now(timezone.utc).isoformat()
        grbs.close()
    finally:
        os.unlink(path)
    # normalize: row 0 north
    if lat1 < lat2:
        vals = vals[::-1]
        lat1, lat2 = lat2, lat1
    # normalize lon to [-180, 180]
    if lon1 > 180:
        lon1 -= 360
    if lon2 > 180:
        lon2 -= 360
    return vals, lat1, lat2, lon1, lon2, vt


# ----------------------------------------------------------------------------
# Ground truth: SPC + IEM LSR hail reports
# ----------------------------------------------------------------------------
def _parse_spc_csv(text: str) -> list:
    out = []
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return out
    header = lines[0].lower().split(",")
    try:
        i_size = header.index("size")
        i_lat = header.index("lat")
        i_lon = header.index("lon")
        i_loc = header.index("location") if "location" in header else -1
        i_time = header.index("time") if "time" in header else -1
    except ValueError:
        return out
    for line in lines[1:]:
        p = line.split(",")
        if len(p) <= max(i_size, i_lat, i_lon):
            continue
        try:
            size_in = float(p[i_size]) / 100.0
            lat, lon = float(p[i_lat]), float(p[i_lon])
        except ValueError:
            continue
        if size_in <= 0:
            continue
        out.append({
            "lat": lat, "lon": lon, "size_in": round(size_in, 2),
            "source": "SPC",
            "place": p[i_loc].title() if 0 <= i_loc < len(p) else "",
            "time": p[i_time] if 0 <= i_time < len(p) else "",
        })
    return out


def fetch_spc_reports() -> list:
    """SPC daily hail CSV -> list of report dicts. Sizes are in 1/100 inch."""
    for url in (SPC_URL, SPC_URL_FALLBACK):
        try:
            text = _http_get(url, timeout=30).decode("utf-8", "replace")
        except Exception as e:
            log.warning("SPC fetch failed %s: %s", url, e)
            continue
        out = _parse_spc_csv(text)
        if out:
            return out
    return []


def fetch_lsr_reports() -> list:
    """IEM Local Storm Reports geojson (last 24h) -> hail reports only."""
    out = []
    try:
        data = json.loads(_http_get(LSR_URL, timeout=30))
    except Exception as e:
        log.warning("LSR fetch failed: %s", e)
        return out
    for f in data.get("features", []):
        props = f.get("properties", {})
        if str(props.get("typetext", "")).upper() != "HAIL":
            continue
        try:
            size_in = float(props.get("magnitude") or 0)
        except (TypeError, ValueError):
            continue
        if size_in <= 0:
            continue
        coords = f.get("geometry", {}).get("coordinates", [None, None])
        if coords[0] is None:
            continue
        out.append({
            "lat": float(coords[1]), "lon": float(coords[0]),
            "size_in": round(size_in, 2), "source": "LSR",
            "place": str(props.get("city") or props.get("county") or "").title(),
            "time": str(props.get("valid") or ""),
        })
    return out


def dedupe_reports(reports: list) -> list:
    """Drop near-duplicate reports (same size within ~2 km). SPC wins."""
    kept = []
    for r in sorted(reports, key=lambda x: 0 if x["source"] == "SPC" else 1):
        dup = False
        for k in kept:
            if (abs(r["lat"] - k["lat"]) < 0.02 and abs(r["lon"] - k["lon"]) < 0.02
                    and abs(r["size_in"] - k["size_in"]) < 0.13):
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


# ----------------------------------------------------------------------------
# Ground-truth fusion: cap swath components that ground reports contradict
# ----------------------------------------------------------------------------
def fuse_ground_truth(mesh_in: np.ndarray, lat_n: float, lon_w: float,
                      dlat: float, dlon: float, reports: list):
    """mesh_in: calibrated grid in INCHES (row 0 north). Modifies in place.

    For each connected swath component: collect ground reports inside it.
    If >= GT_MIN_REPORTS reports exist and the radar peak exceeds the largest
    ground report by more than GT_CAP_MARGIN, cap the component. Returns
    fusion stats and per-report verification flags.
    """
    stats = {"components": 0, "capped": 0, "verified": 0, "reports_in_swaths": 0}
    if not reports or mesh_in.max() <= 0:
        return stats

    ny, nx = mesh_in.shape
    # label on a coarse mask (5x max-pool) to keep labeling cheap on CONUS grids
    f = 5
    py, px = (-ny) % f, (-nx) % f
    if py or px:  # only copy when grid doesn't divide evenly (never for MRMS CONUS)
        padded = np.pad(mesh_in, ((0, py), (0, px)))
    else:
        padded = mesh_in
    coarse = padded.reshape(padded.shape[0] // f, f, padded.shape[1] // f, f).max(axis=(1, 3))
    labels, ncomp = ndimage.label(coarse >= MIN_INCH,
                                  structure=np.ones((3, 3), dtype=int))
    stats["components"] = int(ncomp)
    if ncomp == 0:
        return stats

    # map reports to coarse cells
    comp_reports = {}
    for r in reports:
        gy = int((lat_n - r["lat"]) / dlat / f)
        gx = int((r["lon"] - lon_w) / dlon / f)
        if 0 <= gy < labels.shape[0] and 0 <= gx < labels.shape[1]:
            lbl = 0
            # tolerate small offset: check 3x3 coarse neighborhood
            for oy in (0, -1, 1):
                for ox in (0, -1, 1):
                    yy, xx = gy + oy, gx + ox
                    if 0 <= yy < labels.shape[0] and 0 <= xx < labels.shape[1] and labels[yy, xx]:
                        lbl = labels[yy, xx]
                        break
                if lbl:
                    break
            if lbl:
                comp_reports.setdefault(int(lbl), []).append(r)
                r["in_swath"] = True
                stats["reports_in_swaths"] += 1

    for lbl, rs in comp_reports.items():
        gt_max = max(r["size_in"] for r in rs)
        mask_coarse = labels == lbl
        mask = np.repeat(np.repeat(mask_coarse, f, axis=0), f, axis=1)[:ny, :nx]
        radar_peak = float(mesh_in[mask].max()) if mask.any() else 0.0
        cap = gt_max * GT_CAP_MARGIN
        if len(rs) >= GT_MIN_REPORTS and radar_peak > cap:
            np.minimum(mesh_in, np.where(mask, cap, np.inf), out=mesh_in)
            stats["capped"] += 1
            for r in rs:
                r["verified_component"] = True
        else:
            stats["verified"] += 1
            for r in rs:
                r["verified_component"] = True
    return stats


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_heatmap(mesh_in: np.ndarray, lat_n: float, lat_s: float,
                   lon_w: float, lon_e: float):
    """Calibrated inches grid -> (filled_png, contour_png, bounds, stats).

    Both styles render from a lightly gaussian-smoothed copy so swath shapes
    read as clean organic polygons instead of raw radar pixels:
      - filled: solid banded fills (the classic hail-swath look)
      - contour: nested size-threshold rings (the storm-analysis look)
    Reported max_in comes from the UNsmoothed grid — smoothing is display-only
    and must never change the headline number.
    """
    ny, nx = mesh_in.shape
    dlat = (lat_n - lat_s) / (ny - 1) if ny > 1 else 0.01
    dlon = (lon_e - lon_w) / (nx - 1) if nx > 1 else 0.01

    true_max = float(mesh_in.max()) if mesh_in.size else 0.0
    raw_mask = mesh_in >= MIN_INCH
    if not raw_mask.any():
        img = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        empty = buf.getvalue()
        return empty, empty, None, {"cells": 0, "max_in": 0.0}

    rows = np.flatnonzero(raw_mask.any(axis=1))
    cols = np.flatnonzero(raw_mask.any(axis=0))
    pad = 14
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, ny - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, nx - 1)
    crop = mesh_in[r0:r1 + 1, c0:c1 + 1]
    smooth = ndimage.gaussian_filter(crop, sigma=SMOOTH_SIGMA)
    # preserve peaks the blur would shave off: never display below the raw band
    smooth = np.maximum(smooth, crop)

    # --- filled style ---
    rgba = np.zeros((*smooth.shape, 4), dtype=np.uint8)
    for min_in, _label, color in BANDS:
        rgba[smooth >= min_in] = color
    rgba[smooth < MIN_INCH] = (0, 0, 0, 0)
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    filled_png = buf.getvalue()

    # --- contour style: 3px threshold rings, small->large so big sizes win ---
    rgba_c = np.zeros((*smooth.shape, 4), dtype=np.uint8)
    for min_in, _label, color in BANDS:
        mask = smooth >= min_in
        if not mask.any():
            continue
        ring = mask & ~ndimage.binary_erosion(mask, iterations=3, border_value=0)
        rgba_c[ring] = (color[0], color[1], color[2], CONTOUR_ALPHA)
    img_c = Image.fromarray(rgba_c, "RGBA")
    buf_c = io.BytesIO()
    img_c.save(buf_c, format="PNG", optimize=True)
    contour_png = buf_c.getvalue()

    bounds = {  # [west, south, east, north] of the cropped image edges
        "west": lon_w + (c0 - 0.5) * dlon,
        "east": lon_w + (c1 + 0.5) * dlon,
        "north": lat_n - (r0 - 0.5) * dlat,
        "south": lat_n - (r1 + 0.5) * dlat,
    }
    stats = {
        "cells": int(raw_mask.sum()),
        "max_in": round(true_max, 2),
        "size_px": [int(smooth.shape[1]), int(smooth.shape[0])],
    }
    return filled_png, contour_png, bounds, stats


def build_meta(bounds, stats, valid_time, fusion_stats, layer):
    return {
        "bounds": bounds,
        "size": stats.get("size_px"),
        "max_in": stats.get("max_in", 0.0),
        "hail_cells": stats.get("cells", 0),
        "valid_time": valid_time,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "min_inch": MIN_INCH,
        "calibration": {
            "scheme": "witt1998_mh2021_blend",
            "regional_bias": REGIONAL_BIAS,
            "note": "MRMS Witt MESH inverted to SHI, logistic blend into "
                    "Murillo-Homeyer 2021 75th pct at high SHI",
        },
        "ground_truth": fusion_stats or {},
        "bands": [{"min_in": b[0], "label": b[1],
                   "color": "#%02x%02x%02x" % b[2][:3]} for b in BANDS
                  if b[0] >= MIN_INCH or b[0] == BANDS[0][0]],
        "version": VERSION,
    }


def reports_geojson(reports: list) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "size_in": r["size_in"], "source": r["source"],
                "place": r.get("place", ""), "time": r.get("time", ""),
                "in_swath": bool(r.get("in_swath")),
                "verified": bool(r.get("verified_component")),
            },
        } for r in reports],
    }


# ----------------------------------------------------------------------------
# Supabase Storage backup
# ----------------------------------------------------------------------------
def sb_enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def sb_upload(name: str, data: bytes, content_type: str):
    if not sb_enabled():
        return
    try:
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{name}",
            headers={"Authorization": f"Bearer {SUPABASE_KEY}",
                     "Content-Type": content_type, "x-upsert": "true"},
            data=data, timeout=60)
        if r.status_code >= 300:
            log.warning("Supabase upload %s -> %s %s", name, r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Supabase upload %s failed: %s", name, e)


def sb_download(name: str):
    if not sb_enabled():
        return None
    try:
        r = requests.get(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{name}",
            headers={"Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=60)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        log.warning("Supabase download %s failed: %s", name, e)
    return None


def restore_archive_file(fname: str) -> bool:
    """Try to restore one archive file from Supabase to disk."""
    path = os.path.join(ARCHIVE_DIR, fname)
    if os.path.exists(path):
        return True
    data = sb_download(f"archive/{fname}")
    if data is None:
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


# ----------------------------------------------------------------------------
# Processing cycle
# ----------------------------------------------------------------------------
STATE = {
    "last_cycle": None, "last_error": None, "cycles": 0,
    "last_archive_date": None, "boot": datetime.now(timezone.utc).isoformat(),
}
_lock = threading.Lock()


def _atomic_write(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def process_layer(url: str, layer: str, reports: list):
    """Fetch one MESH product, calibrate, fuse, render, write to LIVE_DIR."""
    grid_mm, lat_n, lat_s, lon_w, lon_e, vt = fetch_mesh_grid(url)
    ny, nx = grid_mm.shape
    dlat = (lat_n - lat_s) / (ny - 1) if ny > 1 else 0.01
    dlon = (lon_e - lon_w) / (nx - 1) if nx > 1 else 0.01

    mesh_in = recalibrate_mesh_mm(grid_mm) / MM_PER_INCH
    del grid_mm

    fusion = None
    if layer == "day":  # fuse ground truth on the 24h layer only
        fusion = fuse_ground_truth(mesh_in, lat_n, lon_w, dlat, dlon, reports)

    filled_png, contour_png, bounds, stats = render_heatmap(mesh_in, lat_n, lat_s, lon_w, lon_e)
    del mesh_in
    meta = build_meta(bounds, stats, vt, fusion, layer)
    meta["styles"] = ["filled", "contour"]

    prefix = "heatmap" if layer == "day" else "heatmap_hour"
    _atomic_write(os.path.join(LIVE_DIR, f"{prefix}.png"), filled_png)
    _atomic_write(os.path.join(LIVE_DIR, f"{prefix}_contour.png"), contour_png)
    _atomic_write(os.path.join(LIVE_DIR, f"{prefix}_meta.json"),
                  json.dumps(meta).encode())
    log.info("[%s] %s cells, max %.2f\", png %dKB",
             layer, stats["cells"], stats["max_in"], len(filled_png) // 1024)
    return meta


def archive_snapshot(date_str: str):
    """Copy current live 24h layer + reports into the archive as `date_str`."""
    copied = []
    for src, dst in [
        ("heatmap.png", f"{date_str}_heatmap.png"),
        ("heatmap_contour.png", f"{date_str}_heatmap_contour.png"),
        ("heatmap_meta.json", f"{date_str}_heatmap_meta.json"),
        ("reports.geojson", f"{date_str}_reports.geojson"),
    ]:
        s = os.path.join(LIVE_DIR, src)
        d = os.path.join(ARCHIVE_DIR, dst)
        if os.path.exists(s):
            shutil.copyfile(s, d)
            copied.append(dst)
            with open(d, "rb") as f:
                ct = "image/png" if dst.endswith(".png") else "application/json"
                sb_upload(f"archive/{dst}", f.read(), ct)
    log.info("archived %s: %s", date_str, copied)


def cycle():
    reports = dedupe_reports(fetch_spc_reports() + fetch_lsr_reports())
    meta_day = process_layer(MESH_1440_URL, "day", reports)
    _atomic_write(os.path.join(LIVE_DIR, "reports.geojson"),
                  json.dumps(reports_geojson(reports)).encode())
    try:
        process_layer(MESH_60_URL, "hour", reports)
    except Exception as e:
        log.warning("hour layer failed (non-fatal): %s", e)

    # local-midnight archive: when the local date advances past the last
    # archived date, snapshot what we have as YESTERDAY's map.
    today_local = datetime.now(LOCAL_TZ).date()
    last = STATE["last_archive_date"]
    marker = os.path.join(DATA_DIR, "last_archive_date.txt")
    if last is None and os.path.exists(marker):
        last = open(marker).read().strip() or None
        STATE["last_archive_date"] = last
    if last != str(today_local):
        if last is not None:  # don't archive on very first boot ever
            archive_snapshot(str(today_local - timedelta(days=1)))
        STATE["last_archive_date"] = str(today_local)
        with open(marker, "w") as f:
            f.write(str(today_local))

    STATE["last_cycle"] = datetime.now(timezone.utc).isoformat()
    STATE["last_error"] = None
    STATE["cycles"] += 1
    return meta_day


def backfill_date(target_date) -> bool:
    """Rebuild one past local day into the archive from the IEM MRMS mirror.

    The MESH_Max_1440min product stamped ~06:00 UTC the NEXT morning covers
    roughly local-midnight-to-midnight (America/Chicago) of `target_date`.
    Reports come from SPC's dated CSV for that day. Runs on boot so the
    Yesterday view is populated even on a first-ever deploy.
    """
    ds = str(target_date)
    if os.path.exists(os.path.join(ARCHIVE_DIR, f"{ds}_heatmap_meta.json")):
        return True
    if restore_archive_file(f"{ds}_heatmap_meta.json"):  # Supabase already has it
        restore_archive_file(f"{ds}_heatmap.png")
        restore_archive_file(f"{ds}_heatmap_contour.png")
        restore_archive_file(f"{ds}_reports.geojson")
        return True

    nxt = target_date + timedelta(days=1)
    grid = None
    for hhmmss in ("060000", "060200", "055800", "060400", "055600"):
        url = (f"{IEM_MESH_BASE}/{nxt.year}/{nxt.month:02d}/{nxt.day:02d}"
               f"/mrms/ncep/MESH_Max_1440min/"
               f"MESH_Max_1440min_00.50_{nxt.year}{nxt.month:02d}{nxt.day:02d}"
               f"-{hhmmss}.grib2.gz")
        try:
            grid = fetch_mesh_grid(url)
            break
        except Exception:
            continue
    if grid is None:
        log.warning("backfill %s: no IEM archive file found", ds)
        return False

    vals, lat_n, lat_s, lon_w, lon_e, vt = grid
    ny, nx = vals.shape
    dlat = (lat_n - lat_s) / (ny - 1) if ny > 1 else 0.01
    dlon = (lon_e - lon_w) / (nx - 1) if nx > 1 else 0.01
    mesh_in = recalibrate_mesh_mm(vals) / MM_PER_INCH
    del vals

    yymmdd = target_date.strftime("%y%m%d")
    reports = []
    for suffix in ("_rpts_raw_hail.csv", "_rpts_hail.csv"):
        try:
            text = _http_get(f"{SPC_DATED_BASE}/{yymmdd}{suffix}", timeout=30
                             ).decode("utf-8", "replace")
            reports = dedupe_reports(_parse_spc_csv(text))
            if reports:
                break
        except Exception:
            continue

    fusion = fuse_ground_truth(mesh_in, lat_n, lon_w, dlat, dlon, reports)
    filled_png, contour_png, bounds, stats = render_heatmap(
        mesh_in, lat_n, lat_s, lon_w, lon_e)
    del mesh_in
    meta = build_meta(bounds, stats, vt, fusion, "day")
    meta["styles"] = ["filled", "contour"]
    meta["backfilled"] = True

    files = {
        f"{ds}_heatmap.png": (filled_png, "image/png"),
        f"{ds}_heatmap_contour.png": (contour_png, "image/png"),
        f"{ds}_heatmap_meta.json": (json.dumps(meta).encode(), "application/json"),
        f"{ds}_reports.geojson": (json.dumps(reports_geojson(reports)).encode(),
                                  "application/json"),
    }
    for name, (data, ct) in files.items():
        _atomic_write(os.path.join(ARCHIVE_DIR, name), data)
        sb_upload(f"archive/{name}", data, ct)
    log.info("backfilled %s: max %.2f\", %d reports", ds, stats["max_in"], len(reports))
    return True


def worker():
    try:  # populate Yesterday before the first live cycle (fresh deploys)
        backfill_date(datetime.now(LOCAL_TZ).date() - timedelta(days=1))
    except Exception as e:
        log.warning("yesterday backfill failed: %s", e)
    while True:
        try:
            with _lock:
                cycle()
        except Exception as e:
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            log.error("cycle failed: %s\n%s", e, traceback.format_exc())
        time.sleep(FETCH_INTERVAL)


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="StormDataPro Hail Maps", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
def _startup():
    threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok" if STATE["last_error"] is None else "degraded",
            "version": VERSION, **STATE,
            "supabase": sb_enabled(), "min_inch": MIN_INCH,
            "regional_bias": REGIONAL_BIAS}


def _serve_file(path: str, media: str, max_age=120):
    if not os.path.exists(path):
        raise HTTPException(404, "Not generated yet — first cycle runs ~60s after boot")
    return FileResponse(path, media_type=media,
                        headers={"Cache-Control": f"public, max-age={max_age}"})


def _serve_json(path: str, empty):
    if not os.path.exists(path):
        return JSONResponse(empty)
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.get("/swaths/heatmap.png")
def live_heatmap():
    return _serve_file(os.path.join(LIVE_DIR, "heatmap.png"), "image/png")


@app.get("/swaths/heatmap_contour.png")
def live_heatmap_contour():
    return _serve_file(os.path.join(LIVE_DIR, "heatmap_contour.png"), "image/png")


@app.get("/swaths/heatmap/meta")
def live_meta():
    return _serve_json(os.path.join(LIVE_DIR, "heatmap_meta.json"),
                       {"bounds": None, "note": "first cycle pending"})


@app.get("/hour/heatmap.png")
def hour_heatmap():
    return _serve_file(os.path.join(LIVE_DIR, "heatmap_hour.png"), "image/png")


@app.get("/hour/heatmap/meta")
def hour_meta():
    return _serve_json(os.path.join(LIVE_DIR, "heatmap_hour_meta.json"),
                       {"bounds": None, "note": "first cycle pending"})


@app.get("/reports")
def live_reports():
    return _serve_json(os.path.join(LIVE_DIR, "reports.geojson"),
                       {"type": "FeatureCollection", "features": []})


@app.get("/history/summaries")
def history_summaries():
    """One row per archived day with headline stats for the calendar UI."""
    days = {}
    for fn in os.listdir(ARCHIVE_DIR):
        if fn.endswith("_heatmap_meta.json"):
            date = fn.split("_heatmap_meta")[0]
            try:
                with open(os.path.join(ARCHIVE_DIR, fn)) as f:
                    m = json.load(f)
                days[date] = {"date": date, "max_in": m.get("max_in", 0),
                              "hail_cells": m.get("hail_cells", 0),
                              "reports": (m.get("ground_truth") or {})
                              .get("reports_in_swaths", 0)}
            except Exception:
                days[date] = {"date": date, "max_in": 0, "hail_cells": 0}
    return {"days": sorted(days.values(), key=lambda d: d["date"], reverse=True)}


@app.get("/history/{date_str}/heatmap.png")
def history_heatmap(date_str: str):
    fn = f"{date_str}_heatmap.png"
    if not restore_archive_file(fn):
        raise HTTPException(404, f"No archived heatmap for {date_str}")
    return _serve_file(os.path.join(ARCHIVE_DIR, fn), "image/png", max_age=86400)


@app.get("/history/{date_str}/heatmap_contour.png")
def history_heatmap_contour(date_str: str):
    fn = f"{date_str}_heatmap_contour.png"
    if not restore_archive_file(fn):
        raise HTTPException(404, f"No archived contour map for {date_str}")
    return _serve_file(os.path.join(ARCHIVE_DIR, fn), "image/png", max_age=86400)


@app.get("/history/{date_str}/heatmap/meta")
def history_meta(date_str: str):
    fn = f"{date_str}_heatmap_meta.json"
    if not restore_archive_file(fn):
        return JSONResponse({"bounds": None,
                             "note": f"No archived heatmap for {date_str}"})
    return _serve_json(os.path.join(ARCHIVE_DIR, fn), {"bounds": None})


@app.get("/history/{date_str}/reports")
def history_reports(date_str: str):
    fn = f"{date_str}_reports.geojson"
    restore_archive_file(fn)
    return _serve_json(os.path.join(ARCHIVE_DIR, fn),
                       {"type": "FeatureCollection", "features": []})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
