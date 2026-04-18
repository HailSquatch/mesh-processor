"""
StormDataPro MESH Processor — Phase 1
Decodes real MRMS MESH GRIB2 files (including PNG-packed template 5.41) into hail data.
"""
import os
import gzip
import tempfile
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pygrib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Setup ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="StormDataPro MESH Processor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MESH_URL = "https://mrms.ncep.noaa.gov/data/2D/MESH/MRMS_MESH.latest.grib2.gz"
USER_AGENT = "StormDataPro/0.1 (colton@transcendentpdr.com)"

# Thresholds for polygon generation (mm). 1.0" = 25.4mm
# Match IHM/HailTrace style — start at 1" (quarter-size)
SIZE_THRESHOLDS_MM = [25.4, 38.1, 50.8, 69.85]  # 1.0", 1.5", 2.0", 2.75"

# Hail size buckets in inches for color labels
SIZE_LABELS = {
    25.4: {"inches": 1.0, "desc": "Quarter", "color": "#eab308"},
    38.1: {"inches": 1.5, "desc": "Walnut", "color": "#f97316"},
    50.8: {"inches": 2.0, "desc": "Golf Ball", "color": "#ef4444"},
    69.85: {"inches": 2.75, "desc": "Baseball", "color": "#a855f7"},
}


def download_mesh() -> Optional[bytes]:
    """Fetch the latest MRMS MESH GRIB2 file, decompress, return raw bytes."""
    log.info(f"Fetching {MESH_URL}")
    try:
        r = requests.get(MESH_URL, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        log.info(f"Downloaded {len(r.content)} bytes (final URL: {r.url})")
        decompressed = gzip.decompress(r.content)
        log.info(f"Decompressed to {len(decompressed)} bytes")
        return decompressed
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def decode_mesh(grib_bytes: bytes) -> Optional[dict]:
    """
    Decode MRMS MESH GRIB2 using pygrib/eccodes.
    Returns dict with values (2D numpy array in mm), lats, lons, and metadata.
    """
    # pygrib needs a file, not bytes — write to temp
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        grbs = pygrib.open(tmp_path)
        messages = list(grbs)
        log.info(f"GRIB contains {len(messages)} messages")
        if not messages:
            return None

        grb = messages[0]  # MESH has one message per file
        log.info(f"Message: {grb}")

        values = grb.values  # 2D masked array, mm
        lats, lons = grb.latlons()

        # MRMS uses -3 and -999 as missing-data sentinels; mask them
        values = np.where(values < 0, np.nan, values)

        valid_count = int(np.sum(~np.isnan(values)))
        hail_count = int(np.sum(values >= 25.4))  # >= 1 inch
        max_val = float(np.nanmax(values)) if valid_count > 0 else 0.0

        # Get timestamp from GRIB metadata
        try:
            ts = grb.validDate.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            ts = datetime.now(timezone.utc).isoformat()

        grbs.close()

        return {
            "values": values,
            "lats": lats,
            "lons": lons,
            "timestamp": ts,
            "shape": values.shape,
            "valid_pixels": valid_count,
            "hail_pixels_1in": hail_count,
            "max_mm": max_val,
            "max_inches": round(max_val / 25.4, 2),
        }
    except Exception as e:
        log.error(f"Decode failed: {e}", exc_info=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Endpoints ──

@app.get("/")
def root():
    return {
        "service": "StormDataPro MESH Processor",
        "version": "0.1.0",
        "endpoints": [
            "/health",
            "/test — fetch latest MESH, decode, return summary",
            "/test-points — same but return top 100 hail points as GeoJSON",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/test")
def test_decode():
    """
    Full end-to-end test: fetch latest MESH, decode, return summary stats.
    If this works, Phase 1 is proven and we can build the full pipeline.
    """
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "Could not fetch MESH file from NOAA")

    result = decode_mesh(grib_bytes)
    if not result:
        raise HTTPException(500, "Could not decode MESH GRIB2")

    return {
        "status": "success",
        "timestamp": result["timestamp"],
        "grid_shape": list(result["shape"]),
        "valid_pixels": result["valid_pixels"],
        "hail_pixels_1in_plus": result["hail_pixels_1in"],
        "max_hail_mm": round(result["max_mm"], 2),
        "max_hail_inches": result["max_inches"],
        "fetched_bytes": len(grib_bytes),
        "message": "✅ MESH decoding is working. PNG-packed GRIB2 successfully decoded." if result["valid_pixels"] > 0 else "⚠️ Decoded but grid appears empty",
    }


@app.get("/test-points")
def test_points():
    """
    Fetch, decode, and return the top hail points as GeoJSON for visual inspection.
    """
    grib_bytes = download_mesh()
    if not grib_bytes:
        raise HTTPException(500, "Could not fetch MESH file")

    result = decode_mesh(grib_bytes)
    if not result:
        raise HTTPException(500, "Could not decode MESH")

    values = result["values"]
    lats = result["lats"]
    lons = result["lons"]

    # Find all pixels with hail >= 1 inch, limit to top 500 by size
    mask = values >= 25.4
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return {
            "type": "FeatureCollection",
            "metadata": {
                "timestamp": result["timestamp"],
                "message": "No hail >= 1 inch detected anywhere in CONUS right now",
            },
            "features": [],
        }

    # Sort by size descending, take top 500 for lightweight payload
    sizes = values[mask]
    order = np.argsort(-sizes)[:500]

    features = []
    for i in order:
        ridx, cidx = indices[i]
        mm = float(values[ridx, cidx])
        inches = round(mm / 25.4, 2)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lons[ridx, cidx]), float(lats[ridx, cidx])],
            },
            "properties": {
                "sizeMM": round(mm, 1),
                "sizeInches": inches,
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "timestamp": result["timestamp"],
            "total_hail_pixels": int(mask.sum()),
            "shown_top": len(features),
            "max_inches": result["max_inches"],
        },
        "features": features,
    }
