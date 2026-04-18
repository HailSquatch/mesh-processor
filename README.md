# StormDataPro MESH Processor

Decodes real MRMS MESH GRIB2 files (including PNG-packed template 5.41) into hail data
for the StormDataPro hail map. Phase 1 — proves decoding works; later phases add
accumulation, polygonization, and storage.

## Endpoints

- `GET /` — service info
- `GET /health` — liveness check
- `GET /test` — fetches latest MESH from NOAA, decodes it, returns summary stats
- `GET /test-points` — same but returns top 500 hail points as GeoJSON

## Deploy on Railway

1. Fork/upload this repo to GitHub
2. In Railway, New Project → Deploy from GitHub → pick this repo
3. Wait ~5 min for Docker build (first time takes longer due to GDAL/eccodes)
4. Generate a public domain (Settings → Networking → Generate Domain)
5. Hit `/test` — should see a JSON response with decoded hail grid stats

## Local development (optional, requires Docker)

```
docker build -t mesh-processor .
docker run -p 8080:8080 mesh-processor
curl localhost:8080/test
```

## Costs

- Railway: ~$5-10/month for this service (512MB RAM, sleeps when idle)
- NOAA MRMS data: free
