# StormDataPro MESH Processor — Phase 5

Multi-product MRMS hail swath pipeline with consensus scoring. Decodes GRIB2 from
NOAA MRMS (including PNG-packed template 5.41), accumulates hail over a 24-hour
rolling window, and generates filtered polygons using 4 independent indicators.

## MRMS products fetched

Every 2 minutes:
- **MESH** — Maximum Estimated Size of Hail (primary)
- **POSH** — Probability of Severe Hail (confidence filter)
- **Reflectivity_-20C** — echo intensity above freezing level
- **EchoTop_50** — 50 dBZ echo top height
- **VIL_Density** — vertically integrated liquid density

Hourly (sanity check):
- **MESHMax1440min** — NOAA's own 24-hour MESH accumulation

## Consensus scoring

Each pixel is scored 0-100 based on how many indicators agree it's hail:
- **+40** MESH ≥ threshold (1.0" / 1.5" / 2.0" / 2.75")
- **+30** POSH ≥ 50%
- **+20** Reflectivity_-20C ≥ 55 dBZ
- **+10** EchoTop_50 ≥ 40 kft (12.2 km)

Polygons are only drawn where score ≥ 50. This rejects radar artifacts that
would otherwise show as MESH spikes without supporting storm structure.

## Key endpoints

- `GET /` — service info and scoring weights
- `GET /health` — liveness check
- `GET /status` — full state: scheduler jobs, accumulator stats, polygonizer stats
- `GET /test-products` — fetches all 5 products, shows current max values
- `GET /swaths` — main output: filtered polygon GeoJSON
- `GET /accumulated?min_inches=1.0` — raw pixel points (with per-pixel attributes)
- `POST /admin/force-tick` — run a tick immediately
- `POST /admin/polygonize` — rebuild polygons immediately
- `POST /admin/sanity-check` — compare accumulator to NOAA's MESHMax1440min
- `POST /admin/reset` — clear accumulator (use carefully)

## Memory / Railway sizing

Phase 5 requires **Railway Pro plan (8GB RAM)** to handle large outbreaks reliably.
Normal storm days fit in ~300-500 MB, but state-sized events can peak near 600 MB
and continental outbreaks near 800 MB. Hobby plan (512 MB) will crash during the
biggest events — which are exactly the days the map matters most.

Persistent state (always loaded):
- 6 accumulator grids × 3500 × 7000 = ~320 MB

Peak during polygonize depends on the hail-pixel bounding box across the 24h window.

## Deploy on Railway

1. Push this repo to GitHub
2. Railway → New Project → Deploy from GitHub → pick this repo
3. Add a Volume mounted at `/data` (for accumulator persistence)
4. Set env var `RAILWAY_RUN_UID=0` (so the container can write to the volume)
5. Upgrade to Pro plan if you expect heavy use
6. Generate a public domain

## Local dev

```
docker build -t mesh-processor .
docker run -p 8080:8080 mesh-processor
curl localhost:8080/test
```

## Costs

- Railway Pro: ~$20-30/month total for this service
- NOAA MRMS data: free
