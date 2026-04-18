FROM python:3.11-slim

# System dependencies needed by pygrib (eccodes) and rasterio (GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes0 \
    libeccodes-dev \
    libgdal-dev \
    gdal-bin \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Railway injects PORT env var; fall back to 8080 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
