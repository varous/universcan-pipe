# scanpipe service image. The apt line is the whole reason this is Dockerized:
# Open3D needs native libraries (libGL, OpenMP) that bare pip can't supply.
FROM python:3.11-slim

# --- Open3D / rendering native deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libusb-1.0-0 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_DIR=/data \
    MONGO_DB=scanpipe \
    PYTHONUNBUFFERED=1
# MONGO_URI is supplied at runtime (Atlas SRV string or compose service name)

VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1