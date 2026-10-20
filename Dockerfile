FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# HTTP and LLM caches live here; docker compose mounts the repo's .cache, so local and
# container runs share one cache (and a rehearsed demo replays in either).
ENV CACHE_DIR=/srv/cache
# The SQLite database (accounts, profiles, saved trips) lives on a volume mounted here,
# so it survives rebuilds of the image.
ENV DATA_DIR=/srv/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
