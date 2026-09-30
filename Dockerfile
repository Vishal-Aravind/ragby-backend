# Image for the INDEXING WORKER (worker_main.py), deployed to Cloud Run.
# The main backend is not built from this: Render runs it natively from
# requirements.txt + Procfile. Same code, different entrypoint — the worker
# imports only the indexing modules, not main.py's routers or scheduler.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Headless Chromium for JavaScript-only websites (sources/browser_render.py).
# Installed here only — not in requirements.txt — so the 512MB main backend
# on Render never gets a browser. --with-deps adds the system libraries
# Chromium needs on this slim base image.
RUN pip install playwright==1.58.0 \
 && playwright install --with-deps --only-shell chromium
ENV PLAYWRIGHT_ENABLED=1

COPY . .

# One worker process, and Cloud Run is set to one request per instance
# (--concurrency 1): each indexing job gets the instance's full memory.
CMD exec uvicorn worker_main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
