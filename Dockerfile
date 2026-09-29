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

COPY . .

# One worker process, and Cloud Run is set to one request per instance
# (--concurrency 1): each indexing job gets the instance's full memory.
CMD exec uvicorn worker_main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
