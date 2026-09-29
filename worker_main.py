"""Indexing worker — a separate service from main.py.

Deployed on its own (Cloud Run, see Dockerfile.worker) with its own memory
per job, so indexing a large sheet or crawling a site can never starve or
kill the main backend that answers chat and every webhook. Receives jobs
from Cloud Tasks (see jobs.py) and runs them to completion within the
request, which is how Cloud Run gives a job CPU for its whole duration.

Deliberately imports nothing from main.py: no scheduler, no webhooks, no
routers — just the indexing code.
"""
import hmac

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from config import WORKER_SECRET
from memlog import limit_malloc_arenas, mem_summary

limit_malloc_arenas()

app = FastAPI(title="Zavo indexing worker")


class Job(BaseModel):
    kind: str
    payload: dict


@app.post("/run")
def run(job: Job, x_worker_secret: str = Header(default="")):
    # The worker writes to every tenant's data with the service-role key,
    # so it only takes jobs from our own backend.
    if not WORKER_SECRET or not hmac.compare_digest(x_worker_secret, WORKER_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")
    from job_runner import run_job
    run_job(job.kind, job.payload)
    # 200 even when the job recorded a failure: a failed sheet or corrupt
    # file isn't fixed by retrying. Only a crash/timeout (no 200) makes
    # Cloud Tasks re-deliver.
    return {"status": "done"}


@app.get("/health")
def health():
    return {"status": "ok"}


print(f"[mem] worker loaded: {mem_summary()}")
