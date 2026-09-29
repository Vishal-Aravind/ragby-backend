"""Hand heavy indexing work to the worker service.

Indexing (documents, sheets, Excel, websites) is the one memory-heavy thing
this backend does, and running it inside the same 512MB process as chat and
every webhook kept getting the instance killed — first website crawling
(Playwright had to go), then large sheets. It now runs as a job on a
separate worker (worker_main.py, deployed as its own Cloud Run service with
its own memory per job). This process only enqueues.

Queue: Google Cloud Tasks. It holds jobs, dispatches a bounded number at a
time to the worker, and retries a job whose worker died mid-way.

Fallback: when Cloud Tasks isn't configured (local dev, or before the
worker is deployed) or enqueueing fails, the job runs here in a background
thread — one at a time, so it can't multiply memory use. Behaviour is the
same either way; only where it runs differs.
"""
import base64
import json
import threading
import time

import jwt
import requests
import sentry_sdk

from config import (
    WORKER_URL, WORKER_SECRET, GCP_PROJECT_ID, GCP_TASKS_LOCATION,
    GCP_TASKS_QUEUE, GCP_SERVICE_ACCOUNT_JSON,
)

# Jobs on the worker can run for minutes (a 5,000-row sheet, a large site
# crawl); Cloud Tasks' own ceiling for an HTTP target is 30 minutes.
_DISPATCH_DEADLINE = "1800s"

_token = {"value": None, "expires_at": 0}
_token_lock = threading.Lock()


def _cloud_tasks_configured() -> bool:
    return all([WORKER_URL, WORKER_SECRET, GCP_PROJECT_ID, GCP_TASKS_LOCATION,
                GCP_TASKS_QUEUE, GCP_SERVICE_ACCOUNT_JSON])


def _access_token() -> str:
    """OAuth token for the Cloud Tasks API from the service account key,
    signed with PyJWT (already a dependency) rather than pulling in the
    google-cloud client libraries, which would add tens of MB to this
    process — the very thing this module exists to avoid."""
    with _token_lock:
        if _token["value"] and time.time() < _token["expires_at"] - 60:
            return _token["value"]
        sa = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        now = int(time.time())
        assertion = jwt.encode(
            {
                "iss": sa["client_email"],
                "scope": "https://www.googleapis.com/auth/cloud-platform",
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600,
            },
            sa["private_key"],
            algorithm="RS256",
        )
        res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            timeout=15,
        )
        res.raise_for_status()
        body = res.json()
        _token["value"] = body["access_token"]
        _token["expires_at"] = now + int(body.get("expires_in", 3600))
        return _token["value"]


def _enqueue_cloud_task(kind: str, payload: dict):
    body = json.dumps({"kind": kind, "payload": payload}).encode()
    queue = f"projects/{GCP_PROJECT_ID}/locations/{GCP_TASKS_LOCATION}/queues/{GCP_TASKS_QUEUE}"
    res = requests.post(
        f"https://cloudtasks.googleapis.com/v2/{queue}/tasks",
        headers={"Authorization": f"Bearer {_access_token()}"},
        json={"task": {
            "dispatchDeadline": _DISPATCH_DEADLINE,
            "httpRequest": {
                "httpMethod": "POST",
                "url": f"{WORKER_URL.rstrip('/')}/run",
                "headers": {"Content-Type": "application/json", "X-Worker-Secret": WORKER_SECRET},
                "body": base64.b64encode(body).decode(),
            },
        }},
        timeout=15,
    )
    res.raise_for_status()


# One at a time: the fallback runs on this 512MB instance, so two large
# jobs at once is exactly what used to get it killed. Extra jobs wait.
_local_lock = threading.Lock()


def _run_locally(kind: str, payload: dict):
    def _target():
        from job_runner import run_job
        with _local_lock:
            run_job(kind, payload)
    threading.Thread(target=_target, daemon=True, name=f"job-{kind}").start()


def enqueue(kind: str, payload: dict):
    """Start an indexing job; returns immediately. The job records its own
    outcome (files.status/error/result, data_sources.config.sync_*), which
    the dashboard polls."""
    if _cloud_tasks_configured():
        try:
            _enqueue_cloud_task(kind, payload)
            print(f"[jobs] queued {kind} on Cloud Tasks")
            return
        except Exception as e:
            # The queue being unreachable must not lose the job.
            sentry_sdk.capture_exception(e)
            print(f"[jobs] Cloud Tasks enqueue failed, running {kind} locally: {e}")
    _run_locally(kind, payload)
