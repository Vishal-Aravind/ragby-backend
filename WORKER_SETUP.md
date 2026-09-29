# Indexing worker — Google Cloud setup

Indexing (documents, sheets, Excel, websites) runs on a separate Cloud Run
service so it never shares memory with chat and webhooks on Render.
Until this is set up, jobs run in a background thread on Render (same
behaviour, just on the smaller machine), so nothing breaks in the meantime.

Region used throughout: `asia-south1` (Mumbai).

## 1. One-time account setup (in the browser)
1. Create a Google Cloud account at https://console.cloud.google.com and a
   project (e.g. `zavo-prod`). Note its **Project ID**.
2. Enable billing on the project (card required; usage stays inside the
   monthly free tier at current volume).

## 2. Install the command-line tool
1. Install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install
2. In a terminal:
   ```
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudtasks.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

## 3. Worker secret + env file
Generate a random secret (shared by Render and the worker):
```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Create `backend/worker.env.yaml` (git-ignored, never commit it) with the
same values Render has, plus the secret:
```yaml
SUPABASE_URL: "..."
SUPABASE_SERVICE_ROLE_KEY: "..."
OPENAI_API_KEY: "..."
QDRANT_URL: "..."
QDRANT_API_KEY: "..."
QDRANT_COLLECTION: "..."
SENTRY_DSN: "..."
WORKER_SECRET: "the-generated-secret"
```

## 4. Deploy the worker (from the `backend` folder)
```
gcloud run deploy zavo-indexer --source . --region asia-south1 \
  --memory 2Gi --cpu 1 --concurrency 1 --max-instances 10 --timeout 1800 \
  --allow-unauthenticated --env-vars-file worker.env.yaml
```
- `--concurrency 1`: one job per instance, so each job gets the full 2GB.
- `--max-instances 10`: at most 10 jobs at once; extra jobs wait in the queue.
- `--allow-unauthenticated`: the worker checks `WORKER_SECRET` itself.

Note the **Service URL** it prints (`https://zavo-indexer-....run.app`).
Check it's up: open `<Service URL>/health` → `{"status":"ok"}`.

## 5. Create the queue
```
gcloud tasks queues create indexing --location asia-south1 \
  --max-concurrent-dispatches 10 --max-attempts 3 --min-backoff 30s
```

## 6. Service account for Render (lets Render add jobs to the queue)
```
gcloud iam service-accounts create zavo-enqueuer
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member serviceAccount:zavo-enqueuer@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role roles/cloudtasks.enqueuer
gcloud iam service-accounts keys create enqueuer-key.json \
  --iam-account zavo-enqueuer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
`enqueuer-key.json` is a secret (git-ignored). Delete it from disk after
step 7.

## 7. Render environment variables (ragby-backend → Environment)
| Key | Value |
|---|---|
| `WORKER_URL` | the Service URL from step 4 |
| `WORKER_SECRET` | the secret from step 3 |
| `GCP_PROJECT_ID` | your project ID |
| `GCP_SERVICE_ACCOUNT_JSON` | the full contents of `enqueuer-key.json` |
| `GCP_TASKS_LOCATION` | `asia-south1` (default, optional) |
| `GCP_TASKS_QUEUE` | `indexing` (default, optional) |

## 8. Check it works
Upload a document. Then:
- Render logs show `[jobs] queued ingest_file on Cloud Tasks`
- Cloud Run logs (zavo-indexer → Logs) show `[job] start ingest_file` … `[job] done`
- The file goes Indexing… → ✓ Added to AI Knowledge

If Render logs show `Cloud Tasks enqueue failed, running ... locally`, the
job still completed on Render; the log line says why the queue refused it.

## Updating the worker later
After changing indexing code, redeploy with the same command as step 4.
Render and the worker share the code, so deploy both.
