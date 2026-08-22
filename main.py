import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config import FRONTEND_URL, SENTRY_DSN

# init() quietly does nothing until SENTRY_DSN is actually set — safe to
# deploy either way. Once set, auto-captures unhandled exceptions (the raw
# 500s that, until now, only ever showed up as a print() in a console
# nobody was watching in real time). Must run before the FastAPI app is
# created. capture_exception() calls elsewhere in this codebase (for
# exceptions that are already caught and handled, not just left to crash)
# are themselves safe no-ops if init() was never called — that's why the
# import above stays unconditional even though init() is not.
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=0.1,
        send_default_pii=False,
    )

# Import all routers
from usage import router as usage_router
from chat import router as chat_router
from ingest import router as ingest_router
from source_routes import router as source_router
from leads import router as leads_router
from telegram import router as telegram_router
from slack import router as slack_router
from whatsapp import router as whatsapp_router
from billing import router as billing_router
from flows import router as flows_router
from api_keys import router as api_keys_router
from campaigns import router as campaigns_router
from template_library import router as template_library_router
from shop import router as shop_router
from send_template_api import router as send_template_router
from appointments import router as appointments_router
from events import router as events_router
from content_gaps import router as content_gaps_router
from auth import router as auth_router
from shopify_oauth import router as shopify_router
from razorpay_oauth import router as razorpay_oauth_router


# -------------------------------------------------
# BACKGROUND SCHEDULER — appointment reminders + scheduled campaigns
# -------------------------------------------------
def _record_job_run(job_name: str, started_at, status: str, detail: dict = None):
    """Best-effort write to job_runs, powering the admin System Health tab's
    last-run/success view. Never allowed to break the job itself — a failed
    write here is swallowed, not raised."""
    try:
        from datetime import datetime, timezone
        from clients import supabase
        supabase.table("job_runs").insert({
            "job_name": job_name,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "detail": detail or {},
        }).execute()
    except Exception:
        pass


def run_appointment_reminders():
    """Runs every hour — sends WhatsApp reminders for upcoming appointments."""
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc)
    try:
        from appointments import send_reminders_job
        send_reminders_job()
        _record_job_run("appointment_reminders", started_at, "success")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: reminder job error: {e}")
        _record_job_run("appointment_reminders", started_at, "failure", {"error": str(e)})


def run_scheduled_campaigns():
    """Runs every 30 seconds — sends any campaign whose scheduled time has arrived."""
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc)
    try:
        from campaigns import dispatch_scheduled_campaigns
        dispatch_scheduled_campaigns()
        _record_job_run("scheduled_campaigns", started_at, "success")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: campaign dispatch error: {e}")
        _record_job_run("scheduled_campaigns", started_at, "failure", {"error": str(e)})


def run_release_expired_holds():
    """Runs every 2 minutes — releases unpaid 'hold_to_confirm' appointment
    bookings whose hold window has expired, freeing the slot back up."""
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc)
    try:
        from appointments import release_expired_holds
        release_expired_holds()
        _record_job_run("appointment_hold_release", started_at, "success")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: hold-release job error: {e}")
        _record_job_run("appointment_hold_release", started_at, "failure", {"error": str(e)})


def run_shopify_reconciliation():
    """Runs every 6 hours — backstop for Shopify's product webhooks, which
    aren't 100% guaranteed to deliver (per Shopify's own docs). Webhooks are
    the fast primary path (see shopify_oauth.py); this just catches anything
    missed. One merchant's failure must never abort the loop for the rest."""
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc)
    failed_count = 0
    total_count = 0
    try:
        from clients import supabase, qdrant, embeddings
        from config import QDRANT_COLLECTION
        from sources.shopify import sync_products

        integrations = supabase.table("shopify_integrations").select("project_id").execute()
        for integration in (integrations.data or []):
            project_id = integration["project_id"]
            total_count += 1
            try:
                source_res = supabase.table("data_sources").select("id").eq("project_id", project_id).eq("type", "shopify").maybe_single().execute()
                source_id = (source_res.data or {}).get("id") if source_res else None
                if source_id:
                    sync_products(project_id, source_id, qdrant, embeddings, QDRANT_COLLECTION)
            except Exception as e:
                failed_count += 1
                sentry_sdk.capture_exception(e)
                print(f"Scheduler: Shopify reconciliation error for project {project_id}: {e}")
        _record_job_run("shopify_reconciliation", started_at, "success" if failed_count == 0 else "failure",
                         {"total": total_count, "failed": failed_count})
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: Shopify reconciliation job error: {e}")
        _record_job_run("shopify_reconciliation", started_at, "failure", {"error": str(e)})


def run_whatsapp_sync_monitor():
    """Runs every 30 minutes — surfaces WhatsApp Coexistence history syncs
    that have been stuck in 'pending'/'in_progress' for a while. Meta gives
    a hard 24-hour window to complete a sync or the client must be fully
    offboarded and redo signup — this is the manual-intervention buffer
    before that cliff, since there's no on-call/paging setup here, only
    the admin System Health tab this surfaces into."""
    from datetime import datetime, timezone, timedelta
    started_at = datetime.now(timezone.utc)
    try:
        from clients import supabase
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        stalled = supabase.table("whatsapp_integrations") \
            .select("project_id, history_sync_status, history_sync_requested_at") \
            .in_("history_sync_status", ["pending", "in_progress"]) \
            .lt("history_sync_requested_at", cutoff) \
            .execute()
        count = len(stalled.data or [])
        _record_job_run("whatsapp_sync_monitor", started_at, "success" if count == 0 else "failure",
                         {"stalled_count": count})
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: WhatsApp sync monitor error: {e}")
        _record_job_run("whatsapp_sync_monitor", started_at, "failure", {"error": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on app startup
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_appointment_reminders, 'interval', hours=1, id='appointment_reminders')
        scheduler.add_job(run_scheduled_campaigns, 'interval', seconds=30, id='scheduled_campaigns')
        scheduler.add_job(run_shopify_reconciliation, 'interval', hours=6, id='shopify_reconciliation')
        scheduler.add_job(run_release_expired_holds, 'interval', minutes=2, id='appointment_hold_release')
        scheduler.add_job(run_whatsapp_sync_monitor, 'interval', minutes=30, id='whatsapp_sync_monitor')
        scheduler.start()
        print("Schedulers started — appointment reminders hourly, campaign dispatch every 30s, Shopify reconciliation every 6h, appointment hold release every 2m, WhatsApp sync monitor every 30m")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler failed to start: {e}")

    yield  # app runs here

    # Shutdown scheduler on app stop
    try:
        scheduler.shutdown()
    except Exception:
        pass


# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS is scoped by path rather than applied globally. The only routes ever
# called by real cross-origin browser JS are under /public/* (the embeddable
# chat widget, loaded on arbitrary merchant sites whose origin can't be
# known in advance — so that half stays a real wildcard). Every other route
# is only ever called server-to-server from the Next.js frontend's own API
# routes (not subject to browser CORS at all), so it gets a real allowlist
# instead of "*" — pure defense-in-depth, doesn't change any real traffic.
# allow_credentials is False on both: the backend never sets cookies (auth
# is Bearer-token/X-API-Key only), so there's no session for CORS to guard,
# and dropping it avoids Starlette's spec-mandated origin-reflection that
# kicks in when "*" is combined with allow_credentials=True.
class PathScopedCORSMiddleware:
    def __init__(self, app):
        self.public = CORSMiddleware(
            app,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        self.restricted = CORSMiddleware(
            app,
            allow_origins=[FRONTEND_URL],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/public/"):
            await self.public(scope, receive, send)
        else:
            await self.restricted(scope, receive, send)


app.add_middleware(PathScopedCORSMiddleware)

# -------------------------------------------------
# ROUTERS
# -------------------------------------------------
app.include_router(usage_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(source_router)
app.include_router(leads_router)
app.include_router(telegram_router)
app.include_router(slack_router)
app.include_router(whatsapp_router)
app.include_router(billing_router)
app.include_router(flows_router)
app.include_router(api_keys_router)
app.include_router(campaigns_router)
app.include_router(template_library_router)
app.include_router(shop_router)
app.include_router(send_template_router)
app.include_router(appointments_router)
app.include_router(events_router)
app.include_router(content_gaps_router)
app.include_router(auth_router)
app.include_router(shopify_router)
app.include_router(razorpay_oauth_router)


# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}