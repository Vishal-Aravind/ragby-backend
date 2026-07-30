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
from stripe_handler import router as stripe_router
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


# -------------------------------------------------
# BACKGROUND SCHEDULER — appointment reminders + scheduled campaigns
# -------------------------------------------------
def run_appointment_reminders():
    """Runs every hour — sends WhatsApp reminders for upcoming appointments."""
    try:
        from appointments import send_reminders_job
        send_reminders_job()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: reminder job error: {e}")


def run_scheduled_campaigns():
    """Runs every 30 seconds — sends any campaign whose scheduled time has arrived."""
    try:
        from campaigns import dispatch_scheduled_campaigns
        dispatch_scheduled_campaigns()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: campaign dispatch error: {e}")


def run_shopify_reconciliation():
    """Runs every 6 hours — backstop for Shopify's product webhooks, which
    aren't 100% guaranteed to deliver (per Shopify's own docs). Webhooks are
    the fast primary path (see shopify_oauth.py); this just catches anything
    missed. One merchant's failure must never abort the loop for the rest."""
    try:
        from clients import supabase, qdrant, embeddings
        from config import QDRANT_COLLECTION
        from sources.shopify import sync_products

        integrations = supabase.table("shopify_integrations").select("project_id").execute()
        for integration in (integrations.data or []):
            project_id = integration["project_id"]
            try:
                source_res = supabase.table("data_sources").select("id").eq("project_id", project_id).eq("type", "shopify").maybe_single().execute()
                source_id = (source_res.data or {}).get("id") if source_res else None
                if source_id:
                    sync_products(project_id, source_id, qdrant, embeddings, QDRANT_COLLECTION)
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"Scheduler: Shopify reconciliation error for project {project_id}: {e}")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Scheduler: Shopify reconciliation job error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on app startup
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_appointment_reminders, 'interval', hours=1, id='appointment_reminders')
        scheduler.add_job(run_scheduled_campaigns, 'interval', seconds=30, id='scheduled_campaigns')
        scheduler.add_job(run_shopify_reconciliation, 'interval', hours=6, id='shopify_reconciliation')
        scheduler.start()
        print("Schedulers started — appointment reminders hourly, campaign dispatch every 30s, Shopify reconciliation every 6h")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # * needed for public shop page
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(stripe_router)
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


# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}