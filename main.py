from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config import FRONTEND_URL

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


# -------------------------------------------------
# BACKGROUND SCHEDULER — appointment reminders + scheduled campaigns
# -------------------------------------------------
def run_appointment_reminders():
    """Runs every hour — sends WhatsApp reminders for upcoming appointments."""
    try:
        from appointments import send_reminders_job
        send_reminders_job()
    except Exception as e:
        print(f"Scheduler: reminder job error: {e}")


def run_scheduled_campaigns():
    """Runs every 2 minutes — sends any campaign whose scheduled time has arrived."""
    try:
        from campaigns import dispatch_scheduled_campaigns
        dispatch_scheduled_campaigns()
    except Exception as e:
        print(f"Scheduler: campaign dispatch error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on app startup
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_appointment_reminders, 'interval', hours=1, id='appointment_reminders')
        scheduler.add_job(run_scheduled_campaigns, 'interval', minutes=2, id='scheduled_campaigns')
        scheduler.start()
        print("Schedulers started — appointment reminders hourly, campaign dispatch every 2 minutes")
    except Exception as e:
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


# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}