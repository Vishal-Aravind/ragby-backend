from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
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


# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}