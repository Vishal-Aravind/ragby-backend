import os
from dotenv import load_dotenv

load_dotenv()

SENTRY_DSN = os.getenv("SENTRY_DSN")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ragby-frontend.vercel.app")
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")

SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

META_APP_ID = os.getenv("META_APP_ID")
META_APP_SECRET = os.getenv("META_APP_SECRET")

SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
# read_orders/write_orders added for Piece 2's order write-back (push_order_to_shopify
# in shop.py) — without these, the OAuth token Piece 1 issues has no permission to
# call orderCreate, and write-back would fail with a Shopify permissions error on
# every single order. Scopes are locked in at connect time, so changing this after a
# merchant already connected requires disconnecting and reconnecting to pick up the
# new permissions — it isn't retroactive.
SHOPIFY_APP_SCOPES = os.getenv("SHOPIFY_APP_SCOPES", "read_products,read_inventory,read_orders,write_orders")
SHOPIFY_REDIRECT_URI = os.getenv("SHOPIFY_REDIRECT_URI", f"{BACKEND_PUBLIC_URL}/shopify/oauth/callback")
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2026-07")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRO_MONTHLY = os.getenv("STRIPE_PRO_MONTHLY")
STRIPE_PRO_YEARLY = os.getenv("STRIPE_PRO_YEARLY")
STRIPE_BUSINESS_MONTHLY = os.getenv("STRIPE_BUSINESS_MONTHLY")
STRIPE_BUSINESS_YEARLY = os.getenv("STRIPE_BUSINESS_YEARLY")

PRICE_TO_PLAN = {
    STRIPE_PRO_MONTHLY: "pro",
    STRIPE_PRO_YEARLY: "pro",
    STRIPE_BUSINESS_MONTHLY: "business",
    STRIPE_BUSINESS_YEARLY: "business",
}

PLAN_LIMITS = {
    "free":     {"conversations": 300,   "seats": 1},
    "pro":      {"conversations": 5000,  "seats": 5},
    "business": {"conversations": 25000, "seats": None},
}

assert QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION