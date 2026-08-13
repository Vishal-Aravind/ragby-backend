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

# Razorpay Technology Partner OAuth app credentials ("Connect with Razorpay"
# flow, backend/razorpay_oauth.py) — distinct from shop.py's
# RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET, which remain the deprecated-but-
# still-supported manual per-merchant key fallback for shops that haven't
# reconnected via OAuth yet.
RAZORPAY_PARTNER_CLIENT_ID = os.getenv("RAZORPAY_PARTNER_CLIENT_ID", "")
RAZORPAY_PARTNER_CLIENT_SECRET = os.getenv("RAZORPAY_PARTNER_CLIENT_SECRET", "")
RAZORPAY_PARTNER_REDIRECT_URI = os.getenv("RAZORPAY_PARTNER_REDIRECT_URI", f"{BACKEND_PUBLIC_URL}/razorpay/oauth/callback")
# "test" | "live" — passed as the token endpoint's `mode` param. Razorpay
# issues separate Development/Production client_id+secret pairs per
# Partner app; per their docs a Production client can only ever request
# live-mode tokens, but a Development client needs this explicit to get a
# test-mode (sandbox) token instead of silently defaulting to live. Flip to
# "live" (and swap in the Production client_id/secret) when going live.
RAZORPAY_PARTNER_MODE = os.getenv("RAZORPAY_PARTNER_MODE", "test")
# One shared secret covering every connected sub-merchant account — Razorpay
# Partner webhooks are configured once at the application level, not per
# merchant (confirmed against Razorpay's Partner OAuth docs).
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

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