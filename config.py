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

# Razorpay Billing — Zavo's OWN Razorpay merchant account, charging Zavo's
# OWN SaaS customers (Pro/Business subscriptions). A THIRD, distinct
# Razorpay credential concept in this codebase — do not confuse with:
#   - RAZORPAY_PARTNER_* above (backend/razorpay_oauth.py) — OAuth so
#     Zavo's Shop/Appointment merchants connect THEIR OWN account.
#   - RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET (backend/shop.py) — legacy
#     manual per-merchant fallback keys, also not Zavo's own account.
# Zavo IS the merchant here — plain Basic Auth via the razorpay-python SDK
# (razorpay.Client(auth=(id, secret))), no OAuth counterparty involved.
RAZORPAY_BILLING_KEY_ID = os.getenv("RAZORPAY_BILLING_KEY_ID", "")
RAZORPAY_BILLING_KEY_SECRET = os.getenv("RAZORPAY_BILLING_KEY_SECRET", "")
RAZORPAY_BILLING_WEBHOOK_SECRET = os.getenv("RAZORPAY_BILLING_WEBHOOK_SECRET", "")

RAZORPAY_PRO_MONTHLY = os.getenv("RAZORPAY_PRO_MONTHLY", "")
RAZORPAY_PRO_YEARLY = os.getenv("RAZORPAY_PRO_YEARLY", "")
RAZORPAY_BUSINESS_MONTHLY = os.getenv("RAZORPAY_BUSINESS_MONTHLY", "")
RAZORPAY_BUSINESS_YEARLY = os.getenv("RAZORPAY_BUSINESS_YEARLY", "")

# Server-side allowlist — /billing/subscribe resolves a {plan}/{billing} key
# to a plan_id itself; the client never sends a raw price/plan id. This is
# actually a fix over the old Stripe flow, where price_id came straight off
# the request body with no validation it was one of the known prices.
PLAN_TO_RAZORPAY_PLAN_ID = {
    ("pro", "monthly"): RAZORPAY_PRO_MONTHLY,
    ("pro", "yearly"): RAZORPAY_PRO_YEARLY,
    ("business", "monthly"): RAZORPAY_BUSINESS_MONTHLY,
    ("business", "yearly"): RAZORPAY_BUSINESS_YEARLY,
}

# Razorpay plan_id -> plan name, mirrors PRICE_TO_PLAN's old shape exactly.
# Used by the billing webhook to resolve profiles.plan from a subscription's
# current plan_id.
RAZORPAY_PLAN_TO_PLAN = {
    RAZORPAY_PRO_MONTHLY: "pro",
    RAZORPAY_PRO_YEARLY: "pro",
    RAZORPAY_BUSINESS_MONTHLY: "business",
    RAZORPAY_BUSINESS_YEARLY: "business",
}

PLAN_LIMITS = {
    "free":     {"conversations": 300,   "seats": 1,   "documents": 25,   "sources": 3},
    "pro":      {"conversations": 5000,  "seats": 5,   "documents": 500,  "sources": 20},
    "business": {"conversations": 25000, "seats": 100, "documents": 5000, "sources": 100},
}

# Hard ceilings that apply on EVERY plan, including business. Embedding is
# billed per chunk against our own OpenAI key, so without these a single
# 5,000-page PDF or a 100k-row sheet is one unbounded bill that no
# per-plan document count would catch.
MAX_CHUNKS_PER_INGEST = 3000
MAX_SHEET_ROWS = 5000

assert QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION