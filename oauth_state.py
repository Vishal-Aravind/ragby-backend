"""Durable, single-use OAuth state nonces shared by Slack/Shopify/Razorpay.

Each provider previously kept these in a module-level dict, which broke on
every restart (fatal on Render's free tier, which sleeps after ~15 minutes)
and recorded only the project, not the user who started the flow.

Note on scope: binding the nonce to a user stops someone redeeming a nonce
they didn't mint, and the target check stops a different store being bound
than the one requested. Neither prevents the phishing variant, where an
attacker starts a flow for their OWN project and persuades a merchant to
approve it — no amount of state binding fixes that, because the attacker
legitimately owns both ends. The provider's consent screen is the control
there.
"""
from datetime import datetime, timedelta, timezone

import secrets
import sentry_sdk

from clients import supabase

_TTL_SECONDS = 600


def issue_state(provider: str, project_id: str, user_id: str, target: str = None) -> str:
    nonce = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=_TTL_SECONDS)
    supabase.table("oauth_states").insert({
        "nonce": nonce,
        "provider": provider,
        "project_id": str(project_id),
        "user_id": str(user_id),
        "target": target,
        "expires_at": expires_at.isoformat(),
    }).execute()

    # Opportunistic cleanup so the table doesn't grow forever.
    try:
        supabase.table("oauth_states") \
            .delete() \
            .lt("expires_at", datetime.now(timezone.utc).isoformat()) \
            .execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)

    return nonce


def consume_state(provider: str, nonce: str) -> dict | None:
    """Pops the state for this nonce. Single use: the row is deleted before
    the caller acts on it, so a replayed callback finds nothing."""
    if not nonce:
        return None

    res = supabase.table("oauth_states") \
        .select("*") \
        .eq("nonce", nonce) \
        .eq("provider", provider) \
        .maybe_single() \
        .execute()
    row = (res.data if res else None) or None
    if not row:
        return None

    supabase.table("oauth_states").delete().eq("nonce", nonce).execute()

    expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        return None

    return row
