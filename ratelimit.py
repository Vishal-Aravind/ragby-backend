"""Shared rate limiter, backed by Postgres so limits survive a restart.

State used to live in a module-level dict. That was fine in theory for a
long-running process, but on Render's free tier the service sleeps after
~15 minutes of inactivity and cold-starts on the next request — so in
practice every counter reset constantly, including the login brute-force
protection and the per-project caps that bound OpenAI spend. The dict also
never evicted keys, so it grew for the life of the process.

Counters now live in the `rate_limits` table via the `check_rate_limit`
function (one atomic upsert per check). If that call fails for any reason,
we fall back to the old in-memory behaviour rather than letting a Supabase
blip take down every protected endpoint — degraded limiting beats none.
"""
import hmac
import ipaddress
import time
from collections import defaultdict

import sentry_sdk

from clients import supabase
from config import INTERNAL_PROXY_SECRET

_hits = defaultdict(list)

# Bounds the fallback dict so a long outage can't exhaust memory the way
# the original unbounded version could.
_MAX_FALLBACK_KEYS = 10000

# is_rate_limited runs on nearly every request to every protected endpoint
# in the backend — its own docstring says "call once per attempt." If the
# check_rate_limit RPC breaks (a migration issue, a Supabase outage), a
# direct capture_exception on every call would fire on every single
# request across the whole app simultaneously — the highest-frequency
# capture site of anything audited in this series, well beyond any
# per-batch loop elsewhere. Cooled down the same way main.py's scheduler
# infrastructure failures are: at most one Sentry event per window,
# however often requests keep arriving in that window.
_RATE_LIMIT_FAILURE_COOLDOWN_SECONDS = 3600
_last_reported_at = None


def _capture_with_cooldown(e: Exception):
    global _last_reported_at
    now = time.time()
    if _last_reported_at is None or (now - _last_reported_at) > _RATE_LIMIT_FAILURE_COOLDOWN_SECONDS:
        sentry_sdk.capture_exception(e)
        _last_reported_at = now


def _in_memory_is_rate_limited(key: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    recent = [t for t in _hits.get(key, []) if now - t < window_seconds]
    recent.append(now)

    if key not in _hits and len(_hits) >= _MAX_FALLBACK_KEYS:
        for stale_key in [
            k for k, v in _hits.items() if not v or now - v[-1] > window_seconds
        ][:1000]:
            _hits.pop(stale_key, None)

    _hits[key] = recent
    return len(recent) > limit


def is_rate_limited(key: str, limit: int, window_seconds: int = 60) -> bool:
    """True if `key` has already hit `limit` events within the last
    `window_seconds`. Call once per attempt — every call counts as an
    attempt, whether or not it turns out to be allowed."""
    try:
        res = supabase.rpc(
            "check_rate_limit",
            {"p_key": key, "p_limit": limit, "p_window_seconds": window_seconds},
        ).execute()
        if res.data is not None:
            return bool(res.data)
    except Exception as e:
        _capture_with_cooldown(e)
        print(f"rate limit check failed for {key}, falling back to memory: {e}")

    return _in_memory_is_rate_limited(key, limit, window_seconds)


def client_ip(request) -> str:
    """The caller's real IP, for rate-limit keys.

    Two problems stacked here, and the second hid the first.

    Every call site used to read the FIRST X-Forwarded-For entry, which is
    supplied by the client. Render appends the real address rather than
    replacing the header, so `-H "X-Forwarded-For: 1.2.3.4"` (rotated per
    request) handed an attacker an unlimited supply of fresh buckets and
    defeated every IP-keyed limit. Trusting only the rightmost hop fixed
    that — it is the one our own proxy appended.

    But almost every anonymous request arrives via our own Next.js routes,
    so that rightmost hop is the FRONTEND's egress address, identical for
    every visitor on earth. Every IP-keyed limit therefore collapsed into a
    single global bucket: no per-attacker login throttling, and one
    attacker able to lock out every user at once.

    So the frontend now passes the visitor's address explicitly, proved by
    a shared secret. That header is trusted only when the secret matches
    AND the value parses as an IP — otherwise anyone who learned the
    secret could inject unbounded junk into rate-limit keys.

    With INTERNAL_PROXY_SECRET unset on either side, this falls through to
    the rightmost-hop behaviour above and nothing changes.
    """
    forwarded_by_us = request.headers.get("X-Visitor-IP", "")
    if forwarded_by_us and INTERNAL_PROXY_SECRET:
        presented = request.headers.get("X-Internal-Proxy-Secret", "")
        if presented and hmac.compare_digest(presented, INTERNAL_PROXY_SECRET):
            candidate = forwarded_by_us.strip()
            try:
                # Normalises as a side effect, so "1.2.3.4" and
                # "::ffff:1.2.3.4" can't hold two separate buckets.
                return str(ipaddress.ip_address(candidate))
            except ValueError:
                pass

    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"
