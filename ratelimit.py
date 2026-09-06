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
import time
from collections import defaultdict

import sentry_sdk

from clients import supabase

_hits = defaultdict(list)

# Bounds the fallback dict so a long outage can't exhaust memory the way
# the original unbounded version could.
_MAX_FALLBACK_KEYS = 10000


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
        sentry_sdk.capture_exception(e)
        print(f"rate limit check failed for {key}, falling back to memory: {e}")

    return _in_memory_is_rate_limited(key, limit, window_seconds)


def client_ip(request) -> str:
    """The caller's real IP, for rate-limit keys.

    Every call site used to read the FIRST X-Forwarded-For entry, which is
    supplied by the client. Render appends the real address rather than
    replacing the header, so `-H "X-Forwarded-For: 1.2.3.4"` (rotated per
    request) handed an attacker an unlimited supply of fresh buckets and
    defeated every IP-keyed limit, including login brute-force protection.

    The rightmost entry is the one our own proxy appended, so that is the
    only value here we can trust. Assumes a single trusted proxy in front
    of the app, which matches how this is deployed.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"
