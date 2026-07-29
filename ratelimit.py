"""
Shared in-memory rate limiter — single-process, no Redis, matching how the
rest of this app is built. Used to stop rapid-fire abuse (spam bookings,
brute-forced passwords, scripted signups) on endpoints that only ever had a
*monthly* cost cap (usage.py) or no protection at all. Not meant to replace
that monthly cap — this is a fast-acting, short-window check on top of it.
"""
import time
from collections import defaultdict

_hits = defaultdict(list)


def is_rate_limited(key: str, limit: int, window_seconds: int = 60) -> bool:
    """True if `key` has already hit `limit` events within the last
    `window_seconds`. Call once per attempt — every call counts as an
    attempt, whether or not it turns out to be allowed."""
    now = time.time()
    recent = [t for t in _hits.get(key, []) if now - t < window_seconds]
    recent.append(now)
    _hits[key] = recent
    return len(recent) > limit
