"""Pincode -> rough coordinates, server-side only, via Nominatim (OSM).

Used for exactly two things: ranking branches by "nearest" when a customer
types a pincode instead of sharing their location, and centring the map.
Never for delivery fees, which always use the customer's own pin.

Nominatim's usage policy is 1 request/second with a real User-Agent, so
results are cached in geocode_cache and every miss is rate limited. Any
failure returns None and the caller carries on without a distance.
"""
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests
import sentry_sdk

from clients import supabase
from config import FRONTEND_URL
from ratelimit import is_rate_limited

NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org").rstrip("/")
# The policy requires a User-Agent that identifies the app with a contact.
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT") or f"ZavoShop/1.0 (+{FRONTEND_URL})"

FOUND_TTL = timedelta(days=180)
NOT_FOUND_TTL = timedelta(days=7)


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _valid(lat, lng) -> bool:
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(lat) and math.isfinite(lng)
            and -90 <= lat <= 90 and -180 <= lng <= 180 and not (lat == 0 and lng == 0))


def _cache_get(cc: str, pincode: str):
    """(hit, coords). hit is False when there is no fresh cache row."""
    try:
        res = supabase.table("geocode_cache").select("found, lat, lng, fetched_at") \
            .eq("country_code", cc).eq("postal_code", pincode).limit(1).execute()
        row = res.data[0] if res.data else None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False, None
    if not row:
        return False, None
    fetched = _parse_ts(row.get("fetched_at"))
    if not fetched:
        return False, None
    age = datetime.now(timezone.utc) - fetched
    if row.get("found"):
        if age < FOUND_TTL and _valid(row.get("lat"), row.get("lng")):
            return True, (float(row["lat"]), float(row["lng"]))
        return False, None
    if age < NOT_FOUND_TTL:
        return True, None
    return False, None


def _cache_put(cc: str, pincode: str, coords: Optional[Tuple[float, float]]):
    try:
        supabase.table("geocode_cache").upsert({
            "country_code": cc,
            "postal_code": pincode,
            "found": coords is not None,
            "lat": coords[0] if coords else None,
            "lng": coords[1] if coords else None,
            "source": "nominatim",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="country_code,postal_code").execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)


def pincode_to_latlng(country_code: str, pincode: str, ip: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """(lat, lng) for an already-validated pincode, or None. Blocking (up to
    ~8 s), so only call it from plain `def` endpoints."""
    cc = (country_code or "IN").upper()
    if not pincode:
        return None

    hit, coords = _cache_get(cc, pincode)
    if hit:
        return coords

    # Graceful, not 429: the branch list still works without a distance.
    if ip and is_rate_limited(f"geocode-ip:{ip}", 10, 60):
        return None
    if is_rate_limited("nominatim:global", 1, 1):
        return None

    try:
        res = requests.get(
            f"{NOMINATIM_URL}/search",
            params={"postalcode": pincode, "countrycodes": cc.lower(), "format": "jsonv2", "limit": 1},
            headers={"User-Agent": NOMINATIM_USER_AGENT, "Accept-Language": "en"},
            timeout=(3, 5),
        )
    except requests.RequestException as e:
        sentry_sdk.capture_exception(e)
        return None
    if res.status_code != 200:
        # Not cached: a Nominatim outage must not mark real pincodes unknown.
        sentry_sdk.capture_message(f"Nominatim returned {res.status_code} for {cc} {pincode}")
        return None

    try:
        results = res.json()
    except ValueError as e:
        sentry_sdk.capture_exception(e)
        return None
    if not isinstance(results, list):
        return None

    coords = None
    if results:
        first = results[0] if isinstance(results[0], dict) else {}
        lat, lng = first.get("lat"), first.get("lon")
        if not _valid(lat, lng):
            # Unusable answer: don't cache it either way.
            return None
        coords = (round(float(lat), 6), round(float(lng), 6))
    _cache_put(cc, pincode, coords)
    return coords
