"""Shared webhook idempotency helper.

Every provider we receive webhooks from retries delivery, and until now
only WhatsApp deduped. Retries are not an edge case: Slack resends when it
doesn't get a 200 within 3 seconds, which an LLM call routinely exceeds, so
duplicate processing was the normal path rather than the exception.
"""
import random

import sentry_sdk

from clients import supabase

# Rows stay useful only for as long as a provider might still retry, but
# prune_webhook_dedup() (added alongside the table) was never actually
# called from anywhere, so the table grew without bound. Pruning on a small
# random fraction of deliveries keeps it bounded without a cron and without
# adding a round trip to the hot path. Same approach as oauth_state.py.
_PRUNE_PROBABILITY = 0.01


def _maybe_prune() -> None:
    if random.random() >= _PRUNE_PROBABILITY:
        return
    try:
        supabase.rpc("prune_webhook_dedup", {}).execute()
    except Exception as e:
        # Housekeeping must never fail a webhook.
        sentry_sdk.capture_exception(e)
        print(f"webhook dedup prune failed: {type(e).__name__}")


def already_processed(source: str, event_id: str) -> bool:
    """True if this event was already handled, and records it if not.

    Insert-and-catch-duplicate rather than select-then-insert, so two
    concurrent retries can't both pass the check. Call this BEFORE doing any
    work — replying, embedding, charging usage — not after.

    Fails OPEN (returns False) if the dedup table itself is unreachable:
    processing a message twice is a worse-but-recoverable outcome than
    dropping every message while Supabase is having a bad minute.
    """
    if not event_id:
        return False

    try:
        supabase.table("webhook_dedup").insert(
            {"source": source, "event_id": str(event_id)}
        ).execute()
        _maybe_prune()
        return False
    except Exception as e:
        message = str(e).lower()
        if "duplicate key" in message or "23505" in message:
            return True
        sentry_sdk.capture_exception(e)
        print(f"webhook dedup check failed for {source}:{event_id}: {e}")
        return False
