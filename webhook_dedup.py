"""Shared webhook idempotency helper.

Every provider we receive webhooks from retries delivery, and until now
only WhatsApp deduped. Retries are not an edge case: Slack resends when it
doesn't get a 200 within 3 seconds, which an LLM call routinely exceeds, so
duplicate processing was the normal path rather than the exception.
"""
import sentry_sdk

from clients import supabase


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
        return False
    except Exception as e:
        message = str(e).lower()
        if "duplicate key" in message or "23505" in message:
            return True
        sentry_sdk.capture_exception(e)
        print(f"webhook dedup check failed for {source}:{event_id}: {e}")
        return False
