"""Embed and upload a source's chunks in bounded batches, then retire the
previous version's points.

Every ingest path used to call embeddings.embed_documents() on ALL chunks at
once and hold every vector in memory before uploading. Each vector is 1536
Python floats (~62KB as Python objects), so a 5,000-row sheet was ~313MB of
vectors on top of the app's ~305MB idle footprint — past Render's 512MB
limit, and the instance was killed mid-request.

Batching bounds peak memory to one batch (~6MB at 100). New points are
uploaded BEFORE the old ones are deleted, so a failure partway through
leaves the previous working index intact (the partial new points are rolled
back) — the same "never wipe on failure" guarantee each caller used to get
by embedding everything up front.
"""
import uuid

import sentry_sdk
from qdrant_client import models

EMBED_BATCH_SIZE = 100
_DELETE_BATCH_SIZE = 1000


def _existing_point_ids(qdrant, collection: str, match_key: str, match_value: str) -> list:
    """IDs only (no vectors, no payload) of the points currently indexed for
    this file/source — a few KB even for thousands of points."""
    ids = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key=match_key, match=models.MatchValue(value=match_value))
            ]),
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(p.id for p in points)
        if offset is None:
            return ids


def _delete_ids(qdrant, collection: str, ids: list):
    for i in range(0, len(ids), _DELETE_BATCH_SIZE):
        qdrant.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=ids[i:i + _DELETE_BATCH_SIZE]),
        )


def replace_points(qdrant, embeddings, collection: str, chunks: list, metas: list,
                   match_key: str, match_value: str):
    """Index `chunks` for one file/source, replacing whatever was indexed for
    it before. `match_key`/`match_value` identify the file/source
    (e.g. "source_id", <id>) — the same field the old purge filtered on."""
    old_ids = _existing_point_ids(qdrant, collection, match_key, match_value)

    new_ids = []
    try:
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            vectors = embeddings.embed_documents(chunks[i:i + EMBED_BATCH_SIZE])
            points = [
                models.PointStruct(id=str(uuid.uuid4()), vector=v, payload=m)
                for v, m in zip(vectors, metas[i:i + EMBED_BATCH_SIZE])
            ]
            qdrant.upload_points(collection_name=collection, points=points)
            new_ids.extend(p.id for p in points)
    except Exception:
        # Roll back the partial new version so the old index stays the only
        # one answering questions.
        try:
            _delete_ids(qdrant, collection, new_ids)
        except Exception as cleanup_error:
            sentry_sdk.capture_exception(cleanup_error)
        raise

    _delete_ids(qdrant, collection, old_ids)
