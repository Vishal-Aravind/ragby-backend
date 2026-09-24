"""Delete every vector in the Qdrant collection. Run manually, never imported.

Was hardcoding a live Qdrant URL and API key directly in this file, committed
to git — anyone with repo access (now or from git history, since the key
stays recoverable even after this fix) could read or wipe the vector store
with it. Reads from config.py like every other client in this codebase, and
that key should be treated as already compromised: rotate it in Qdrant's
dashboard regardless of this fix.

This deletes ALL projects' vectors, not one. There is no per-project undo.
"""
from clients import qdrant
from config import QDRANT_COLLECTION
from qdrant_client import models

if __name__ == "__main__":
    confirm = input(
        f"This deletes every vector in '{QDRANT_COLLECTION}', across ALL projects. "
        f"Type the collection name to confirm: "
    )
    if confirm != QDRANT_COLLECTION:
        print("Aborted — input did not match.")
    else:
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.Filter(must=[]),
        )
        print(f"All points deleted from '{QDRANT_COLLECTION}'.")
