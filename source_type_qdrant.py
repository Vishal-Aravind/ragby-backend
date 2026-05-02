
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
load_dotenv()

qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
collection = os.getenv('QDRANT_COLLECTION')

# Create indexes
qdrant.create_payload_index(collection, 'source_type', models.PayloadSchemaType.KEYWORD)
qdrant.create_payload_index(collection, 'source_id', models.PayloadSchemaType.KEYWORD)

# Patch existing document chunks
offset = None
patched = 0
while True:
    results, offset = qdrant.scroll(collection, limit=100, offset=offset, with_payload=True)
    for point in results:
        if 'file_id' in point.payload and 'source_type' not in point.payload:
            qdrant.set_payload(collection, {'source_type': 'document'}, points=[point.id])
            patched += 1
    if offset is None:
        break

print(f'Done — patched {patched} points')
