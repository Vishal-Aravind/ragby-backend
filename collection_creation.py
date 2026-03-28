from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()

print("URL:", os.getenv("QDRANT_URL"))
print("KEY:", os.getenv("QDRANT_API_KEY")[:10])
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# qdrant.recreate_collection(
#     collection_name="documents",   # 👈 SAME as env
#     vectors_config=models.VectorParams(
#         size=1536,
#         distance=models.Distance.COSINE
#     )
# )

qdrant.create_payload_index(
    collection_name="documents",
    field_name="project_id",
    field_schema=models.PayloadSchemaType.KEYWORD
)

print("Index created ✅")

print("Collection created ✅")