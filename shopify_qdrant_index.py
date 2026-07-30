from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

qdrant.create_payload_index(
    collection_name=os.getenv("QDRANT_COLLECTION"),
    field_name="shopify_product_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

print("shopify_product_id index created")
