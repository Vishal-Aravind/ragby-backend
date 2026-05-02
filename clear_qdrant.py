from qdrant_client import QdrantClient, models

qdrant = QdrantClient(
    url="https://5b7d6025-1177-4768-9528-4f2956029865.eu-west-2-0.aws.cloud.qdrant.io",  # 👈 from dashboard
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.131SJphdYAMmXbolnaxysNpOkIL8CEgRyCuEpCanLvs",                     # 👈 from dashboard
)

qdrant.delete(
    collection_name="documents",
    points_selector=models.Filter(must=[])
)

print("✅ All points deleted")