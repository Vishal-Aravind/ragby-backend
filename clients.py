from supabase import create_client
from openai import OpenAI
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
    OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)