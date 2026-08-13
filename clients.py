import httpx
from supabase import create_client, ClientOptions
from openai import OpenAI
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
    OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
)

# This client is a module-level singleton, reused for every request for the
# life of the process — including across Render's free-tier "hibernate"
# freeze/thaw cycle. httpx's default HTTP/2 keep-alive pool has no way to
# notice a connection went stale while the whole process (and its idle-
# timeout clock) was frozen, not just idle — so the first request after a
# thaw tries to reuse a connection Supabase's side already dropped, and
# fails with a raw ReadError instead of transparently reconnecting.
# http2=False + retries=1 on the transport makes that a silent retry-with-
# a-fresh-connection instead of a 500.
_supabase_httpx_client = httpx.Client(
    http2=False,
    transport=httpx.HTTPTransport(retries=1),
)

supabase = create_client(
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
    options=ClientOptions(httpx_client=_supabase_httpx_client),
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)