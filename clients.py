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

# Neither of these set a timeout, and the OpenAI SDK defaults to 600
# seconds with 2 automatic retries — so one hung call could pin a worker
# for half an hour. Every chat handler is a plain `def`, so it holds a
# threadpool slot for the whole time, and a single chat turn makes up to
# six OpenAI calls (embedding, classifier, text-to-SQL, and up to four
# completions). A handful of slow requests was enough to stall the entire
# application, dashboard included.
#
# Same posture as the _TimeoutSession used for every outbound HTTP call in
# whatsapp.py, telegram.py and billing.py.
OPENAI_TIMEOUT_SECONDS = 30

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT_SECONDS,
    # One retry, not two: a chat turn already chains several calls, and the
    # visitor is waiting on all of them.
    max_retries=1,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT_SECONDS,
    max_retries=1,
)