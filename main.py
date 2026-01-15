import os
import io
import uuid
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pdfplumber
from docx import Document
from pptx import Presentation
import pandas as pd

from supabase import create_client
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from fastapi.staticfiles import StaticFiles

import requests
from qdrant_client import QdrantClient, models


# -------------------------------------------------
# ENV & CLIENTS
# -------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

assert QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class IngestRequest(BaseModel):
    projectId: str
    filename: str
    filePath: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    projectId: str
    message: str
    history: Optional[List[ChatMessage]] = []

class PublicChatRequest(BaseModel):
    projectId: str
    message: str
    history: Optional[List[ChatMessage]] = []


# -------------------------------------------------
# TEXT EXTRACTORS
# -------------------------------------------------
def extract_pdf(b): 
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        return [(i+1, p.extract_text() or "") for i, p in enumerate(pdf.pages) if p.extract_text()]

def extract_docx(b):
    d = Document(io.BytesIO(b))
    return [(1, "\n".join(p.text for p in d.paragraphs if p.text.strip()))]

def extract_pptx(b):
    prs = Presentation(io.BytesIO(b))
    out = []
    for i, s in enumerate(prs.slides):
        txt = "\n".join(sh.text for sh in s.shapes if hasattr(sh, "text"))
        if txt.strip():
            out.append((i+1, txt))
    return out

def extract_excel(b):
    xls = pd.ExcelFile(io.BytesIO(b))
    return [(n, xls.parse(n).astype(str).fillna("").to_csv(index=False)) for n in xls.sheet_names]

def extract_txt(b):
    return [(1, b.decode("utf-8", errors="ignore"))]


# -------------------------------------------------
# INGEST
# -------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    row = supabase.table("files").select("id").eq("project_id", req.projectId).eq("filename", req.filename).execute()
    if not row.data:
        return {"error": "file not found"}

    file_id = row.data[0]["id"]
    supabase.table("files").update({"status": "processing"}).eq("id", file_id).execute()

    b = supabase.storage.from_("documents").download(req.filePath)
    ext = req.filename.lower().split(".")[-1]

    units = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "ppt": extract_pptx,
        "pptx": extract_pptx,
        "xls": extract_excel,
        "xlsx": extract_excel,
        "txt": extract_txt,
    }.get(ext)

    if not units:
        return {"error": "unsupported file type"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metas = [], []

    for page, text in units(b):
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({
                "project_id": req.projectId,
                "file_id": file_id,
                "filename": req.filename,
                "page_number": page,
                "text": c,
            })

    vectors = embeddings.embed_documents(chunks)

    qdrant.upload_points(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=m
            ) for v, m in zip(vectors, metas)
        ]
    )

    supabase.table("files").update({"status": "indexed"}).eq("id", file_id).execute()
    return {"status": "indexed", "chunks_indexed": len(chunks)}


# -------------------------------------------------
# STRICT CHAT (NO HALLUCINATION)
# -------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Use ONLY the provided context.\n"
    "If the answer is not explicitly present, reply exactly:\n"
    "\"I don’t know based on the provided documents.\"\n"
    "Do NOT infer, guess, or add external information.\n"
    "Keep the answer concise (max 3 sentences)."
)

def is_greeting(text: str) -> bool:
    text = text.lower().strip()
    return text in {
        "hi", "hello", "hey", "hi there", "hello there"      
    }

def is_thanking(text: str) -> bool:
    text = text.lower().strip()
    return text in {
        "ok", "k", "thanks", "thank you"      
    }


def run_chat(project_id: str, message: str, history: List[ChatMessage]):
    sources = [] 
    # 1. Initialize messages immediately
    messages = [] 
    
    try:
        if is_greeting(message):
            return {"answer": "Hello! How can I help you?", "sources": []}
        
        if is_thanking(message):
            return {"answer": "Great, looking forward to help you!", "sources": []}

        q = embeddings.embed_query(message)

        res = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q,
            limit=5,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )]
            )
        )

        hits = res.points
        if not hits:
            return {"answer": "I don’t know based on the provided documents.", "sources": sources}

        context = "\n\n---\n\n".join(h.payload.get("text", "") for h in hits)
        
        # 2. Setup the System Prompt and History
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for h in history:
            messages.append({"role": h.role, "content": h.content})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{message}"
        })

        # 3. Call OpenAI
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=120,
        )

        # ... (rest of your sources logic)
        return {
            "answer": completion.choices[0].message.content.strip(),
            "sources": sources
        }

    except Exception as e:
        print(f"ERROR IN RUN_CHAT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(req: ChatRequest):
    return run_chat(req.projectId, req.message, req.history)


@app.post("/public/chat")
def public_chat(req: PublicChatRequest):
    return run_chat(req.projectId, req.message, req.history)


# -------------------------------------------------
# DELETE DOCUMENT
# -------------------------------------------------
@app.delete("/document/{file_id}")
def delete_document(file_id: str):
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="file_id",
                match=models.MatchValue(value=file_id)
            )]
        )
    )
    supabase.table("files").delete().eq("id", file_id).execute()
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {"status": "ok"}


from fastapi import Request, BackgroundTasks

@app.post("/zoom/bot")
async def handle_zoom_bot(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    # 1. URL validation (one-time)
    if payload.get("event") == "endpoint.url_validation":
        return validate_zoom_token(payload)

    # 2. Slash command
    if payload.get("event") == "bot_notification":
        user_query = payload["payload"]["cmd"]
        channel_id = payload["payload"]["to_jid"]

        # Run RAG in background
        background_tasks.add_task(process_and_reply, channel_id, user_query)

    return {"message": "ok"}


def process_and_reply(channel_id: str, user_query: str):
    answer = run_rag_pipeline(user_query)
    send_chatbot_response(channel_id, answer["answer"])

import hmac, hashlib, base64

SECRET_TOKEN = os.getenv("ZOOM_SECRET_TOKEN")

def validate_zoom_token(payload):
    plain = payload["payload"]["plainToken"]
    encrypted = hmac.new(
        SECRET_TOKEN.encode(),
        plain.encode(),
        hashlib.sha256
    ).hexdigest()
    return {
        "plainToken": plain,
        "encryptedToken": encrypted
    }
