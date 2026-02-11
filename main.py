import os
import io
import uuid
from fastapi import Query
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
from fastapi.responses import PlainTextResponse
import hmac
import hashlib

@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
async def verify_whatsapp(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    # Ensure this variable exactly matches the token in your Meta dashboard
    MY_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
    
    if hub_mode == "subscribe" and hub_verify_token == MY_VERIFY_TOKEN:
        # Returning this as PlainTextResponse ensures NO quotes are added
        return hub_challenge
    
    raise HTTPException(status_code=403, detail="Verification failed")

def verify_signature(payload: bytes, signature: str):
    app_secret = os.getenv("FB_APP_SECRET")

    expected = hmac.new(
        app_secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)

@app.post("/webhook/whatsapp")
async def handle_whatsapp_msg(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers.get("X-Hub-Signature-256")
    body = await request.body()

    if not signature or not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()

    try:
        if "entry" not in data:
            return {"status": "ignored"}

        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        message = value["messages"][0]
        from_phone = message["from"]
        text_body = message.get("text", {}).get("body")
        phone_number_id = value["metadata"]["phone_number_id"]

        if not text_body:
            return {"status": "no_text"}

        background_tasks.add_task(
            process_whatsapp_rag,
            phone_number_id,
            from_phone,
            text_body
        )

    except Exception as e:
        print("Webhook error:", e)

    return {"status": "received"}

def process_whatsapp_rag(phone_number_id, from_phone, text_body):
    try:
        row = supabase.table("projects") \
            .select("id") \
            .eq("whatsapp_phone_number_id", phone_number_id) \
            .execute()

        if not row.data:
            return

        project_id = row.data[0]["id"]

        result = run_chat(project_id, text_body, [])
        answer = result["answer"]

        send_whatsapp_message(phone_number_id, from_phone, answer)

    except Exception as e:
        print("Background task error:", e)

def send_whatsapp_message(phone_number_id, to, message):
    row = supabase.table("projects") \
        .select("whatsapp_access_token") \
        .eq("whatsapp_phone_number_id", phone_number_id) \
        .execute()

    if not row.data:
        print("No token found")
        return

    access_token = row.data[0]["whatsapp_access_token"]

    url = f"https://graph.facebook.com/v24.0/{phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }

    r = requests.post(url, headers=headers, json=payload, timeout=10)
    print("WhatsApp send:", r.status_code, r.text)

@app.post("/whatsapp/onboard")
async def onboard_whatsapp(req: dict):
    code = req.get("code")
    project_id = req.get("projectId")

    if not code or not project_id:
        raise HTTPException(status_code=400, detail="Missing code or projectId")

    token_res = requests.get(
        "https://graph.facebook.com/v24.0/oauth/access_token",
        params={
            "client_id": os.getenv("FB_APP_ID"),
            "client_secret": os.getenv("FB_APP_SECRET"),
            "code": code,
        },
    ).json()

    access_token = token_res.get("access_token")
    if not access_token:
        raise HTTPException(status_code=500, detail=token_res)

    # Get WABA
    waba_res = requests.get(
        "https://graph.facebook.com/v24.0/me",
        params={
            "fields": "whatsapp_business_accounts",
            "access_token": access_token,
        },
    ).json()

    waba_id = waba_res["whatsapp_business_accounts"]["data"][0]["id"]

    # Subscribe app to WABA
    requests.post(
        f"https://graph.facebook.com/v24.0/{waba_id}/subscribed_apps",
        params={"access_token": access_token}
    )

    # Get phone number
    phone_res = requests.get(
        f"https://graph.facebook.com/v24.0/{waba_id}/phone_numbers",
        params={"access_token": access_token},
    ).json()

    phone_number_id = phone_res["data"][0]["id"]

    supabase.table("projects").update({
        "whatsapp_access_token": access_token,
        "whatsapp_phone_number_id": phone_number_id
    }).eq("id", project_id).execute()

    return {"status": "connected"}

