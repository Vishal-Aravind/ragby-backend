import os
import io
import uuid
from fastapi import Query, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
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

from sources.gsheets import sync_sheet
from sources.postgres import run_text_to_sql, introspect_schema, validate_url
from sources.excel import sync_excel_url, sync_excel_bytes
from sources.website import sync_website

import sqlalchemy

import hmac
import hashlib
import time

from starlette.responses import PlainTextResponse

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
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")  # FIX: lock down CORS origin
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

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
    # FIX: Restrict CORS to your frontend origin instead of wildcard
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# AUTH
# -------------------------------------------------
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    FIX: Verify the Supabase JWT on every protected endpoint.
    Uses Supabase's get_user() which validates the token server-side.
    Returns the user object so endpoints can use user.id if needed.
    """
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


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
    chatId: str
    message: str

class PublicChatRequest(BaseModel):
    projectId: str
    message: str
    # FIX: Accept an optional sessionId so public chat can maintain history across turns
    sessionId: Optional[str] = None


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
# FIX: Protected with verify_token — only authenticated users can ingest
# -------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest, user=Depends(verify_token)):
    row = supabase.table("files").select("id").eq("project_id", req.projectId).eq("filename", req.filename).execute()
    if not row.data:
        return {"error": "file not found"}

    file_id = row.data[0]["id"]
    supabase.table("files").update({"status": "processing"}).eq("id", file_id).execute()

    b = supabase.storage.from_("documents").download(req.filePath)
    ext = req.filename.lower().split(".")[-1]

    extractor = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "ppt": extract_pptx,
        "pptx": extract_pptx,
        "xls": extract_excel,
        "xlsx": extract_excel,
        "txt": extract_txt,
    }.get(ext)

    if not extractor:
        supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
        return {"error": "unsupported file type"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks, metas = [], []

    for page, text in extractor(b):
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({
                "project_id": req.projectId,
                "file_id": file_id,
                "filename": req.filename,
                "page_number": page,
                "source_type": "document",  # ADD THIS
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
# CHAT HISTORY HELPERS
# -------------------------------------------------
def get_history(chat_id: str, limit: int = 7):
    res = supabase.table("chat_messages") \
        .select("role, content") \
        .eq("chat_id", chat_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return list(reversed(res.data)) if res.data else []

def save_message(chat_id: str, role: str, content: str):
    supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content
    }).execute()

# -------------------------------------------------
# INTENT DETECTION
# -------------------------------------------------
def classify_intent(message: str) -> str:
    msg = message.lower().strip()
    words = msg.split()

    if len(words) <= 3 and msg in {"hi", "hello", "hey", "hi there", "hello there"}:
        return "greeting"

    if len(words) <= 4 and any(w in msg for w in ["thanks", "thank you", "thx"]):
        return "thanks"

    if any(k in msg for k in [
        "earlier", "previous", "you said", "we talked",
        "last message", "first question"
    ]):
        return "conversational"

    return "document_query"


def get_project_domain(project_id: str):
    res = supabase.table("projects") \
        .select("domain") \
        .eq("id", project_id) \
        .single() \
        .execute()
    return res.data.get("domain") if res.data else None


# -------------------------------------------------
# SOURCE INTENT CLASSIFIER
# -------------------------------------------------
def classify_source_intent(message: str) -> str:
    """
    Classifies whether the question is looking up a specific record (structured)
    or asking about a concept/process (conceptual).
    structured → search gsheets/postgres
    conceptual → search documents
    """
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify this question as either 'structured' or 'conceptual'.

'structured' = looking up a specific record, person, value, date, status, or list
Examples: "what are John's remarks", "show sales for March", "find order status for ID 123", "list all employees"

'conceptual' = asking about a process, policy, explanation, or general knowledge
Examples: "how does the refund process work", "what is the leave policy", "explain the onboarding steps"

Question: {message}
Reply with only one word: structured or conceptual"""
        }],
        temperature=0,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip().lower()
    # Safety fallback — if LLM returns something unexpected, default to conceptual
    return result if result in ("structured", "conceptual") else "conceptual"


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful RAG AI assistant. Use ONLY the provided context and recent conversation if relevant. "
    "If the user asks about you, introduce yourself politely.\n\n"

    "Context source guidance:\n"
    "- Context chunks labeled [Source: gsheets] or [Source: database] contain structured data like records, names, values, dates.\n"
    "- Context chunks labeled [Source: document] contain policies, procedures, or explanatory content.\n"
    "- Always prefer the source that best matches the question type.\n\n"

    "Style:\n"
    "- Simple question → short answer\n"
    "- Complex question → structured answer, no heavy formatting, medium length\n"
    "- Be concise, do not over-explain unless necessary\n\n"
    "- Pricing/packages/plans questions → always list ALL options, never just one\n"

    "Logic:\n"
    "- One clear answer → answer directly\n"
    "- Multiple answers → ask a clarification question briefly\n"
    "- Partial info → answer + mention missing briefly\n"
    "- No answer → say you couldn't find the information and ask if they have more details\n"
    "- Match answer length to question complexity\n\n"

    "Formatting rules:\n"
    "- Use bullet points ONLY when needed\n\n"

    "Rules:\n"
    "- No hallucination\n"
    "- No external knowledge\n"
    "- Use previous conversation history for context when relevant"
)


# -------------------------------------------------
# CORE CHAT LOGIC
# -------------------------------------------------
def run_chat(project_id: str, chat_id: str, message: str, history: list):
    try:
        history = history or []
        domain = get_project_domain(project_id)

        system_prompt = SYSTEM_PROMPT
        if domain:
            system_prompt += f"\n\nDomain:\n- You are specialized in {domain}."

        intent = classify_intent(message)

        # 1. Greeting
        if intent == "greeting":
            return {"answer": "Hey! 👋 What can I help you with?", "sources": []}

        # 2. Thanks
        if intent == "thanks":
            return {"answer": "You're welcome! 😊", "sources": []}

        # Save user message for all real intents
        save_message(chat_id, "user", message)

        # 3. Conversational — history only, no retrieval needed
        if intent == "conversational":
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-7:]:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": message})

            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            answer = completion.choices[0].message.content.strip()
            save_message(chat_id, "assistant", answer)
            return {"answer": answer, "sources": []}

        # 4. Classify source intent — structured (sheet/db) or conceptual (docs)
        source_intent = classify_source_intent(message)

        # Expand short queries with last user message for better recall
        query_for_embedding = message
        if len(message.split()) <= 4 and history:
            last_user_msgs = [m for m in history if m["role"] == "user"]
            if last_user_msgs:
                query_for_embedding = last_user_msgs[-1]["content"] + " " + message

        q = embeddings.embed_query(query_for_embedding)

        context = None

        # -------------------------------------------------
        # 5. Route to correct source based on intent
        # -------------------------------------------------
        if source_intent == "structured":
            # Try gsheets chunks in Qdrant first
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q,
                limit=7,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
                        models.FieldCondition(key="source_type", match=models.MatchAny(any=["gsheets", "excel"])),
                    ]
                )
            )
            hits = res.points

            if hits:
                context = "\n\n---\n\n".join(
                    f"[Source: gsheets]\n{h.payload.get('text', '')}" for h in hits
                )
            else:
                # Fallback: try postgres text-to-SQL
                pg_source = supabase.table("data_sources") \
                    .select("config, allowed_schema") \
                    .eq("project_id", project_id) \
                    .eq("type", "postgres") \
                    .limit(1) \
                    .execute()

                if pg_source.data:
                    db_url = pg_source.data[0]["config"]["url"]
                    allowed_schema = pg_source.data[0].get("allowed_schema")
                    sql_result = run_text_to_sql(message, db_url, openai_client, allowed_schema)
                    context = f"[Source: database]\n{sql_result}"
                else:
                    # No structured source found — fall through to document search
                    source_intent = "conceptual"

        if source_intent == "conceptual":
            # Search document chunks only
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q,
                limit=7,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
                        models. models.FieldCondition(key="source_type", match=models.MatchAny(any=["document", "website"])),
                    ]
                )
            )
            hits = res.points

            if hits:
                context = "\n\n---\n\n".join(
                    f"[Source: document]\n{h.payload.get('text', '')}" for h in hits
                )

        # -------------------------------------------------
        # 6. If nothing found anywhere, give up cleanly
        # -------------------------------------------------
        if not context:
            answer = "I couldn't find that in your documents or data sources."
            save_message(chat_id, "assistant", answer)
            return {"answer": answer, "sources": []}

        # -------------------------------------------------
        # 7. Build LLM messages and generate answer
        # -------------------------------------------------
        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-7:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{message}"
        })

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

        answer = completion.choices[0].message.content.strip()
        save_message(chat_id, "assistant", answer)
        return {"answer": answer, "sources": []}

    except Exception as e:
        print(f"ERROR IN RUN_CHAT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# INGEST — make sure source_type: "document" is set
# (update your existing ingest endpoint's metas to include this)
# -------------------------------------------------
# metas.append({
#     "project_id": req.projectId,
#     "file_id": file_id,
#     "filename": req.filename,
#     "page_number": page,
#     "source_type": "document",   <-- ADD THIS LINE
#     "text": c,
# })


# -------------------------------------------------
# CHAT ENDPOINTS
# FIX: /chat is protected with verify_token
# -------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest, user=Depends(verify_token)):
    history = get_history(req.chatId, limit=7)
    return run_chat(req.projectId, req.chatId, req.message, history)

@app.post("/public/chat")
def public_chat(req: PublicChatRequest):
    session_id = req.sessionId or str(uuid.uuid4())
    
    # FIX: ensure a chat record exists for this session
    # so chat_messages foreign key constraint doesn't fail
    existing = supabase.table("chats").select("id").eq("id", session_id).execute()
    if not existing.data:
        supabase.table("chats").insert({
            "id": session_id,
            "project_id": req.projectId,
            "title": "Public Chat",
            "channel": "public",
        }).execute()

    history = get_history(session_id, limit=7) if req.sessionId else []
    result = run_chat(req.projectId, session_id, req.message, history)
    result["sessionId"] = session_id
    return result

# -------------------------------------------------
# DELETE DOCUMENT
# FIX: Protected with verify_token
# -------------------------------------------------
@app.delete("/document/{file_id}")
def delete_document(file_id: str, user=Depends(verify_token)):
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




########
#Sources
########

@app.get("/sources")
def list_sources(project_id: str, user=Depends(verify_token)):
    res = supabase.table("data_sources") \
        .select("*") \
        .eq("project_id", project_id) \
        .execute()

    return res.data

@app.post("/sources/add")
def add_source(data: dict, user=Depends(verify_token)):
    res = supabase.table("data_sources").insert({
        "project_id": data["projectId"],
        "type": data["type"],
        "label": data.get("label") or data["type"],
        "config": data["config"],
        "allowed_schema": data.get("allowed_schema"),
    }).execute()
    source = res.data[0]
 
    skipped_tabs = []
 
    if data["type"] == "gsheets":
        cfg = data["config"]
        result = sync_sheet(
            cfg["sheet_id"], cfg["range"], data["projectId"],
            source["id"], qdrant, embeddings, QDRANT_COLLECTION
        )
        skipped_tabs = result.get("skipped_tabs", [])
 
    elif data["type"] == "excel_online":
        cfg = data["config"]
        sync_excel_url(
            cfg["url"], data["projectId"],
            source["id"], qdrant, embeddings, QDRANT_COLLECTION
        )
 
    elif data["type"] == "website":
        cfg = data["config"]
        result = sync_website(
            url=cfg["url"],
            project_id=data["projectId"],
            source_id=source["id"],
            qdrant=qdrant,
            embeddings=embeddings,
            collection=QDRANT_COLLECTION,
            full_site=cfg.get("full_site", True),
            max_pages=cfg.get("max_pages", 50),
        )
        # FIX: if nothing was indexed, delete the source record and tell the user
        if result["pages_indexed"] == 0:
            supabase.table("data_sources").delete().eq("id", source["id"]).execute()
            raise HTTPException(
                status_code=400,
                detail="Could not crawl this website. It may be blocking automated access (anti-bot protection). Try a different website or contact the site owner."
            )
 
    return {"id": source["id"], "skipped_tabs": skipped_tabs}

@app.delete("/sources/{source_id}")
def delete_source(source_id: str, user=Depends(verify_token)):
    qdrant.delete(collection_name=QDRANT_COLLECTION, points_selector=models.Filter(
        must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
    ))
    supabase.table("data_sources").delete().eq("id", source_id).execute()
    return {"status": "deleted"}

@app.post("/sources/sync/{source_id}")
def resync_source(source_id: str, user=Depends(verify_token)):
    res = supabase.table("data_sources").select("*").eq("id", source_id).single().execute()
    s = res.data
 
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )
 
    if s["type"] == "gsheets":
        cfg = s["config"]
        sync_sheet(
            cfg["sheet_id"], cfg["range"],
            s["project_id"], source_id,
            qdrant, embeddings, QDRANT_COLLECTION
        )
    elif s["type"] == "excel_online":
        cfg = s["config"]
        sync_excel_url(
            cfg["url"], s["project_id"],
            source_id, qdrant, embeddings, QDRANT_COLLECTION
        )
    elif s["type"] == "website":
        cfg = s["config"]
        sync_website(
            url=cfg["url"],
            project_id=s["project_id"],
            source_id=source_id,
            qdrant=qdrant,
            embeddings=embeddings,
            collection=QDRANT_COLLECTION,
            full_site=cfg.get("full_site", True),
            max_pages=cfg.get("max_pages", 50),
        )
 
    return {"status": "synced"}

@app.post("/sources/introspect")
def introspect(data: dict, user=Depends(verify_token)):
    db_url = data.get("db_url", "")
    try:
        validate_url(db_url)
        schema = introspect_schema(db_url)
        return {"schema": schema}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not connect: {str(e)}")

@app.post("/sources/upload-excel")
async def upload_excel(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    label: str = Form(""),
    source_id: str = Form(""),   # optional — if provided, overwrite this source
    user=Depends(verify_token)
):
    file_bytes = await file.read()

    if source_id:
        # Overwrite existing source — delete old Qdrant points and reuse record
        s = supabase.table("data_sources").select("*").eq("id", source_id).single().execute()
        sync_excel_bytes(file_bytes, projectId, source_id, qdrant, embeddings, QDRANT_COLLECTION)
        supabase.table("data_sources").update({
            "config": {"filename": file.filename},
            "label": label or file.filename,
        }).eq("id", source_id).execute()
        return {"id": source_id, "filename": file.filename}

    # New source
    res = supabase.table("data_sources").insert({
        "project_id": projectId,
        "type": "excel_local",
        "label": label or file.filename,
        "config": {"filename": file.filename},
        "allowed_schema": None,
    }).execute()
    source = res.data[0]

    sync_excel_bytes(file_bytes, projectId, source["id"], qdrant, embeddings, QDRANT_COLLECTION)
    return {"id": source["id"], "filename": file.filename}



# Add this import at the top of main.py
import httpx

# ─────────────────────────────────────────────────────────
# TELEGRAM HELPER
# ─────────────────────────────────────────────────────────
def send_telegram_message(bot_token: str, chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)


def set_telegram_webhook(bot_token: str, webhook_url: str):
    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    res = requests.post(url, json={"url": webhook_url})
    return res.json()


def get_bot_info(bot_token: str):
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    res = requests.get(url)
    return res.json()


# ─────────────────────────────────────────────────────────
# CONNECT TELEGRAM — save token + set webhook
# ─────────────────────────────────────────────────────────
@app.post("/telegram/connect")
def telegram_connect(data: dict, user=Depends(verify_token)):
    bot_token = data["bot_token"]
    project_id = data["projectId"]

    # Verify token is valid by calling getMe
    bot_info = get_bot_info(bot_token)
    if not bot_info.get("ok"):
        raise HTTPException(status_code=400, detail="Invalid bot token. Make sure you copied it correctly from @BotFather.")

    bot_username = bot_info["result"]["username"]

    # Save to Supabase
    supabase.table("telegram_integrations").upsert({
        "project_id": project_id,
        "bot_token": bot_token,
        "bot_username": bot_username,
    }, on_conflict="project_id").execute()

    # Set webhook so Telegram sends messages to our server
    webhook_url = f"{os.getenv('BACKEND_PUBLIC_URL')}/webhook/telegram/{project_id}"
    result = set_telegram_webhook(bot_token, webhook_url)

    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=f"Could not set webhook: {result}")

    return {"success": True, "bot_username": bot_username}


# ─────────────────────────────────────────────────────────
# DISCONNECT TELEGRAM
# ─────────────────────────────────────────────────────────
@app.delete("/telegram/disconnect/{project_id}")
def telegram_disconnect(project_id: str, user=Depends(verify_token)):
    # Get token first to remove webhook
    res = supabase.table("telegram_integrations") \
        .select("bot_token") \
        .eq("project_id", project_id) \
        .single() \
        .execute()

    if res.data:
        bot_token = res.data["bot_token"]
        # Remove webhook
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/deleteWebhook"
        )

    supabase.table("telegram_integrations") \
        .delete() \
        .eq("project_id", project_id) \
        .execute()

    return {"success": True}


# ─────────────────────────────────────────────────────────
# GET TELEGRAM STATUS
# ─────────────────────────────────────────────────────────
@app.get("/telegram/status/{project_id}")
def telegram_status(project_id: str, user=Depends(verify_token)):
    res = supabase.table("telegram_integrations") \
        .select("bot_username, created_at") \
        .eq("project_id", project_id) \
        .execute()

    if res.data:
        return {"connected": True, "bot_username": res.data[0]["bot_username"]}
    return {"connected": False}


# ─────────────────────────────────────────────────────────
# TELEGRAM WEBHOOK — receives messages from Telegram
# ─────────────────────────────────────────────────────────
@app.post("/webhook/telegram/{project_id}")
async def telegram_webhook(project_id: str, req: Request):
    body = await req.json()
    print(f"TELEGRAM BODY: {body}") 
 
    try:
        message = body.get("message") or body.get("edited_message")
        if not message or "text" not in message:
            return {"status": "ignored"}
 
        text = message["text"]
        chat_id = message["chat"]["id"]
        chat_type = message["chat"]["type"]  # "private", "group", "supergroup"
        telegram_user_id = str(message["from"]["id"])
        username = message["from"].get("username") or message["from"].get("first_name", "User")
 
        # Get integration — single DB fetch for everything we need
        res = supabase.table("telegram_integrations") \
            .select("bot_token, bot_username") \
            .eq("project_id", project_id) \
            .single() \
            .execute()
 
        if not res.data:
            return {"error": "integration not found"}
 
        bot_token = res.data["bot_token"]
        bot_username = res.data["bot_username"]
 
        # ── Group chat handling ──────────────────────────────
        if chat_type in ("group", "supergroup"):
            mention = f"@{bot_username}"
            if mention.lower() not in text.lower():
                # Bot not mentioned — ignore completely
                return {"status": "ignored"}
            # Strip the mention so RAG gets a clean query
            text = text.replace(mention, "").replace(mention.lower(), "").strip()
            if not text:
                # Message was just the mention with no question
                send_telegram_message(bot_token, chat_id, "👋 Yes? Ask me anything!")
                return {"status": "ok"}
 
        # ── Handle /start ────────────────────────────────────
        if text.startswith("/start"):
            send_telegram_message(
                bot_token, chat_id,
                f"👋 Hi @{username}! I'm ready to help. Ask me anything!"
            )
            return {"status": "ok"}
 
        # ── Ignore other commands ────────────────────────────
        if text.startswith("/"):
            return {"status": "ignored"}
 
        # ── Get or create per-user chat session ──────────────
        # Use telegram_user_id (not chat_id) so each person has
        # their own memory even in group chats
        chat = supabase.table("chats") \
            .select("id") \
            .eq("project_id", project_id) \
            .eq("external_id", telegram_user_id) \
            .eq("channel", "telegram") \
            .limit(1) \
            .execute()
 
        if chat.data:
            chat_id_db = chat.data[0]["id"]
        else:
            new_chat = supabase.table("chats").insert({
                "project_id": project_id,
                "external_id": telegram_user_id,
                "channel": "telegram",
                "title": f"Telegram @{username}",
            }).execute()
            chat_id_db = new_chat.data[0]["id"]
 
        # ── RAG + reply ──────────────────────────────────────
        history = get_history(chat_id_db, limit=5)
        result = run_chat(project_id, chat_id_db, text, history)
        answer = result["answer"]
 
        send_telegram_message(bot_token, chat_id, answer)
        return {"status": "ok"}
 
    except Exception as e:
        print(f"TELEGRAM WEBHOOK ERROR: {e}")
        return {"status": "error"}

# -------------------------------------------------
# LEAD CAPTURE CONFIG — public (widget fetches on load)
# -------------------------------------------------
@app.get("/public/lead-config/{project_id}")
def get_lead_config(project_id: str):
    res = supabase.table("lead_capture_config") \
        .select("enabled, trigger_after_messages, form_title, form_subtitle") \
        .eq("project_id", project_id) \
        .execute()

    if not res.data or not res.data[0]["enabled"]:
        return {"enabled": False}

    return res.data[0]


# -------------------------------------------------
# LEAD SUBMIT — public (widget posts here)
# -------------------------------------------------
class LeadSubmitRequest(BaseModel):
    project_id: str
    session_id: str
    name: str
    email: str
    phone: str

@app.post("/public/leads")
def submit_lead(req: LeadSubmitRequest):
    # Prevent duplicate for same session
    existing = supabase.table("leads") \
        .select("id") \
        .eq("session_id", req.session_id) \
        .eq("project_id", req.project_id) \
        .execute()

    if existing.data:
        return {"status": "already_captured"}

    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email")

    if len(req.phone.strip()) < 7:
        raise HTTPException(status_code=400, detail="Invalid phone")

    supabase.table("leads").insert({
        "project_id": req.project_id,
        "session_id": req.session_id,
        "name": req.name.strip(),
        "email": req.email.strip(),
        "phone": req.phone.strip(),
        "source": "widget",
    }).execute()

    return {"status": "captured"}


# -------------------------------------------------
# LEAD CONFIG SAVE — dashboard (protected)
# -------------------------------------------------
class LeadConfigRequest(BaseModel):
    projectId: str
    enabled: bool
    triggerAfterMessages: Optional[int] = 2
    formTitle: Optional[str] = "Before we continue..."
    formSubtitle: Optional[str] = "Please share your details to keep chatting."

@app.put("/lead-config")
def save_lead_config(req: LeadConfigRequest, user=Depends(verify_token)):
    supabase.table("lead_capture_config").upsert({
        "project_id": req.projectId,
        "enabled": req.enabled,
        "trigger_after_messages": req.triggerAfterMessages,
        "form_title": req.formTitle,
        "form_subtitle": req.formSubtitle,
    }, on_conflict="project_id").execute()
    return {"status": "saved"}


# -------------------------------------------------
# LEADS LIST — dashboard (protected)
# -------------------------------------------------
@app.get("/leads")
def get_leads(project_id: str, user=Depends(verify_token)):
    res = supabase.table("leads") \
        .select("*") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data


SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
 
 
# ─────────────────────────────────────────────────────────
# SLACK HELPERS
# ─────────────────────────────────────────────────────────
def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify the request is genuinely from Slack."""
    if abs(time.time() - int(timestamp)) > 300:
        return False  # reject requests older than 5 minutes
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(my_sig, signature)
 
 
def send_slack_message(access_token: str, channel: str, text: str):
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"channel": channel, "text": text}
    )
 
 
# ─────────────────────────────────────────────────────────
# SLACK OAUTH — Step 1: Generate auth URL
# ─────────────────────────────────────────────────────────
@app.get("/slack/auth-url")
def slack_auth_url(project_id: str, user=Depends(verify_token)):
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    redirect_uri = f"{frontend_url}/api/slack/callback"
    scopes = "app_mentions:read,chat:write,channels:history,im:history,im:write"
    url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}"
        f"&scope={scopes}"
        f"&redirect_uri={redirect_uri}"
        f"&state={project_id}"
    )
    return {"url": url}
 
 
# ─────────────────────────────────────────────────────────
# SLACK OAUTH — Step 2: Handle callback, exchange code for token
# ─────────────────────────────────────────────────────────
@app.post("/slack/callback")
def slack_callback(data: dict):
    code = data["code"]
    project_id = data["project_id"]
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    redirect_uri = f"{frontend_url}/api/slack/callback"
 
    # Exchange code for access token
    res = requests.post("https://slack.com/api/oauth.v2.access", data={
        "client_id": SLACK_CLIENT_ID,
        "client_secret": SLACK_CLIENT_SECRET,
        "code": code,
        "redirect_uri": redirect_uri,
    })
    token_data = res.json()
 
    if not token_data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Slack OAuth failed: {token_data.get('error')}")
 
    access_token = token_data["access_token"]
    team_id = token_data["team"]["id"]
    team_name = token_data["team"]["name"]
    bot_user_id = token_data["bot_user_id"]
 
    supabase.table("slack_integrations").upsert({
        "project_id": project_id,
        "access_token": access_token,
        "team_id": team_id,
        "team_name": team_name,
        "bot_user_id": bot_user_id,
    }, on_conflict="project_id").execute()
 
    return {"success": True, "team_name": team_name}
 
 
# ─────────────────────────────────────────────────────────
# SLACK STATUS
# ─────────────────────────────────────────────────────────
@app.get("/slack/status/{project_id}")
def slack_status(project_id: str, user=Depends(verify_token)):
    res = supabase.table("slack_integrations") \
        .select("team_name, team_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, "team_name": res.data[0]["team_name"]}
    return {"connected": False}
 
 
# ─────────────────────────────────────────────────────────
# SLACK DISCONNECT
# ─────────────────────────────────────────────────────────
@app.delete("/slack/disconnect/{project_id}")
def slack_disconnect(project_id: str, user=Depends(verify_token)):
    supabase.table("slack_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}
 
 
# ─────────────────────────────────────────────────────────
# SLACK WEBHOOK — receives events from Slack
# ─────────────────────────────────────────────────────────
@app.post("/webhook/slack")
async def slack_webhook(req: Request):
    body_bytes = await req.body()
    body = await req.json()
 
    # Handle Slack URL verification challenge (one-time setup)
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}
 
    # Verify signature
    timestamp = req.headers.get("X-Slack-Request-Timestamp", "")
    signature = req.headers.get("X-Slack-Signature", "")
    if not verify_slack_signature(body_bytes, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
 
    event = body.get("event", {})
    event_type = event.get("type")
 
    # Handle app_mention and direct messages
    if event_type not in ("app_mention", "message"):
        return {"status": "ignored"}
 
    # Ignore bot's own messages
    if event.get("bot_id") or event.get("subtype"):
        return {"status": "ignored"}
 
    text = event.get("text", "").strip()
    channel = event.get("channel")
    user_id = event.get("user")
    team_id = body.get("team_id")
 
    if not text or not channel or not user_id:
        return {"status": "ignored"}
 
    # Find project by team_id
    res = supabase.table("slack_integrations") \
        .select("project_id, access_token, bot_user_id") \
        .eq("team_id", team_id) \
        .single() \
        .execute()
 
    if not res.data:
        return {"error": "integration not found"}
 
    project_id = res.data["project_id"]
    access_token = res.data["access_token"]
    bot_user_id = res.data["bot_user_id"]
 
    # Strip bot mention from text
    text = text.replace(f"<@{bot_user_id}>", "").strip()
    if not text:
        send_slack_message(access_token, channel, "👋 Yes? Ask me anything!")
        return {"status": "ok"}
 
    # Get or create chat session per Slack user
    chat = supabase.table("chats") \
        .select("id") \
        .eq("project_id", project_id) \
        .eq("external_id", user_id) \
        .eq("channel", "slack") \
        .limit(1) \
        .execute()
 
    if chat.data:
        chat_id = chat.data[0]["id"]
    else:
        new_chat = supabase.table("chats").insert({
            "project_id": project_id,
            "external_id": user_id,
            "channel": "slack",
            "title": f"Slack {user_id}",
        }).execute()
        chat_id = new_chat.data[0]["id"]
 
    # RAG + reply
    history = get_history(chat_id, limit=5)
    result = run_chat(project_id, chat_id, text, history)
    send_slack_message(access_token, channel, result["answer"])
 
    return {"status": "ok"}


WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP HELPER — send a message
# ─────────────────────────────────────────────────────────
def send_whatsapp_message(to: str, text: str, phone_number_id: str = None, token: str = None):
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    tok = token or WHATSAPP_TOKEN
    url = f"https://graph.facebook.com/v19.0/{pid}/messages"
    headers = {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    res = requests.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp send error: {res.text}")
    return res
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP WEBHOOK — GET (verification)
# ─────────────────────────────────────────────────────────
@app.get("/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
 
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print("WhatsApp webhook verified")
        return PlainTextResponse(challenge)
 
    raise HTTPException(status_code=403, detail="Verification failed")
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP WEBHOOK — POST (incoming messages)
# ─────────────────────────────────────────────────────────
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    body = await request.json()
 
    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
 
        # Ignore status updates (delivered, read, etc.)
        if "statuses" in value:
            return {"status": "ignored"}
 
        messages = value.get("messages", [])
        if not messages:
            return {"status": "ignored"}
 
        message = messages[0]
        msg_type = message.get("type")
 
        # Only handle text messages
        if msg_type != "text":
            return {"status": "ignored"}
 
        from_number = message["from"]  # e.g. "919876543210"
        text = message["text"]["body"].strip()
        waba_id = value.get("metadata", {}).get("phone_number_id")
 
        # Find project by phone_number_id
        res = supabase.table("whatsapp_integrations") \
            .select("project_id") \
            .eq("phone_number_id", waba_id or WHATSAPP_PHONE_NUMBER_ID) \
            .single() \
            .execute()
 
        if not res.data:
            print(f"No project found for phone_number_id: {waba_id}")
            return {"status": "ignored"}
 
        project_id = res.data["project_id"]
 
        # Get or create chat session per WhatsApp user
        chat = supabase.table("chats") \
            .select("id") \
            .eq("project_id", project_id) \
            .eq("external_id", from_number) \
            .eq("channel", "whatsapp") \
            .limit(1) \
            .execute()
 
        if chat.data:
            chat_id = chat.data[0]["id"]
        else:
            new_chat = supabase.table("chats").insert({
                "project_id": project_id,
                "external_id": from_number,
                "channel": "whatsapp",
                "title": f"WhatsApp {from_number}",
            }).execute()
            chat_id = new_chat.data[0]["id"]
 
        # RAG + reply
        history = get_history(chat_id, limit=5)
        result = run_chat(project_id, chat_id, text, history)
        answer = result["answer"]
 
        send_whatsapp_message(from_number, answer)
        return {"status": "ok"}
 
    except Exception as e:
        print(f"WHATSAPP WEBHOOK ERROR: {e}")
        return {"status": "error"}
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP STATUS — get connected number for a project
# ─────────────────────────────────────────────────────────
@app.get("/whatsapp/status/{project_id}")
def whatsapp_status(project_id: str, user=Depends(verify_token)):
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, display_phone_number, waba_id") \
        .eq("project_id", project_id) \
        .execute()
 
    if res.data:
        return {"connected": True, **res.data[0]}
    return {"connected": False}
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP CONNECT — save phone number + project mapping
# Called after embedded signup or manual setup
# ─────────────────────────────────────────────────────────
@app.post("/whatsapp/connect")
def whatsapp_connect(data: dict, user=Depends(verify_token)):
    project_id = data["projectId"]
    phone_number_id = data["phone_number_id"]
    waba_id = data.get("waba_id", "")
    display_phone_number = data.get("display_phone_number", "")
 
    supabase.table("whatsapp_integrations").upsert({
        "project_id": project_id,
        "phone_number_id": phone_number_id,
        "waba_id": waba_id,
        "display_phone_number": display_phone_number,
    }, on_conflict="project_id").execute()
 
    return {"success": True}
 
 
# ─────────────────────────────────────────────────────────
# WHATSAPP DISCONNECT
# ─────────────────────────────────────────────────────────
@app.delete("/whatsapp/disconnect/{project_id}")
def whatsapp_disconnect(project_id: str, user=Depends(verify_token)):
    supabase.table("whatsapp_integrations") \
        .delete() \
        .eq("project_id", project_id) \
        .execute()
    return {"success": True}

# ─────────────────────────────────────────────────────────
# WHATSAPP ONBOARD — exchange OAuth code for token
# Called after embedded signup FB.login callback
# ─────────────────────────────────────────────────────────
@app.post("/whatsapp/onboard")
def whatsapp_onboard(data: dict, user=Depends(verify_token)):
    code = data["code"]
    project_id = data["projectId"]

    META_APP_ID = os.getenv("META_APP_ID")
    META_APP_SECRET = os.getenv("META_APP_SECRET")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Exchange code for access token
    token_res = requests.get(
        "https://graph.facebook.com/v19.0/oauth/access_token",
        params={
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "code": code,
        }
    )
    token_data = token_res.json()

    if "access_token" not in token_data:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_data}")

    access_token = token_data["access_token"]

    # Get WhatsApp Business Account details
    waba_res = requests.get(
        "https://graph.facebook.com/v19.0/me/businesses",
        params={"access_token": access_token}
    )
    waba_data = waba_res.json()

    # Get phone numbers linked to this WABA
    phone_res = requests.get(
        "https://graph.facebook.com/v19.0/me/phone_numbers",
        params={"access_token": access_token}
    )
    phone_data = phone_res.json()
    phone_number_id = phone_data.get("data", [{}])[0].get("id", "")
    display_phone = phone_data.get("data", [{}])[0].get("display_phone_number", "")
    waba_id = waba_data.get("data", [{}])[0].get("id", "")

    # Save to DB
    supabase.table("whatsapp_integrations").upsert({
        "project_id": project_id,
        "phone_number_id": phone_number_id,
        "waba_id": waba_id,
        "display_phone_number": display_phone,
    }, on_conflict="project_id").execute()

    return {
        "success": True,
        "phone_number_id": phone_number_id,
        "display_phone_number": display_phone,
        "waba_id": waba_id,
    }