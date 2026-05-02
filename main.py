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
    chatId: str
    message: str
    

class PublicChatRequest(BaseModel):
    projectId: str
    message: str


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


def get_history(chat_id: str, limit: int = 5):
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
# LOGIC FOR CHAT
# -------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful RAG AI assistant based on domain. Use ONLY the provided context and recent conversation if relevant. If user asks about you introduce yourself politely\n\n"

    "Style:\n"
    "- Simple question → short answer\n"
    "- Complex question → structured answer no formatting medium length\n"
    "- Be concise, do not over-explain unless necessary\n\n"

    "Logic:\n"
    "- One clear answer → answer directly\n"
    "- Multiple answers → ask a clarification question briefly\n"
    "- Partial info → answer + mention missing briefly\n"
    "- No answer → say that you couldn't find the information and ask if they have more information briefly\n\n"
    "- Match answer length to question complexity\n"
    
    "Formatting rules:\n"
    "- Use bullet points ONLY when needed\n"

    "Rules:\n"
    "- No hallucination\n"
    "- No external knowledge"
    "- You will be feeded with previous conversation history also, take context from their if needed "
)

# -------------------------------
# Lightweight intent detection
# -------------------------------
def classify_intent(message: str) -> str:
    msg = message.lower().strip()
    words = msg.split()

    if len(words) <= 3 and msg in ["hi", "hello", "hey", "hi there", "hello there"]:
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

    if res.data:
        return res.data.get("domain")

    return None


# -------------------------------
# Main Chat Function 
# -------------------------------
def run_chat(project_id: str, chat_id: str, message: str, history):
    try:
        history = history or []
        domain = get_project_domain(project_id)

        system_prompt = SYSTEM_PROMPT

        if domain:
            system_prompt += f"\n\nDomain:\n- You are specialized in {domain}."

        intent = classify_intent(message)

        # -------------------------------
        # 1. Greeting
        # -------------------------------
        if intent == "greeting":
            answer = "Hey! 👋 What can I help you with?"
            save_message(chat_id, "assistant", answer)

            return {"answer": answer, "sources": []}

        # save user message
        save_message(chat_id, "user", message)

        # -------------------------------
        # 2. Thanks
        # -------------------------------
        if intent == "thanks":
            answer = "You're welcome! 😊"
            save_message(chat_id, "assistant", answer)

            return {"answer": answer, "sources": []}

        # -------------------------------
        # 3. Conversational
        # -------------------------------
        if intent == "conversational":
            messages = [{"role": "system", "content": system_prompt}]

            for h in history[-7:]:
                messages.append({
                    "role": h["role"],
                    "content": h["content"]
                })

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

        # -------------------------------
        # 4. RAG Retrieval
        # -------------------------------
        query_for_embedding = message

        if len(message.split()) <= 4 and history:
            last_user_msgs = [m for m in history if m["role"] == "user"]
            if last_user_msgs:
                query_for_embedding = last_user_msgs[-1]["content"] + " " + message

        q = embeddings.embed_query(query_for_embedding)

        res = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q,
            limit=7,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )]
            )
        )

        hits = res.points

        if not hits:
            answer = "I couldn’t find that in your documents."
            save_message(chat_id, "assistant", answer)

            return {"answer": answer, "sources": []}

        context = "\n\n---\n\n".join(
            f"{h.payload.get('text', '')}"
            for h in hits
        )

        messages = [{"role": "system", "content": system_prompt}]

        for h in history[-7:]:
            messages.append({
                "role": h["role"],
                "content": h["content"]
            })

        # ✅ Add current query WITH context
        messages.append({
            "role": "user",
            "content": f"""
        Context:
        {context}

        Question:
        {message}
        """
        })

        import json

        print("\n================ LLM INPUT ================\n")
        print(json.dumps(messages, indent=2))
        print("\n===========================================\n")

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

@app.post("/chat")
def chat(req: ChatRequest):
    history = get_history(req.chatId, limit=7)
    return run_chat(req.projectId, req.chatId, req.message, history)


@app.post("/public/chat")
def public_chat(req: PublicChatRequest):
    chat_id = str(uuid.uuid4())  # or session-based ID
    history = []
    return run_chat(req.projectId, chat_id, req.message, history)


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


@app.post("/whatsapp/onboard")
def whatsapp_onboard(data: dict):
    code = data["code"]
    project_id = data["projectId"]

    url = "https://graph.facebook.com/v19.0/oauth/access_token"

    params = {
        "client_id": os.getenv("META_APP_ID"),
        "client_secret": os.getenv("META_APP_SECRET"),
        "code": code,
    }

    res = requests.get(url, params=params)
    token_data = res.json()

    access_token = token_data.get("access_token")

    # TEMP store token (update later when metadata arrives)
    supabase.table("whatsapp_integrations").upsert({
        "project_id": project_id,
        "access_token": access_token
    }).execute()

    return {"success": True}

@app.post("/whatsapp/save-metadata")
def save_metadata(data: dict):
    project_id = data["projectId"]

    supabase.table("whatsapp_integrations").upsert({
        "project_id": project_id,
        "phone_number_id": data["phone_number_id"],
        "waba_id": data["waba_id"]
    }).execute()

    return {"success": True}

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(req: Request):
    body = await req.json()

    try:
        value = body["entry"][0]["changes"][0]["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        msg = value["messages"][0]

        # ignore non-text
        if msg.get("type") != "text":
            return {"status": "ignored"}

        phone = msg["from"]
        text = msg["text"]["body"]
        phone_number_id = value["metadata"]["phone_number_id"]

        # -------------------------------
        # 1. Get project
        # -------------------------------
        res = supabase.table("whatsapp_integrations") \
            .select("project_id, access_token") \
            .eq("phone_number_id", phone_number_id) \
            .single() \
            .execute()

        if not res.data:
            return {"error": "integration not found"}

        project_id = res.data["project_id"]
        access_token = res.data["access_token"]

        # -------------------------------
        # 2. Get or create chat
        # -------------------------------
        chat = supabase.table("chats") \
            .select("id") \
            .eq("project_id", project_id) \
            .eq("external_id", phone) \
            .eq("channel", "whatsapp") \
            .limit(1) \
            .execute()

        if chat.data:
            chat_id = chat.data[0]["id"]
        else:
            new_chat = supabase.table("chats").insert({
                "project_id": project_id,
                "external_id": phone,
                "channel": "whatsapp",
                "title": f"WhatsApp {phone}"
            }).execute()

            chat_id = new_chat.data[0]["id"]

        # -------------------------------
        # 3. Memory + RAG
        # -------------------------------
        history = get_history(chat_id, 5)
        result = run_chat(project_id, chat_id, text, history)

        answer = result["answer"]

        # -------------------------------
        # 4. Send reply
        # -------------------------------
        url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "messaging_product": "whatsapp",
            "to": phone,
            "text": {"body": answer}
        }

        requests.post(url, headers=headers, json=payload)

        return {"status": "ok"}

    except Exception as e:
        print("ERROR:", e)
        return {"status": "error"}

@app.get("/webhook/whatsapp")
def verify(mode: str = Query(...), challenge: str = Query(...), verify_token: str = Query(...)):
    if verify_token == os.getenv("VERIFY_TOKEN"):
        return challenge
    return "error"