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
    "You are a helpful and friendly AI assistant.\n"
    "Use ONLY the provided context to answer.\n\n"

    "Guidelines:\n"
    "- Be natural, conversational, and clear\n"
    "- Keep answers concise (max 3 sentences)\n"
    "- Do not repeat the question\n\n"

    "If answer is found:\n"
    "- Respond confidently\n\n"

    "If partially found:\n"
    "- Share what is available\n"
    "- Briefly mention what is missing\n\n"

    "If NOT found:\n"
    "- Say: \"I couldn’t find that in your documents.\"\n"
    "- Suggest asking more specifically if helpful\n\n"

    "Strict rules:\n"
    "- No hallucination\n"
    "- No external knowledge"
)


# -------------------------------
# Lightweight intent detection
# -------------------------------
def classify_intent(message: str) -> str:
    msg = message.lower().strip()

    words = msg.split()

    # -------------------------------
    # 1. Greeting (ONLY short messages)
    # -------------------------------
    if len(words) <= 3 and msg in ["hi", "hello", "hey", "hi there", "hello there"]:
        return "greeting"

    # -------------------------------
    # 2. Thanks (ONLY short messages)
    # -------------------------------
    if len(words) <= 4 and any(w in msg for w in ["thanks", "thank you", "thx"]):
        return "thanks"

    # -------------------------------
    # 3. Conversational (memory-type)
    # -------------------------------
    if any(k in msg for k in [
        "earlier", "previous", "you said", "we talked",
        "last message", "first question"
    ]):
        return "conversational"

    # -------------------------------
    # 4. Everything else = RAG
    # -------------------------------
    return "document_query"


def generate_clarification(context, question):
    prompt = f"""
The user asked: "{question}"

The context contains multiple possible answers.

Ask a short clarification question to help the user choose.

Context:
{context}

Return ONLY the question.
"""

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()

# -------------------------------------------------
# Decide if ambiguous
# -------------------------------------------------
def check_ambiguity(context, question):
    prompt = f"""
You are an assistant.

Question: "{question}"

Context:
{context}

Determine:
- If there is ONE clear answer → respond: CLEAR
- If there are MULTIPLE possible answers → respond: AMBIGUOUS

Return ONLY one word.
"""

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()

# -------------------------------
# Main Chat Function
# -------------------------------
def run_chat(project_id: str, message: str, history: List[ChatMessage]):
    try:
        intent = classify_intent(message)

        # -------------------------------
        # 1. Greeting
        # -------------------------------
        if intent == "greeting":
            return {
                "answer": "Hey! 👋 What can I help you with?",
                "sources": []
            }

        # -------------------------------
        # 2. Thanks
        # -------------------------------
        if intent == "thanks":
            return {
                "answer": "You're welcome! 😊 Let me know if you need anything else.",
                "sources": []
            }

        # -------------------------------
        # 3. Conversational (NO RAG)
        # -------------------------------
        if intent == "conversational":
            messages = [{"role": "system", "content": "You are a helpful assistant."}]

            for h in history:
                messages.append({"role": h.role, "content": h.content})

            messages.append({"role": "user", "content": message})

            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=120,
            )

            return {
                "answer": completion.choices[0].message.content.strip(),
                "sources": []
            }

        # -------------------------------
        # 4. RAG Retrieval
        # -------------------------------
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

        # -------------------------------
        # 5. No results
        # -------------------------------
        if not hits:
            return {
                "answer": "I couldn’t find that in your documents. Try asking more specifically or upload related content.",
                "sources": []
            }

        # -------------------------------
        # 6. Build context
        # -------------------------------
        context = "\n\n---\n\n".join(
            h.payload.get("text", "") for h in hits
        )

        # -------------------------------
        # 7. 🔥 LLM Ambiguity Check (FIXED)
        # -------------------------------
        ambiguity_check_prompt = f"""
Question: "{message}"

Context:
{context}

Check carefully:

- If the question can have MORE THAN ONE valid answer from the context (e.g., multiple companies, multiple packages, multiple values), respond: AMBIGUOUS
- If there is ONLY ONE correct answer, respond: CLEAR

Important:
Even if one answer seems more prominent, if another valid answer exists, it is AMBIGUOUS.

Return ONLY:
CLEAR or AMBIGUOUS
"""

        decision_res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ambiguity_check_prompt}],
            temperature=0
        )

        decision = decision_res.choices[0].message.content.strip()

        # -------------------------------
        # 8. If ambiguous → ask clarification
        # -------------------------------
        if decision == "AMBIGUOUS":
            return {
                "answer": generate_clarification(context, message),
                "sources": []
            }

        # -------------------------------
        # 9. Normal Answer (CLEAR case)
        # -------------------------------
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for h in history:
            messages.append({"role": h.role, "content": h.content})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{message}"
        })

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=120,
        )

        answer = completion.choices[0].message.content.strip()

        if answer and not answer.endswith((".", "!", "?")):
            answer += "."

        return {
            "answer": answer,
            "sources": []
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

