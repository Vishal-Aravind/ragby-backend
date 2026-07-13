"""
Unanswered Questions ("Content Gaps") — the "Answer & train the bot" action.

Listing/grouping the unanswered questions themselves is done entirely on the
Next.js side (src/app/api/analytics/unanswered-questions/route.js) by scanning
existing chat_messages for the bot's fallback string — no logging table
needed for that. This file only handles the one action that actually needs
the Python backend: embedding a merchant-written answer and adding it to the
project's knowledge base, exactly like a normal document upload would.
"""
import secrets
import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from auth import verify_token

router = APIRouter()


class FaqAnswerRequest(BaseModel):
    project_id: str
    question: str
    answer: str


@router.post("/content-gaps/answer")
def answer_content_gap(req: FaqAnswerRequest, user=Depends(verify_token)):
    text = f"Q: {req.question}\nA: {req.answer}"
    filename = f"FAQ: {req.question[:50]} #{secrets.token_hex(4)}"

    # Create the files row first — this is what makes the answer show up
    # in the Documents tab file list, and lets it be deleted later through
    # the existing DELETE /document/{file_id} endpoint (which only touches
    # Qdrant + this row, never storage, so a placeholder path is safe).
    file_res = supabase.table("files").insert({
        "project_id": req.project_id,
        "user_id": user.id,
        "filename": filename,
        "storage_path": f"faq/{uuid.uuid4()}",
        "status": "indexed",
    }).execute()
    file_id = file_res.data[0]["id"]

    vector = embeddings.embed_documents([text])[0]
    qdrant.upload_points(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "project_id": req.project_id,
                    "file_id": file_id,
                    "filename": filename,
                    "page_number": 1,
                    "source_type": "document",
                    "text": text,
                },
            )
        ],
    )

    question_key = req.question.strip().lower()
    supabase.table("content_gap_resolutions").upsert({
        "project_id": req.project_id,
        "question_key": question_key,
        "resolved_by": user.id,
    }, on_conflict="project_id,question_key").execute()

    return {"status": "added", "file_id": file_id}
