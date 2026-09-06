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

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from auth import verify_token, require_project_access

router = APIRouter()


class FaqAnswerRequest(BaseModel):
    project_id: str
    question: str
    answer: str


@router.post("/content-gaps/answer")
def answer_content_gap(req: FaqAnswerRequest, user=Depends(verify_token)):
    require_project_access(user.id, req.project_id, tab="documents")
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
        "status": "processing",
    }).execute()
    file_id = file_res.data[0]["id"]

    # The status was previously written as "indexed" up front, before the
    # embedding call below — so an OpenAI or Qdrant failure left a row that
    # looked perfectly healthy in the Documents tab with no vectors behind
    # it, indistinguishable from a working one. Flip to "indexed" only once
    # the content is genuinely searchable, and drop the row if it isn't.
    try:
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
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"content gap answer failed for project {req.project_id}: {e}")
        supabase.table("files").delete().eq("id", file_id).execute()
        raise HTTPException(
            status_code=502,
            detail="Couldn't save that answer right now. Please try again.",
        )

    supabase.table("files").update({"status": "indexed"}).eq("id", file_id).execute()

    question_key = req.question.strip().lower()
    supabase.table("content_gap_resolutions").upsert({
        "project_id": req.project_id,
        "question_key": question_key,
        "resolved_by": user.id,
    }, on_conflict="project_id,question_key").execute()

    return {"status": "added", "file_id": file_id}
