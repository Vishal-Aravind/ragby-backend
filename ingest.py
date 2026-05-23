import io
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import pdfplumber
from docx import Document
from pptx import Presentation
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from auth import verify_token

router = APIRouter()


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class IngestRequest(BaseModel):
    projectId: str
    filename: str
    filePath: str


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


EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "ppt": extract_pptx,
    "pptx": extract_pptx,
    "xls": extract_excel,
    "xlsx": extract_excel,
    "txt": extract_txt,
}


# -------------------------------------------------
# INGEST ENDPOINT
# -------------------------------------------------
@router.post("/ingest")
def ingest(req: IngestRequest, user=Depends(verify_token)):
    row = supabase.table("files") \
        .select("id") \
        .eq("project_id", req.projectId) \
        .eq("filename", req.filename) \
        .execute()

    if not row.data:
        return {"error": "file not found"}

    file_id = row.data[0]["id"]
    supabase.table("files").update({"status": "processing"}).eq("id", file_id).execute()

    b = supabase.storage.from_("documents").download(req.filePath)
    ext = req.filename.lower().split(".")[-1]

    extractor = EXTRACTORS.get(ext)
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
                "source_type": "document",
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


@router.delete("/document/{file_id}")
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