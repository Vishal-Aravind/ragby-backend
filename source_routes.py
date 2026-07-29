import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from auth import verify_token
from sources.gsheets import sync_sheet
from sources.postgres import introspect_schema, validate_url
from sources.excel import sync_excel_url, sync_excel_bytes
from sources.website import sync_website

router = APIRouter()


@router.get("/sources")
def list_sources(project_id: str, user=Depends(verify_token)):
    res = supabase.table("data_sources") \
        .select("*") \
        .eq("project_id", project_id) \
        .execute()
    return res.data


@router.post("/sources/add")
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

    # The data_sources row above is inserted BEFORE any of these sync calls
    # run. If a sync call raises (network error, crawler blocked, embedding
    # API failure, etc.) instead of cleanly returning, the row was
    # previously left behind with no actual indexed content — showing up
    # as "connected" in the UI while the bot has nothing to answer from.
    # Wrapping each in try/except so ANY failure is treated the same way:
    # clean up the orphaned row and tell the user honestly, instead of a
    # raw 500 and a silently broken "connected" source.
    try:
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
            if result["pages_indexed"] == 0:
                raise ValueError("Could not crawl this website. It may be blocking automated access.")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"add_source sync error ({data['type']}): {e}")
        supabase.table("data_sources").delete().eq("id", source["id"]).execute()
        raise HTTPException(
            status_code=400,
            detail=str(e) if isinstance(e, ValueError) else "Failed to connect this source. Please check the details and try again."
        )

    return {"id": source["id"], "skipped_tabs": skipped_tabs}


@router.delete("/sources/{source_id}")
def delete_source(source_id: str, user=Depends(verify_token)):
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )
    supabase.table("data_sources").delete().eq("id", source_id).execute()
    return {"status": "deleted"}


@router.post("/sources/sync/{source_id}")
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

    try:
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
            result = sync_website(
                url=cfg["url"],
                project_id=s["project_id"],
                source_id=source_id,
                qdrant=qdrant,
                embeddings=embeddings,
                collection=QDRANT_COLLECTION,
                full_site=cfg.get("full_site", True),
                max_pages=cfg.get("max_pages", 50),
            )
            if result["pages_indexed"] == 0:
                raise ValueError("Could not crawl this website. It may be blocking automated access.")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        # The old points for this source were already deleted above before
        # this ran — a failed resync used to silently report "synced"
        # anyway, leaving the source connected but genuinely empty with no
        # indication anything went wrong.
        print(f"resync_source error ({s['type']}): {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e) if isinstance(e, ValueError) else "Failed to refresh this source. Please try again."
        )

    return {"status": "synced"}


@router.post("/sources/introspect")
def introspect(data: dict, user=Depends(verify_token)):
    db_url = data.get("db_url", "")
    try:
        validate_url(db_url)
        schema = introspect_schema(db_url)
        return {"schema": schema}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=400, detail=f"Could not connect: {str(e)}")


@router.post("/sources/upload-excel")
async def upload_excel(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    label: str = Form(""),
    source_id: str = Form(""),
    user=Depends(verify_token)
):
    file_bytes = await file.read()

    if source_id:
        sync_excel_bytes(file_bytes, projectId, source_id, qdrant, embeddings, QDRANT_COLLECTION)
        supabase.table("data_sources").update({
            "config": {"filename": file.filename},
            "label": label or file.filename,
        }).eq("id", source_id).execute()
        return {"id": source_id, "filename": file.filename}

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