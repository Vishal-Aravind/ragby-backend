from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from clients import supabase
from auth import verify_token

router = APIRouter()


@router.get("/public/lead-config/{project_id}")
def get_lead_config(project_id: str):
    res = supabase.table("lead_capture_config") \
        .select("enabled, trigger_after_messages, form_title, form_subtitle") \
        .eq("project_id", project_id) \
        .execute()

    if not res.data or not res.data[0]["enabled"]:
        return {"enabled": False}

    return res.data[0]


class LeadSubmitRequest(BaseModel):
    project_id: str
    session_id: str
    name: str
    email: str
    phone: str


@router.post("/public/leads")
def submit_lead(req: LeadSubmitRequest):
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


class LeadConfigRequest(BaseModel):
    projectId: str
    enabled: bool
    triggerAfterMessages: Optional[int] = 2
    formTitle: Optional[str] = "Before we continue..."
    formSubtitle: Optional[str] = "Please share your details to keep chatting."


@router.put("/lead-config")
def save_lead_config(req: LeadConfigRequest, user=Depends(verify_token)):
    supabase.table("lead_capture_config").upsert({
        "project_id": req.projectId,
        "enabled": req.enabled,
        "trigger_after_messages": req.triggerAfterMessages,
        "form_title": req.formTitle,
        "form_subtitle": req.formSubtitle,
    }, on_conflict="project_id").execute()
    return {"status": "saved"}


@router.get("/leads")
def get_leads(project_id: str, user=Depends(verify_token)):
    res = supabase.table("leads") \
        .select("*") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data