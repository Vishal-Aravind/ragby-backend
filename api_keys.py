import secrets
from fastapi import APIRouter, Depends, HTTPException, Request
from clients import supabase
from auth import verify_token, require_project_role
from whatsapp import send_whatsapp_message

router = APIRouter()


def generate_key():
    return f"ak_{secrets.token_urlsafe(32)}"


def get_project_by_key(api_key: str):
    """Validate API key and return project_id."""
    res = supabase.table("api_keys") \
        .select("project_id, id") \
        .eq("key", api_key) \
        .limit(1) \
        .execute()
    if not res.data:
        return None
    # Update last_used_at
    supabase.table("api_keys") \
        .update({"last_used_at": "now()"}) \
        .eq("id", res.data[0]["id"]) \
        .execute()
    return res.data[0]["project_id"]


# -------------------------------------------------
# MANAGEMENT ENDPOINTS (authenticated)
# -------------------------------------------------
@router.get("/api-keys/{project_id}")
def get_api_key(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("api_keys") \
        .select("id, key, name, created_at, last_used_at") \
        .eq("project_id", project_id) \
        .execute()

    if not res.data:
        # Auto-create key if none exists
        key = generate_key()
        res = supabase.table("api_keys").insert({
            "project_id": project_id,
            "key": key,
            "name": "Default",
        }).execute()
        return res.data[0]

    return res.data[0]


@router.post("/api-keys/{project_id}/regenerate")
def regenerate_api_key(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    key = generate_key()
    existing = supabase.table("api_keys") \
        .select("id") \
        .eq("project_id", project_id) \
        .execute()

    if existing.data:
        res = supabase.table("api_keys") \
            .update({"key": key}) \
            .eq("project_id", project_id) \
            .execute()
    else:
        res = supabase.table("api_keys").insert({
            "project_id": project_id,
            "key": key,
            "name": "Default",
        }).execute()

    return {"key": key}


# -------------------------------------------------
# PUBLIC SEND ENDPOINT (API key auth)
# -------------------------------------------------
@router.post("/public/send")
async def public_send(request: Request):
    # Get API key from header
    api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    project_id = get_project_by_key(api_key)
    if not project_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    to      = body.get("to", "").strip()
    message = body.get("message", "").strip()

    if not to:
        raise HTTPException(status_code=400, detail="'to' phone number is required")
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required")

    # Normalize phone number
    if not to.startswith("+"):
        to = f"+{to}"
    to = to.replace(" ", "").replace("-", "")

    # Get WhatsApp integration for this project
    wa = supabase.table("whatsapp_integrations") \
        .select("phone_number_id") \
        .eq("project_id", project_id) \
        .execute()

    if not wa.data:
        raise HTTPException(status_code=400, detail="WhatsApp not connected for this project")

    phone_number_id = wa.data[0]["phone_number_id"]

    # Check rate limit
    from usage import check_rate_limit, increment_usage
    rate_check = check_rate_limit(project_id)
    if not rate_check["allowed"]:
        raise HTTPException(status_code=429, detail="Monthly message limit reached")

    # Send message
    res = send_whatsapp_message(to, message, phone_number_id)
    increment_usage(project_id)

    if res and res.ok:
        return {"status": "sent", "to": to}
    else:
        raise HTTPException(status_code=500, detail="Failed to send WhatsApp message")