from fastapi import APIRouter, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from clients import supabase
from ratelimit import is_rate_limited

bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_project_role(user_id: str, project_id: str):
    """Returns "owner" | "admin" | "agent" | None for a given user + project.
    Mirrors src/lib/supabase-api.js's getProjectRole — the `supabase` client
    here already uses the service-role key, so this bypasses RLS on purpose
    (it only reads ownership/membership rows, never project content)."""
    if not project_id:
        return None
    project = supabase.table("projects").select("user_id").eq("id", project_id).maybe_single().execute()
    project_data = project.data if project else None
    if not project_data:
        return None
    if project_data["user_id"] == user_id:
        return "owner"
    member = (
        supabase.table("project_members")
        .select("role")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    member_data = member.data if member else None
    return member_data["role"] if member_data else None


def require_project_role(user_id: str, project_id: str):
    """Raises 403 if the user has no role on the project; otherwise returns the role."""
    role = get_project_role(user_id, project_id)
    if not role:
        raise HTTPException(status_code=403, detail="Forbidden")
    return role


# -------------------------------------------------
# LOGIN/SIGNUP BRUTE-FORCE PROTECTION
# -------------------------------------------------
# Login/signup themselves stay in Next.js (they use Supabase Auth directly
# via @supabase/ssr for session cookies — not worth re-plumbing just for
# this). But a Next.js API route runs serverless, so an in-memory counter
# there wouldn't reliably persist between requests — this backend process
# is long-running, so the check lives here; the Next.js routes call it
# first, before attempting the real Supabase Auth call.
router = APIRouter()

_AUTH_RATE_LIMITS = {
    "login": (10, 900),     # 10 attempts / 15 min per IP
    "signup": (5, 3600),    # 5 signups / hour per IP
}

class AuthRateLimitCheck(BaseModel):
    action: str

@router.post("/auth/rate-limit-check")
def auth_rate_limit_check(body: AuthRateLimitCheck, request: Request):
    if body.action not in _AUTH_RATE_LIMITS:
        raise HTTPException(status_code=400, detail="Unknown action")
    limit, window = _AUTH_RATE_LIMITS[body.action]
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"{body.action}:{ip}", limit=limit, window_seconds=window):
        raise HTTPException(status_code=429, detail="Too many attempts — please wait and try again.")
    return {"allowed": True}