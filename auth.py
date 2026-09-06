from fastapi import APIRouter, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from clients import supabase
from ratelimit import is_rate_limited, client_ip

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


def get_project_access(user_id: str, project_id: str):
    """Returns (role, permissions) for a user + project; (None, []) if no access.
    Mirrors src/lib/supabase-api.js's getProjectAccess — the `supabase` client
    here already uses the service-role key, so this bypasses RLS on purpose
    (it only reads ownership/membership rows, never project content)."""
    if not project_id:
        return None, []
    project = supabase.table("projects").select("user_id").eq("id", project_id).maybe_single().execute()
    project_data = project.data if project else None
    if not project_data:
        return None, []
    if project_data["user_id"] == user_id:
        return "owner", []
    member = (
        supabase.table("project_members")
        .select("role, permissions")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    member_data = member.data if member else None
    if not member_data:
        return None, []
    return member_data["role"], (member_data.get("permissions") or [])


def get_project_role(user_id: str, project_id: str):
    """Returns "owner" | "admin" | "agent" | None for a given user + project."""
    role, _ = get_project_access(user_id, project_id)
    return role


def require_project_role(user_id: str, project_id: str):
    """Raises 403 if the user has no role on the project; otherwise returns the role."""
    role = get_project_role(user_id, project_id)
    if not role:
        raise HTTPException(status_code=403, detail="Forbidden")
    return role


# Mirrors src/lib/project-access.js's hasProjectTabAccess EXACTLY — the two
# must agree, or the UI will show a tab the API rejects (or worse, hide one
# the API allows). Keep these three sets in sync with that file.
_ALWAYS_ALLOWED_TABS = {"leads", "conversations"}
_OWNER_ADMIN_ONLY_TABS = {"team"}
_ROLE_RANK = {"agent": 1, "admin": 2, "owner": 3}


def require_project_access(user_id: str, project_id: str, tab: str = None, min_role: str = None):
    """The real authorization gate for project-scoped endpoints.

    Until this existed, `permissions` was never read anywhere in the backend
    and no call site ever compared the returned role — so tab permissions
    were frontend decoration, and an agent could curl any endpoint an owner
    could, including destructive ones. `tab` enforces the per-member
    permission grid; `min_role` gates destructive operations.
    """
    role, permissions = get_project_access(user_id, project_id)
    if not role:
        raise HTTPException(status_code=403, detail="Forbidden")

    if min_role and _ROLE_RANK.get(role, 0) < _ROLE_RANK.get(min_role, 99):
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to do this. Ask a project admin.",
        )

    if tab:
        is_owner_or_admin = role in ("owner", "admin")
        if tab in _OWNER_ADMIN_ONLY_TABS:
            allowed = is_owner_or_admin
        elif tab in _ALWAYS_ALLOWED_TABS:
            allowed = True
        else:
            allowed = is_owner_or_admin or tab in permissions
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this section of the project.",
            )

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
    "login": (10, 900),        # 10 attempts / 15 min per IP
    "signup": (5, 3600),       # 5 signups / hour per IP
    "team_invite": (10, 600),  # 10 invite attempts / 10 min per IP — the
                                # invite endpoint's "no account found for
                                # that email" response is an email-existence
                                # oracle; this caps how fast it can be probed.
}

class AuthRateLimitCheck(BaseModel):
    action: str

@router.post("/auth/rate-limit-check")
def auth_rate_limit_check(body: AuthRateLimitCheck, request: Request):
    if body.action not in _AUTH_RATE_LIMITS:
        raise HTTPException(status_code=400, detail="Unknown action")
    limit, window = _AUTH_RATE_LIMITS[body.action]
    ip = client_ip(request)
    if is_rate_limited(f"{body.action}:{ip}", limit=limit, window_seconds=window):
        raise HTTPException(status_code=429, detail="Too many attempts — please wait and try again.")
    return {"allowed": True}