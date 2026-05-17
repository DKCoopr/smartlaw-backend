"""
Admin API — system health, user management, stats, error logs.
All endpoints require profiles.role = 'admin'.
"""
import time
from typing import Optional, Literal
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr, Field

from app.auth import require_admin, get_current_user_id
from app.database import get_supabase
from app.config import get_settings

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Schemas ──────────────────────────────────────────────────────────────────
class HealthCheck(BaseModel):
    name: str
    ok: bool
    latency_ms: Optional[int] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    checks: list[HealthCheck]
    checked_at: str


class StatsResponse(BaseModel):
    users_total: int
    users_active: int
    users_disabled: int
    cases_total: int
    documents_total: int
    documents_summarized: int
    transactions_total: int
    billings_total: int
    cases_last_7d: int
    documents_last_7d: int


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str = "lawyer"
    disabled: bool = False
    created_at: Optional[str] = None
    last_sign_in_at: Optional[str] = None


class UserCreateIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: Optional[str] = None
    role: Literal["lawyer", "officer", "admin"] = "lawyer"


class UserPatchIn(BaseModel):
    password: Optional[str] = Field(default=None, min_length=6)
    full_name: Optional[str] = None
    role: Optional[Literal["lawyer", "officer", "admin"]] = None
    disabled: Optional[bool] = None


class ErrorLogIn(BaseModel):
    level: Literal["error", "warn", "info"] = "error"
    message: str
    stack: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    context: Optional[dict] = None


# ── Health checks ────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def admin_health(_: str = Depends(require_admin)):
    """Probe each external integration. Each check is wrapped so one failure
    doesn't crash the whole response."""
    settings = get_settings()
    checks: list[HealthCheck] = []

    # Supabase
    t0 = time.time()
    try:
        db = get_supabase()
        db.table("profiles").select("id", count="exact").limit(1).execute()
        checks.append(HealthCheck(name="supabase", ok=True, latency_ms=int((time.time() - t0) * 1000)))
    except Exception as e:
        checks.append(HealthCheck(name="supabase", ok=False, detail=str(e)[:200]))

    # OpenAI (ChatGPT)
    if settings.openai_api_key:
        t0 = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                )
            ok = r.status_code == 200
            checks.append(HealthCheck(
                name="openai", ok=ok,
                latency_ms=int((time.time() - t0) * 1000),
                detail=None if ok else f"HTTP {r.status_code}",
            ))
        except Exception as e:
            checks.append(HealthCheck(name="openai", ok=False, detail=str(e)[:200]))
    else:
        checks.append(HealthCheck(name="openai", ok=False, detail="API key not configured"))

    # Anthropic (Claude)
    if settings.anthropic_api_key:
        t0 = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": settings.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
            ok = r.status_code == 200
            checks.append(HealthCheck(
                name="claude", ok=ok,
                latency_ms=int((time.time() - t0) * 1000),
                detail=None if ok else f"HTTP {r.status_code}",
            ))
        except Exception as e:
            checks.append(HealthCheck(name="claude", ok=False, detail=str(e)[:200]))
    else:
        checks.append(HealthCheck(name="claude", ok=False, detail="API key not configured"))

    # Google (Gemini)
    if settings.google_api_key:
        t0 = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.google_api_key}",
                )
            ok = r.status_code == 200
            checks.append(HealthCheck(
                name="gemini", ok=ok,
                latency_ms=int((time.time() - t0) * 1000),
                detail=None if ok else f"HTTP {r.status_code}",
            ))
        except Exception as e:
            checks.append(HealthCheck(name="gemini", ok=False, detail=str(e)[:200]))
    else:
        checks.append(HealthCheck(name="gemini", ok=False, detail="API key not configured"))

    return HealthResponse(checks=checks, checked_at=datetime.now(timezone.utc).isoformat())


# ── Stats ────────────────────────────────────────────────────────────────────
@router.get("/stats", response_model=StatsResponse)
async def admin_stats(_: str = Depends(require_admin)):
    db = get_supabase()
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    def _count(table: str, **filters) -> int:
        try:
            q = db.table(table).select("id", count="exact")
            for k, v in filters.items():
                if k.endswith("__neq"):
                    q = q.neq(k[:-5], v)
                elif k.endswith("__not_null"):
                    q = q.not_.is_(k[:-10], "null")
                elif k.endswith("__gte"):
                    q = q.gte(k[:-5], v)
                elif k.endswith("__eq"):
                    q = q.eq(k[:-4], v)
                else:
                    q = q.eq(k, v)
            return q.execute().count or 0
        except Exception:
            return 0

    # User counts via profiles (auth.users isn't directly queryable from PostgREST)
    try:
        users_all = db.table("profiles").select("id, role").execute().data or []
        users_total = len(users_all)
        # 'disabled' tracked via metadata on auth user — we approximate via profile.role='disabled'
        users_disabled = sum(1 for u in users_all if u.get("role") == "disabled")
        users_active = users_total - users_disabled
    except Exception:
        users_total = users_active = users_disabled = 0

    return StatsResponse(
        users_total=users_total,
        users_active=users_active,
        users_disabled=users_disabled,
        cases_total=_count("cases", status__neq="deleted"),
        documents_total=_count("documents"),
        documents_summarized=_count("documents", is_processed=True),
        transactions_total=_count("transactions"),
        billings_total=_count("billings"),
        cases_last_7d=_count("cases", created_at__gte=week_ago),
        documents_last_7d=_count("documents", created_at__gte=week_ago),
    )


# ── Users ────────────────────────────────────────────────────────────────────
@router.get("/users")
async def list_users(_: str = Depends(require_admin)):
    """List all users by joining auth.users (admin API) with profiles."""
    db = get_supabase()
    out: list[UserOut] = []
    try:
        # supabase-py 2.x exposes auth.admin.list_users()
        page = db.auth.admin.list_users()
        # page may be a list directly or a paginated wrapper
        users = page if isinstance(page, list) else getattr(page, "users", []) or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list users: {e}")

    # Pull profiles in one batch
    try:
        prof_rows = db.table("profiles").select("*").execute().data or []
        profiles = {p["id"]: p for p in prof_rows}
    except Exception:
        profiles = {}

    for u in users:
        prof = profiles.get(u.id, {})
        meta = (u.user_metadata or {}) if hasattr(u, "user_metadata") else {}
        # 'disabled' flag stored in app_metadata.disabled (set by patch endpoint)
        app_meta = (u.app_metadata or {}) if hasattr(u, "app_metadata") else {}
        out.append(UserOut(
            id=u.id,
            email=u.email or "",
            full_name=prof.get("full_name") or meta.get("full_name"),
            role=prof.get("role") or "lawyer",
            disabled=bool(app_meta.get("disabled", False)),
            created_at=str(u.created_at) if getattr(u, "created_at", None) else None,
            last_sign_in_at=str(u.last_sign_in_at) if getattr(u, "last_sign_in_at", None) else None,
        ))
    # Sort: admins first, then by email
    out.sort(key=lambda x: (x.role != "admin", x.email))
    return {"users": [u.model_dump() for u in out]}


@router.post("/users")
async def create_user(payload: UserCreateIn, _: str = Depends(require_admin)):
    db = get_supabase()
    try:
        created = db.auth.admin.create_user({
            "email": payload.email,
            "password": payload.password,
            "email_confirm": True,
            "user_metadata": {"full_name": payload.full_name} if payload.full_name else {},
        })
        new_user = created.user if hasattr(created, "user") else created
        if not new_user or not getattr(new_user, "id", None):
            raise HTTPException(status_code=500, detail="User created but no id returned")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Create user failed: {e}")

    # Upsert profile with role + name
    try:
        db.table("profiles").upsert({
            "id": new_user.id,
            "full_name": payload.full_name,
            "role": payload.role,
        }).execute()
    except Exception as e:
        # User exists but profile failed — surface the error but don't roll back
        raise HTTPException(status_code=500, detail=f"User created, profile upsert failed: {e}")

    return {"id": new_user.id, "email": new_user.email}


@router.patch("/users/{user_id}")
async def patch_user(user_id: str, payload: UserPatchIn, admin_id: str = Depends(require_admin)):
    db = get_supabase()
    auth_updates: dict = {}
    if payload.password:
        auth_updates["password"] = payload.password
    if payload.full_name is not None:
        auth_updates["user_metadata"] = {"full_name": payload.full_name}
    if payload.disabled is not None:
        # Self-protection: don't let an admin disable themselves
        if payload.disabled and user_id == admin_id:
            raise HTTPException(status_code=400, detail="Cannot disable yourself")
        auth_updates["app_metadata"] = {"disabled": payload.disabled}
        # Supabase 'ban' user as well so they can't sign in
        auth_updates["ban_duration"] = "876600h" if payload.disabled else "none"

    if auth_updates:
        try:
            db.auth.admin.update_user_by_id(user_id, auth_updates)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Auth update failed: {e}")

    # Profile updates
    profile_updates: dict = {}
    if payload.full_name is not None:
        profile_updates["full_name"] = payload.full_name
    if payload.role is not None:
        profile_updates["role"] = payload.role

    if profile_updates:
        try:
            db.table("profiles").upsert({"id": user_id, **profile_updates}).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Profile update failed: {e}")

    return {"ok": True}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin_id: str = Depends(require_admin)):
    """Per project policy: do NOT hard-delete. Disable instead."""
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot disable yourself")
    db = get_supabase()
    try:
        db.auth.admin.update_user_by_id(user_id, {
            "ban_duration": "876600h",
            "app_metadata": {"disabled": True},
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Disable failed: {e}")
    return {"ok": True, "disabled": True}


# ── Error logs ───────────────────────────────────────────────────────────────
@router.post("/errors")
async def log_error(payload: ErrorLogIn, request: Request, user_id: str = Depends(get_current_user_id)):
    """Frontend reports JS errors here. Open to any authenticated user (not admin-only)
    so we can capture anonymous user errors. Reads, however, require admin."""
    db = get_supabase()
    try:
        db.table("error_logs").insert({
            "user_id": user_id,
            "level": payload.level,
            "message": payload.message[:1000],
            "stack": (payload.stack or "")[:8000] or None,
            "url": payload.url,
            "user_agent": payload.user_agent or request.headers.get("user-agent"),
            "context": payload.context or {},
        }).execute()
    except Exception as e:
        # Don't fail the request — error logging is best-effort
        return {"ok": False, "detail": str(e)[:200]}
    return {"ok": True}


@router.get("/errors")
async def list_errors(
    limit: int = 100,
    level: Optional[Literal["error", "warn", "info"]] = None,
    _: str = Depends(require_admin),
):
    db = get_supabase()
    try:
        q = db.table("error_logs").select("*").order("created_at", desc=True).limit(limit)
        if level:
            q = q.eq("level", level)
        return {"errors": q.execute().data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/errors")
async def clear_errors(_: str = Depends(require_admin)):
    db = get_supabase()
    try:
        # PostgREST requires a filter on bulk delete. created_at is a NOT NULL
        # timestamptz that's always populated, so a far-past lower bound nukes
        # everything. (Previously used .gte("id", 0) which silently no-op'd on
        # UUID primary keys.)
        db.table("error_logs").delete().gte("created_at", "1970-01-01T00:00:00Z").execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


# ── Whoami (used by frontend to check admin status) ──────────────────────────
@router.get("/me")
async def admin_me(user_id: str = Depends(get_current_user_id)):
    """Returns profile + isAdmin flag for the current session.
    Open to any authenticated user — used by the frontend to decide whether to
    show the Admin Panel button."""
    db = get_supabase()
    try:
        res = db.table("profiles").select("*").eq("id", user_id).single().execute()
        prof = res.data or {}
    except Exception:
        prof = {}
    return {
        "user_id": user_id,
        "role": prof.get("role") or "lawyer",
        "is_admin": prof.get("role") == "admin",
        "full_name": prof.get("full_name"),
    }
