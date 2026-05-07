"""JWT auth dependency for protected routes."""
from fastapi import HTTPException, Header, Depends
from typing import Optional
from app.database import get_supabase_anon, get_supabase


def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency: extract authenticated user_id from Supabase JWT.
    Raises 401 if missing/invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token")

    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")

    try:
        anon = get_supabase_anon()
        user_response = anon.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_response.user.id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth verification failed: {str(e)}")


def require_admin(user_id: str = Depends(get_current_user_id)) -> str:
    """
    FastAPI dependency: assert that the authenticated user has role='admin'
    in public.profiles. Returns the user_id on success, 403 otherwise.
    """
    db = get_supabase()
    try:
        res = db.table("profiles").select("role").eq("id", user_id).single().execute()
        role = (res.data or {}).get("role")
    except Exception:
        # Profile row missing — treat as non-admin
        raise HTTPException(status_code=403, detail="Admin access required")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user_id
