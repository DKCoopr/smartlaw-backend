"""
CRUD routes for cases stored in Supabase
GET    /api/cases          → list user's cases
POST   /api/cases          → create new case
GET    /api/cases/{id}     → get single case
PUT    /api/cases/{id}     → update case
DELETE /api/cases/{id}     → delete case
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.models.case import CaseCreate, CaseOut, CaseListOut
from app.database import get_supabase

router = APIRouter(prefix="/api", tags=["cases"])


def _get_user_id(authorization: Optional[str]) -> str:
    """Extract user ID from Supabase JWT token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token")
    # Supabase handles JWT verification — we just pass the token along
    # For full implementation, decode JWT here to get user_id
    # For now, return a placeholder — will be replaced with real JWT decode
    return "user_placeholder"


@router.get("/cases", response_model=CaseListOut)
async def list_cases(
    limit: int = 20,
    offset: int = 0,
    authorization: Optional[str] = Header(None),
):
    """List all cases for the authenticated user"""
    user_id = _get_user_id(authorization)
    db = get_supabase()

    try:
        response = (
            db.table("cases")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        count_response = db.table("cases").select("id", count="exact").eq("user_id", user_id).execute()

        return CaseListOut(
            cases=response.data or [],
            total=count_response.count or 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cases", response_model=CaseOut)
async def create_case(
    case: CaseCreate,
    authorization: Optional[str] = Header(None),
):
    """Save a new case to Supabase"""
    user_id = _get_user_id(authorization)
    db = get_supabase()

    try:
        payload = {
            "user_id": user_id,
            "title": case.title,
            "transcript": case.transcript,
            "form_data": case.form_data.model_dump(),
            "status": "open",
        }
        response = db.table("cases").insert(payload).execute()
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cases/{case_id}", response_model=CaseOut)
async def get_case(
    case_id: str,
    authorization: Optional[str] = Header(None),
):
    """Get a single case by ID"""
    user_id = _get_user_id(authorization)
    db = get_supabase()

    try:
        response = (
            db.table("cases")
            .select("*")
            .eq("id", case_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Case not found")
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cases/{case_id}")
async def update_case(
    case_id: str,
    updates: dict,
    authorization: Optional[str] = Header(None),
):
    """Update case fields"""
    user_id = _get_user_id(authorization)
    db = get_supabase()

    try:
        response = (
            db.table("cases")
            .update(updates)
            .eq("id", case_id)
            .eq("user_id", user_id)
            .execute()
        )
        return response.data[0] if response.data else {"message": "Updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str,
    authorization: Optional[str] = Header(None),
):
    """Soft delete a case"""
    user_id = _get_user_id(authorization)
    db = get_supabase()

    try:
        db.table("cases").update({"status": "deleted"}).eq("id", case_id).eq("user_id", user_id).execute()
        return {"message": "Case deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
