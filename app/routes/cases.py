"""
CRUD routes for cases stored in Supabase
GET    /api/cases          → list user's cases
POST   /api/cases          → create new case
GET    /api/cases/{id}     → get single case
PUT    /api/cases/{id}     → update case
DELETE /api/cases/{id}     → delete case (soft)
"""
from fastapi import APIRouter, HTTPException, Depends
from app.models.case import CaseCreate, CaseOut, CaseListOut
from app.database import get_supabase
from app.auth import get_current_user_id

router = APIRouter(prefix="/api", tags=["cases"])


@router.get("/cases", response_model=CaseListOut)
async def list_cases(
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        response = (
            db.table("cases")
            .select("*")
            .eq("user_id", user_id)
            .neq("status", "deleted")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        count_response = (
            db.table("cases")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .neq("status", "deleted")
            .execute()
        )
        return CaseListOut(
            cases=response.data or [],
            total=count_response.count or 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cases", response_model=CaseOut)
async def create_case(
    case: CaseCreate,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        payload = {
            "user_id": user_id,
            "title": case.title,
            "case_type": case.case_type,
            "court": case.court,
            "plaintiff_name": case.plaintiff_name,
            "defendant_name": case.defendant_name,
            "our_client": case.our_client,
            "claim_amount": case.claim_amount,
            "assigned_lawyer": case.assigned_lawyer,
            "next_hearing": case.next_hearing,
            "status": case.status or "active",
            "transcript": case.transcript or "",
            "form_data": case.form_data.model_dump() if case.form_data else {},
            "analysis": case.analysis if case.analysis else {},
            "ai_strength_score": case.ai_strength_score,
        }
        response = db.table("cases").insert(payload).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Insert returned no data")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cases/{case_id}", response_model=CaseOut)
async def get_case(
    case_id: str,
    user_id: str = Depends(get_current_user_id),
):
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cases/{case_id}", response_model=CaseOut)
async def update_case(
    case_id: str,
    updates: dict,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    allowed = {
        "title", "case_type", "court", "plaintiff_name", "defendant_name",
        "our_client", "claim_amount", "assigned_lawyer", "next_hearing",
        "status", "transcript", "form_data", "analysis", "ai_strength_score",
    }
    clean = {k: v for k, v in updates.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    try:
        response = (
            db.table("cases")
            .update(clean)
            .eq("id", case_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Case not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        db.table("cases").update({"status": "deleted"}).eq("id", case_id).eq("user_id", user_id).execute()
        return {"message": "Case deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
