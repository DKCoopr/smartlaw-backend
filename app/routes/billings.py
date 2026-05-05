"""Billings CRUD — invoice/fee tracking per case."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import date
from app.auth import get_current_user_id
from app.database import get_supabase

router = APIRouter(prefix="/api", tags=["billings"])


class BillingCreate(BaseModel):
    case_id: Optional[str] = None
    description: str
    amount: float
    paid_amount: float = 0
    status: str = "invoiced"      # invoiced | paid | overdue
    invoice_number: Optional[str] = None
    due_date: Optional[date] = None


@router.get("/billings")
async def list_billings(
    case_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        q = db.table("billings").select("*").eq("user_id", user_id).order("created_at", desc=True)
        if case_id:
            q = q.eq("case_id", case_id)
        response = q.execute()
        return {"billings": response.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/billings")
async def create_billing(
    payload: BillingCreate,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        data = payload.model_dump()
        data["user_id"] = user_id
        if data.get("due_date"):
            data["due_date"] = data["due_date"].isoformat()
        response = db.table("billings").insert(data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Insert failed")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/billings/{billing_id}")
async def update_billing(
    billing_id: str,
    updates: dict,
    user_id: str = Depends(get_current_user_id),
):
    allowed = {"case_id", "description", "amount", "paid_amount", "status", "invoice_number", "due_date"}
    clean = {k: v for k, v in updates.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid fields")
    db = get_supabase()
    try:
        response = db.table("billings").update(clean).eq("id", billing_id).eq("user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Billing not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/billings/{billing_id}")
async def delete_billing(
    billing_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        db.table("billings").delete().eq("id", billing_id).eq("user_id", user_id).execute()
        return {"message": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
