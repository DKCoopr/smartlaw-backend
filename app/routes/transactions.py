"""Transactions CRUD — financial trail for a case."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import date
from app.auth import get_current_user_id
from app.database import get_supabase

router = APIRouter(prefix="/api", tags=["transactions"])


class TransactionCreate(BaseModel):
    case_id: Optional[str] = None
    txn_date: date
    from_name: Optional[str] = ""
    from_account: Optional[str] = ""
    from_bank: Optional[str] = ""
    to_name: Optional[str] = ""
    to_account: Optional[str] = ""
    to_bank: Optional[str] = ""
    amount: float = 0
    txn_type: str = "transfer"
    description: Optional[str] = ""
    ref_no: Optional[str] = ""
    is_flagged: bool = False


@router.get("/transactions")
async def list_transactions(
    case_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        q = db.table("transactions").select("*").eq("user_id", user_id).order("txn_date", desc=False)
        if case_id:
            q = q.eq("case_id", case_id)
        response = q.execute()
        return {"transactions": response.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transactions")
async def create_transaction(
    payload: TransactionCreate,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        data = payload.model_dump()
        data["user_id"] = user_id
        data["txn_date"] = data["txn_date"].isoformat()
        response = db.table("transactions").insert(data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Insert failed")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/transactions/{txn_id}")
async def update_transaction(
    txn_id: str,
    updates: dict,
    user_id: str = Depends(get_current_user_id),
):
    allowed = {"case_id", "txn_date", "from_name", "from_account", "from_bank",
               "to_name", "to_account", "to_bank", "amount", "txn_type",
               "description", "ref_no", "is_flagged"}
    clean = {k: v for k, v in updates.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid fields")
    db = get_supabase()
    try:
        response = db.table("transactions").update(clean).eq("id", txn_id).eq("user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/transactions/{txn_id}")
async def delete_transaction(
    txn_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        db.table("transactions").delete().eq("id", txn_id).eq("user_id", user_id).execute()
        return {"message": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
