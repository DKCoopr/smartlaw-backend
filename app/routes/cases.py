"""
CRUD routes for cases stored in Supabase
GET    /api/cases          → list user's cases
POST   /api/cases          → create new case
GET    /api/cases/{id}     → get single case
PUT    /api/cases/{id}     → update case
DELETE /api/cases/{id}     → delete case (soft)
POST   /api/cases/cleanup  → hard-delete: ephemeral >24h + soft-deleted >30d
"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from app.models.case import CaseCreate, CaseOut, CaseListOut
from app.database import get_supabase
from app.auth import get_current_user_id

# Title prefix used by the "สรุปอย่างเดียว" upload flow — anything matching is
# treated as an ephemeral case with a 24-hour TTL. Keep in sync with the
# frontend (UploadFlowSection.handleFilesPicked).
EPHEMERAL_TITLE_PREFIX = "[สรุปชั่วคราว]"
EPHEMERAL_TTL_HOURS    = 24
SOFT_DELETED_TTL_DAYS  = 30

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
    """Soft-delete: marks status='deleted' + bumps updated_at so the cleanup
    job can compute a 30-day TTL from this exact moment instead of from
    created_at (which would unfairly purge old cases the user just soft-deleted)."""
    db = get_supabase()
    try:
        db.table("cases").update({
            "status": "deleted",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", case_id).eq("user_id", user_id).execute()
        return {"message": "Case deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cases/cleanup")
async def cleanup_old_cases(
    user_id: str = Depends(get_current_user_id),
):
    """Hard-delete two classes of soft-deleted cases for the calling user:

      1. Ephemeral "สรุปชั่วคราว" cases older than 24 hours (by created_at)
      2. Regular soft-deleted cases older than 30 days (by updated_at, which
         delete_case() bumps to now() at the moment of soft-delete)

    For each purged case we:
      a) fetch its documents and remove storage objects (best-effort)
      b) hard-delete the case row (FK on delete cascade kills documents in DB)

    Idempotent + per-user. Frontend calls this on login + before starting a
    new ephemeral upload. Returns counts so the UI can show feedback if needed.
    """
    db = get_supabase()
    now = datetime.now(timezone.utc)
    eph_cutoff  = (now - timedelta(hours=EPHEMERAL_TTL_HOURS)).isoformat()
    soft_cutoff = (now - timedelta(days=SOFT_DELETED_TTL_DAYS)).isoformat()

    summary = {"ephemeral": 0, "soft_deleted": 0, "files_removed": 0, "errors": []}

    try:
        # All soft-deleted cases for this user (we'll partition in Python — easier
        # to express the LIKE/NOT-LIKE split here than to fight PostgREST query syntax)
        rows = (
            db.table("cases")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user_id)
            .eq("status", "deleted")
            .execute()
        ).data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List soft-deleted cases failed: {e}")

    to_purge: list[str] = []
    for r in rows:
        title = (r.get("title") or "")
        is_ephemeral = title.startswith(EPHEMERAL_TITLE_PREFIX)
        if is_ephemeral:
            if (r.get("created_at") or "") < eph_cutoff:
                to_purge.append(r["id"])
                summary["ephemeral"] += 1
        else:
            # Use updated_at as proxy for "deleted at" — see delete_case()
            if (r.get("updated_at") or "") < soft_cutoff:
                to_purge.append(r["id"])
                summary["soft_deleted"] += 1

    if not to_purge:
        return summary

    # Step (a): remove storage objects for these cases' documents
    try:
        docs = (
            db.table("documents")
            .select("id, storage_path")
            .in_("case_id", to_purge)
            .eq("user_id", user_id)
            .execute()
        ).data or []
        paths = [d["storage_path"] for d in docs if d.get("storage_path")]
        if paths:
            try:
                db.storage.from_("documents").remove(paths)
                summary["files_removed"] = len(paths)
            except Exception as e:
                summary["errors"].append(f"storage remove: {str(e)[:200]}")
    except Exception as e:
        summary["errors"].append(f"list documents: {str(e)[:200]}")

    # Step (b): hard-delete the cases (FK cascade removes document rows)
    try:
        db.table("cases").delete().in_("id", to_purge).eq("user_id", user_id).execute()
    except Exception as e:
        summary["errors"].append(f"hard delete: {str(e)[:200]}")
        raise HTTPException(status_code=500, detail=f"Hard delete failed: {e}")

    return summary
