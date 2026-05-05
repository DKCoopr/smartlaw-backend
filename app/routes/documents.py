"""Documents: upload to Supabase Storage + AI summary via Gemini."""
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth import get_current_user_id
from app.database import get_supabase
from app.services.documents import summarize_document

router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_TYPES = {
    "application/pdf":           "pdf",
    "image/png":                 "png",
    "image/jpeg":                "jpg",
    "image/jpg":                 "jpg",
    "image/webp":                "webp",
    "application/msword":        "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain":                "txt",
}

MAX_BYTES = 50 * 1024 * 1024   # 50 MB hard limit


def _category_from_label(label: str) -> str:
    label = (label or "").lower()
    if any(k in label for k in ["สัญญา", "contract"]):       return "สัญญา"
    if any(k in label for k in ["ฟ้อง", "complaint"]):       return "คำฟ้อง"
    if any(k in label for k in ["พยาน", "witness"]):         return "พยาน"
    return "หลักฐาน"


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(None),
    doc_label: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    file_ext = ALLOWED_TYPES[file.content_type]
    original_name = file.filename or f"upload.{file_ext}"
    label = doc_label or original_name
    # Use ASCII-only storage key (Supabase Storage rejects non-ASCII).
    # Original name is preserved in the documents table.
    storage_path = f"{user_id}/{uuid.uuid4()}.{file_ext}"

    db = get_supabase()

    # Upload to Storage bucket "documents"
    try:
        db.storage.from_("documents").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": file.content_type, "upsert": "false"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    # AI summary (best-effort, non-blocking on failure)
    ai_summary = await summarize_document(file_bytes, file.content_type)

    # Insert metadata row
    payload = {
        "user_id":       user_id,
        "case_id":       case_id,
        "title":         label,
        "doc_label":     label,
        "original_name": original_name,
        "doc_category":  _category_from_label(label),
        "file_type":     file_ext,
        "file_size":     len(file_bytes),
        "storage_path":  storage_path,
        "is_processed":  bool(ai_summary),
        "ai_summary":    ai_summary or None,
    }

    try:
        response = db.table("documents").insert(payload).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Insert failed")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        # Try to clean up the orphaned upload
        try: db.storage.from_("documents").remove([storage_path])
        except Exception: pass
        raise HTTPException(status_code=500, detail=f"DB insert failed: {str(e)}")


@router.get("/documents")
async def list_documents(
    case_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        q = db.table("documents").select("*").eq("user_id", user_id).order("created_at", desc=True)
        if case_id:
            q = q.eq("case_id", case_id)
        response = q.execute()
        return {"documents": response.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}/url")
async def get_document_url(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return a signed URL valid for 1 hour to download the file."""
    db = get_supabase()
    try:
        response = (
            db.table("documents").select("storage_path").eq("id", doc_id).eq("user_id", user_id).single().execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        path = response.data.get("storage_path")
        if not path:
            raise HTTPException(status_code=404, detail="No storage path")
        signed = db.storage.from_("documents").create_signed_url(path, 3600)
        return {"url": signed.get("signedURL") or signed.get("signed_url") or signed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_supabase()
    try:
        existing = db.table("documents").select("storage_path").eq("id", doc_id).eq("user_id", user_id).single().execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Document not found")
        path = existing.data.get("storage_path")
        if path:
            try: db.storage.from_("documents").remove([path])
            except Exception: pass
        db.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
        return {"message": "Deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
