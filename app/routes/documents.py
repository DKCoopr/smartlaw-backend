"""Documents: upload to Supabase Storage + AI summary via Gemini."""
import os
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth import get_current_user_id
from app.database import get_supabase
from app.services.documents import summarize_document

router = APIRouter(prefix="/api", tags=["documents"])

# MIME → file extension. Some browsers send DOCX as octet-stream/zip — we
# fall back on the filename extension below if the MIME isn't here.
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

# Filename extension → (canonical MIME, ext for storage key). Used when the
# browser sends a generic content_type like application/octet-stream — common
# for .docx on Windows/Linux Chrome and some mobile clients.
EXT_FALLBACK = {
    ".pdf":  ("application/pdf",  "pdf"),
    ".png":  ("image/png",        "png"),
    ".jpg":  ("image/jpeg",       "jpg"),
    ".jpeg": ("image/jpeg",       "jpg"),
    ".webp": ("image/webp",       "webp"),
    ".doc":  ("application/msword", "doc"),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    ".txt":  ("text/plain",       "txt"),
}

MAX_BYTES = 50 * 1024 * 1024   # 50 MB hard limit


def _resolve_type(filename: Optional[str], content_type: Optional[str]) -> tuple[str, str]:
    """Returns (canonical_mime, file_ext) or raises 415.

    Strategy: trust MIME first; if browser sent something we don't recognize
    (octet-stream / zip / empty), fall back to the filename extension. This
    fixes DOCX uploads from Windows/Linux where browsers often emit
    application/octet-stream instead of the canonical OOXML MIME.
    """
    if content_type and content_type in ALLOWED_TYPES:
        return content_type, ALLOWED_TYPES[content_type]
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in EXT_FALLBACK:
        return EXT_FALLBACK[ext]
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type: content_type={content_type!r} ext={ext!r}",
    )


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
    folder: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    canonical_mime, file_ext = _resolve_type(file.filename, file.content_type)

    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

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
            file_options={"content-type": canonical_mime, "upsert": "false"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    # AI summary (best-effort, non-blocking on failure)
    ai_summary = await summarize_document(file_bytes, canonical_mime)

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
        "folder":        (folder or None),
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


@router.patch("/documents/{doc_id}")
async def update_document(
    doc_id: str,
    updates: dict,
    user_id: str = Depends(get_current_user_id),
):
    """Update document metadata: folder, doc_label, doc_category."""
    allowed = {"folder", "doc_label", "doc_category"}
    clean = {k: (v if v != "" else None) for k, v in updates.items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No valid fields")
    db = get_supabase()
    try:
        response = db.table("documents").update(clean).eq("id", doc_id).eq("user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/folder/rename")
async def rename_folder(
    payload: dict,
    user_id: str = Depends(get_current_user_id),
):
    """Bulk rename a folder: { case_id, old_name, new_name }."""
    case_id = payload.get("case_id")
    old_name = payload.get("old_name")
    new_name = payload.get("new_name")
    if not case_id or old_name is None:
        raise HTTPException(status_code=400, detail="case_id and old_name required")
    db = get_supabase()
    try:
        q = db.table("documents").update({"folder": (new_name or None)}).eq("user_id", user_id).eq("case_id", case_id)
        if old_name:
            q = q.eq("folder", old_name)
        else:
            q = q.is_("folder", "null")
        response = q.execute()
        return {"updated": len(response.data or [])}
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
