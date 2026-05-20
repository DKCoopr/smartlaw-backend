"""Legal analysis — Claude reads PDFs natively (no Gemini middleman)."""
import asyncio
import base64
import json
import re
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncIterator
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings
from app.database import get_supabase
from app.services.documents import ocr_image, _extract_docx_text, _extract_doc_text, DOCX_MIME, DOC_MIME
from app.services.brave_search import search_thai_legal, format_for_prompt
from app.services.law_retrieval import get_law_context_for_prompt

router = APIRouter(prefix="/api/legal", tags=["legal"])

# ── In-memory job store for async analysis ────────────────────────────────────
# Each entry: {status: "pending"|"complete"|"failed", started_at, finished_at?,
#              user_id, result?, error?}
# Jobs older than 1 hour are auto-evicted on access.
_jobs: Dict[str, Dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600


def _evict_stale_jobs():
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [k for k, j in _jobs.items() if j.get("started_at", 0) < cutoff]
    for k in stale:
        _jobs.pop(k, None)

settings = get_settings()
_claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-7"

MAX_PAGES       = 150            # หน้ารวมสูงสุด: PDF นับหน้าจริง, รูป/ไฟล์อื่น = 1 หน้า
MAX_DOC_FILES   = 100            # safety cap กัน loop ยาวเกิน
MAX_DOC_BYTES   = 32 * 1024 * 1024   # 32 MB รวมทุกไฟล์


# ── Page counting helpers ─────────────────────────────────────────────────────

def _count_pdf_pages(file_bytes: bytes) -> int:
    """นับหน้า PDF จริงด้วย pypdf — fallback 1 หน้าถ้าอ่านไม่ได้"""
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return max(1, len(reader.pages))
    except Exception:
        return 1


def _count_docx_pages(file_bytes: bytes) -> int:
    """
    ประมาณหน้า DOCX จาก word count
    ~250 คำต่อหน้า (เอกสารกฎหมายไทยมีความหนาแน่นปานกลาง)
    """
    try:
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        words = sum(len(p.text.split()) for p in doc.paragraphs if p.text.strip())
        return max(1, round(words / 250))
    except Exception:
        return 1


def _count_pages(file_bytes: bytes, mime: str) -> int:
    """คืนจำนวนหน้าของไฟล์ตาม MIME type"""
    if mime == "application/pdf":
        return _count_pdf_pages(file_bytes)
    if mime == DOCX_MIME:
        return _count_docx_pages(file_bytes)
    # DOC (legacy), images, text → นับเป็น 1 หน้าเสมอ
    return 1


class CaseInput(BaseModel):
    title: Optional[str] = ""
    case_type: Optional[str] = ""
    court: Optional[str] = ""
    plaintiff_name: Optional[str] = ""
    defendant_name: Optional[str] = ""
    our_client: Optional[str] = "plaintiff"
    claim_amount: Optional[float] = 0
    transcript: Optional[str] = ""
    notes: Optional[str] = ""
    documents_summary: Optional[str] = ""   # legacy field — ignored when case_id supplied


class AnalyzeIn(BaseModel):
    case: CaseInput
    perspective: str = "both"
    document_types: List[str] = []
    case_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None    # if set: only analyze these docs (subset)
    language: str = "th"                   # th | en | zh — response language


def _case_block(case: CaseInput) -> str:
    plaintiff = case.plaintiff_name or "(ยังไม่ระบุ)"
    defendant = case.defendant_name or "(ยังไม่ระบุ)"
    our = "ฝ่ายโจทก์" if case.our_client == "plaintiff" else ("ฝ่ายจำเลย" if case.our_client == "defendant" else "ทั้งสองฝ่าย")
    block = (
        f"ชื่อคดี: {case.title or '—'}\n"
        f"ประเภทคดี: {case.case_type or '—'}\n"
        f"ศาล: {case.court or '—'}\n"
        f"โจทก์/ผู้ฟ้อง: {plaintiff}\n"
        f"จำเลย/ผู้ถูกฟ้อง: {defendant}\n"
        f"ลูกความที่เรารับ: {our}\n"
        f"ทุนทรัพย์: {case.claim_amount or 0:,.0f} บาท"
    )
    if case.transcript or case.notes:
        block += f"\n\n== คำบอกเล่า / รายละเอียดเพิ่มเติมจากผู้ใช้ ==\n{case.transcript or case.notes}"
    return block


def _perspective_label(p: str) -> str:
    return {"plaintiff": "ฝ่ายโจทก์", "defendant": "ฝ่ายจำเลย"}.get(p, "ทั้งสองฝ่าย")


def _perspective_directive(perspective: str, plaintiff_name: str = "", defendant_name: str = "") -> str:
    """Render an explicit instruction block that tells Claude how to think for
    this perspective. The 14-section template stays the same, but the *emphasis*,
    *recommendations*, and *scenario assessment* must shift based on which side
    we're representing. Without this block the model defaults to balanced
    analysis even when the user asked for one-sided strategy."""
    p = (plaintiff_name or "โจทก์").strip()
    d = (defendant_name or "จำเลย").strip()

    if perspective == "plaintiff":
        return f"""🎯 **โหมดการวิเคราะห์: เป็นทนาย{p} (ฝ่ายโจทก์)**

คุณคือทนายฝ่ายโจทก์ที่กำลังเตรียมคดีเพื่อ**ฟ้องและชนะคดี**ให้ {p}
ทุกหัวข้อในรายงานต้องเขียนจากมุมมองของทนายโจทก์ — เน้น **กลยุทธ์เชิงรุก** เพื่อนำชัยชนะ

🔑 **กรอบความคิดหลักที่ต้องครอบคลุมทุกหัวข้อ:**

1. **เตรียมก่อนยื่นฟ้อง — อุดทุกรอยรั่วก่อนถึงมือศาล**
   - คำฟ้องต้องครบองค์ประกอบความผิด/ครบมูลคดี — ระบุมาตรา + ข้อเท็จจริง + คำขอท้ายฟ้องอย่างละเอียด
   - อายุความ / อำนาจฟ้อง / เขตอำนาจ — ตรวจให้ชัดว่าฟ้องในเวลาที่กฎหมายกำหนด มีอำนาจฟ้อง และยื่นในศาลที่ถูกต้อง
   - ความสามารถในการดำเนินคดี (capacity to sue) ของ {p}
   - หลักฐาน 5 ระดับ: เอกสารต้นฉบับ / สำเนารับรอง / พยานบุคคล / พยานผู้เชี่ยวชาญ / พยานวัตถุ — เก็บให้ครบก่อนยื่นฟ้อง
   - **ป้องกัน {d} ขอให้ยกฟ้องในชั้นต้น** — คาดการณ์ทุกข้อต่อสู้เชิงรูปแบบที่ {d} จะใช้ (motion to dismiss, demurrer, จำเลยร้องขอให้รวมหรือแยกประเด็น) แล้วเตรียมคำตอบไว้ล่วงหน้า

2. **วิเคราะห์การโต้กลับของฝ่าย {d} แล้วอุดรอยรั่ว**
   ในหัวข้อจุดอ่อน/ความเสี่ยง — **เน้นคิดเหมือน {d}** : "ถ้าผมเป็นทนายฝ่าย {d} ผมจะใช้ช่องไหนยกฟ้อง / สู้ลดทุนทรัพย์ / กลับเป็นโต้แย้ง?"
   จากนั้นเขียน "วิธีอุดรอยรั่ว" แต่ละข้อ — ทำอะไรล่วงหน้าเพื่อปิดช่องนั้น

3. **ลำดับการนำสืบ — เริ่มแรงเพื่อ pin จำเลย**
   - พยานปากแรกควรเป็นพยานที่ทำให้ศาลเห็นภาพรวมและความเสียหาย
   - หลักฐานเอกสารที่ทรงพลังที่สุดต้องเข้าก่อน เพื่อให้ {d} ต้องตั้งรับตลอดทาง
   - แนวซักค้านพยานฝ่าย {d} — ระบุปากๆ พร้อมคำถามที่จะใช้ทำลายความน่าเชื่อถือ

4. **คำขอท้ายฟ้อง — เรียกให้ครบ ห้ามขาดทุน**
   ทุนทรัพย์หลัก / ดอกเบี้ย / ค่าเสียหายต่อเนื่อง / ค่าทนาย / ค่าฤชา — คำนวณให้ชัด
   ระบุ "คำขอประธาน" และ "คำขอรอง" เผื่อศาลให้ไม่ครบ

5. **เจรจา/ประนอม — จากจุดที่แข็งกว่า**
   ระบุราคาเป้าหมายที่ {p} ควรยอม และเงื่อนไขที่ห้ามยอมเด็ดขาด

⚠️ **กฎพิเศษสำหรับโหมดโจทก์:**
- หัวข้อ 7 (จุดแข็ง) เขียนของฝ่าย {p}
- หัวข้อ 8 (จุดอ่อน/ความเสี่ยง) เขียนของฝ่าย {p} — แต่เน้น "{d} จะโจมตีจุดไหน + เราจะอุดยังไง"
- หัวข้อ 9 (กลยุทธ์) เขียนแบบทนายโจทก์เชิงรุก
- หัวข้อ 14.5 (คำแนะนำขั้นสุดท้าย) — แนะ {p} โดยตรงว่าควรทำอะไร เมื่อไร เพื่อชนะ
"""

    if perspective == "defendant":
        return f"""🛡️ **โหมดการวิเคราะห์: เป็นทนาย{d} (ฝ่ายจำเลย)**

คุณคือทนายฝ่ายจำเลยที่กำลังเตรียมแก้ต่างให้ {d} เพื่อ**ยกฟ้อง / ชนะคดี / ลดความเสียหายให้น้อยที่สุด**
ทุกหัวข้อในรายงานต้องเขียนจากมุมมองของทนายจำเลย — เน้น **กลยุทธ์เชิงรับ + โต้กลับ**

🔑 **กรอบความคิดหลักที่ต้องครอบคลุมทุกหัวข้อ:**

1. **โจมตีคำฟ้องเชิงรูปแบบ — ขอให้ยกฟ้องโดยไม่ต้องเข้าเนื้อหา**
   ตรวจช่องโหว่เชิงรูปแบบของคำฟ้อง {p} ทุกข้อ:
   - **อายุความ** — ฟ้องเลยกำหนดหรือไม่ + เริ่มนับเมื่อใด ตามมาตราใด
   - **อำนาจฟ้อง / Standing** — {p} เป็นผู้เสียหายโดยตรงไหม มีนิติสัมพันธ์ไหม
   - **เขตอำนาจศาล** — ฟ้องผิดศาลหรือไม่ ขอให้โอนคดีหรือยกฟ้อง
   - **ความครบขององค์ประกอบความผิด/มูลคดี** — คำฟ้องระบุข้อเท็จจริงครบไหม
   - **โมฆะ/โมฆียะของนิติกรรม** ที่ {p} อ้าง
   - **เอกสารต้นฉบับ/หลักฐานน่าเชื่อถือ** — ขอให้ {p} ส่งต้นฉบับ ถ้าส่งไม่ได้ → คัดค้าน

2. **ข้อต่อสู้เชิงเนื้อหา (Substantive Defenses)**
   - ปฏิเสธข้อกล่าวหาทีละข้อ พร้อมเหตุผลและพยานหลักฐาน
   - ความผิด/ความรับผิดของ {p} เอง — ผู้เสียหายมีส่วนผิดหรือไม่ (Contributory negligence / pari delicto)
   - การชำระหนี้/การปลดหนี้/การหักกลบลบหนี้ที่เคยเกิดขึ้น
   - เหตุสุดวิสัย / เหตุพ้นวิสัย / ความยินยอม / การสละสิทธิ์
   - ลดทุนทรัพย์ — แม้แพ้ ก็ต้องทำให้ค่าเสียหายน้อยที่สุด

3. **ทำลายพยานหลักฐาน {p}**
   - ตรวจเอกสารทุกชิ้น — มีข้อพิรุธ การปลอมแปลง การแก้ไขไหม
   - แนวซักค้านพยาน {p} ทุกปาก — ทำลายความน่าเชื่อถือเรื่องตัวเอง / เรื่องเบิกความ / ความขัดแย้งภายใน
   - ขอให้ศาลเรียกเอกสารเพิ่มที่ {p} ไม่อยากเอามาแสดง

4. **ฟ้องแย้ง (Counterclaim) ถ้าเป็นไปได้**
   ถ้า {d} มีสิทธิเรียกร้องกลับ — เสนอให้ฟ้องแย้งในคดีเดียวกัน เพื่อสร้างแรงต่อรอง

5. **เจรจา/ประนอม — จากจุดที่อึดที่สุด**
   - ใช้จุดอ่อนคำฟ้อง {p} กดดันให้ลด/ถอนฟ้อง
   - เงื่อนไขที่ {d} ควรยอมรับ vs ห้ามยอม
   - ราคาที่ {d} ควรจ่ายในกรณีเลวร้ายที่สุด

⚠️ **กฎพิเศษสำหรับโหมดจำเลย:**
- หัวข้อ 7 (จุดแข็ง) เขียนของฝ่าย {d}
- หัวข้อ 8 (จุดอ่อน/ความเสี่ยง) เขียนของฝ่าย {d}
- หัวข้อ 9 (กลยุทธ์) เขียนแบบทนายจำเลยเชิงรับ + โต้กลับ — เริ่มจาก motion ให้ยกฟ้องก่อน ถ้าไม่ได้ค่อยสู้เนื้อหา
- หัวข้อ 14.5 (คำแนะนำขั้นสุดท้าย) — แนะ {d} โดยตรงว่าควรทำอะไร เมื่อไร เพื่อยกฟ้องหรือลดความเสียหาย
"""

    # Both
    return f"""⚖️ **โหมดการวิเคราะห์: ทั้งสองฝ่าย (มุมมองเป็นกลาง)**

คุณคือทนายอาวุโสที่ประเมินคดีอย่างเป็นกลาง — ลูกความอาจเป็น{p}หรือ{d}ก็ได้
ทุกหัวข้อต้องเขียน **ทั้งสองมุมมอง** — กลยุทธ์ของ{p} และกลยุทธ์ของ{d} อย่างเท่าเทียม

🔑 **กรอบความคิดหลักที่ต้องครอบคลุมทุกหัวข้อ:**

1. **แยกหัวข้อเป็น 2 ช่องเสมอ** — ฝ่าย {p} กับ ฝ่าย {d}
2. **ในหัวข้อจุดแข็ง/จุดอ่อน** — เขียนทั้งของ {p} และของ {d} (รวมกันแล้ว ≥ 10 ข้อ)
3. **ในหัวข้อกลยุทธ์ (หัวข้อ 9)** — แตกเป็น
   - 9.A กลยุทธ์ฝ่ายโจทก์ ({p}) — เชิงรุก ฟ้องอย่างไรให้ชนะ
   - 9.B กลยุทธ์ฝ่ายจำเลย ({d}) — เชิงรับ ยกฟ้องหรือแก้ต่างอย่างไรให้รอด
   - 9.C จุดปะทะหลัก — ที่ทั้งสองฝ่ายต้องระวัง
4. **ในตารางหัวข้อ 14.2** — เปรียบเทียบทุกปัจจัยทั้งฝ่ายโจทก์/จำเลย
5. **ในหัวข้อ 14.5** — เขียนคำแนะนำให้**ทั้งสองฝ่าย**: "ถ้าคุณคือ {p} → ทำ X" / "ถ้าคุณคือ {d} → ทำ Y"

⚠️ **กฎพิเศษสำหรับโหมดทั้งสองฝ่าย:**
- ห้ามเข้าข้างฝ่ายใดฝ่ายหนึ่ง — ต้องเป็นกลางอย่างแท้จริง
- โอกาสชนะ X% / Y% ต้องสะท้อนความเป็นจริงตามหลักฐาน ไม่ใช่ 50/50 แบบหลีกเลี่ยง
- ทุกการแนะนำต้องครบทั้งสองฝ่าย
"""


def _content_type(file_type: str) -> str:
    return {
        "pdf":  "application/pdf",
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "txt":  "text/plain",
        "docx": DOCX_MIME,
        "doc":  "application/msword",   # legacy .doc — _extract_docx_text won't parse it; will skip
    }.get((file_type or "").lower(), "application/pdf")


async def _load_case_attachments(case_id: str, user_id: str, doc_ids: Optional[List[str]] = None) -> tuple[list[dict], dict]:
    """Pull case docs from Supabase Storage. Returns (attachments, debug_info).

    Routing rules:
    - PDF / TXT  → straight to Claude (Claude reads natively)
    - Images     → run Gemini OCR first → send OCR'd text + image to Claude

    If doc_ids is provided, only load those specific documents (subset selection).
    """
    debug = {"case_id": case_id, "user_id": user_id, "rows_found": 0, "loaded": [], "errors": [], "skipped": [], "subset": bool(doc_ids)}
    if not case_id:
        debug["errors"].append("no case_id provided")
        return [], debug

    db = get_supabase()
    try:
        q = (
            db.table("documents")
            .select("*")
            .eq("case_id", case_id)
            .eq("user_id", user_id)
            .order("created_at", desc=False)
        )
        if doc_ids:
            q = q.in_("id", doc_ids)
        rows = q.execute()
    except Exception as e:
        debug["errors"].append(f"db query failed: {e}")
        print(f"[legal/analyze] DB query failed for case_id={case_id}: {e}")
        return [], debug

    rows_data = rows.data or []
    debug["rows_found"] = len(rows_data)
    if doc_ids:
        print(f"[legal/analyze] case_id={case_id} subset={len(doc_ids)} → {len(rows_data)} document rows")
    else:
        print(f"[legal/analyze] case_id={case_id} → {len(rows_data)} document rows")

    out = []
    total_bytes = 0
    total_pages = 0
    for d in rows_data[:MAX_DOC_FILES]:
        path = d.get("storage_path")
        name = d.get("original_name") or d.get("doc_label") or "document"
        mime = _content_type(d.get("file_type"))
        if not path:
            debug["errors"].append(f"{name}: no storage_path")
            continue
        try:
            file_bytes = db.storage.from_("documents").download(path)
        except Exception as e:
            debug["errors"].append(f"{name}: download failed — {e}")
            print(f"[legal/analyze] download failed for {path}: {e}")
            continue
        if not file_bytes:
            debug["errors"].append(f"{name}: empty file")
            continue

        # ── Page-based limit ──────────────────────────────────────────────────
        file_pages = _count_pages(file_bytes, mime)
        if total_pages + file_pages > MAX_PAGES:
            debug["skipped"].append({
                "name":   name,
                "reason": "page_limit",
                "pages":  file_pages,
                "detail": f"ไฟล์นี้มี {file_pages} หน้า — เกินโควต้า {MAX_PAGES} หน้า (ใช้แล้ว {total_pages} หน้า)",
            })
            print(f"[legal/analyze] page limit: skip {name} ({file_pages}p, used={total_pages}/{MAX_PAGES})")
            continue   # skip ไฟล์นี้ แต่ยังดูไฟล์ถัดไปต่อ (ไม่ break)

        # ── Size limit ────────────────────────────────────────────────────────
        if total_bytes + len(file_bytes) > MAX_DOC_BYTES:
            debug["skipped"].append({
                "name":   name,
                "reason": "size_limit",
                "pages":  file_pages,
                "detail": f"ขนาดไฟล์รวมเกิน 32 MB",
            })
            print(f"[legal/analyze] size limit: skip {name}")
            continue

        attachment = {
            "name": name,
            "mime": mime,
            "bytes": file_bytes,
            "ocr_text": None,
        }

        # DOCX / DOC → Claude can't ingest Word formats; extract text first
        # and feed as text/plain. DOCX uses python-docx (preserves tables);
        # legacy .doc uses antiword/catdoc via subprocess (text only).
        if mime in (DOCX_MIME, DOC_MIME):
            try:
                text = _extract_docx_text(file_bytes) if mime == DOCX_MIME else _extract_doc_text(file_bytes)
                fmt  = "DOCX" if mime == DOCX_MIME else "DOC"
                if text:
                    attachment["mime"]  = "text/plain"
                    attachment["bytes"] = text.encode("utf-8")
                    print(f"[legal/analyze] {fmt}→text {name}: {len(text)} chars")
                else:
                    debug["errors"].append(f"{name}: {fmt} text extract returned empty")
                    continue   # skip — nothing useful to send
            except Exception as e:
                debug["errors"].append(f"{name}: Word extract failed — {e}")
                continue

        # Image → run OCR first to capture handwriting/text reliably
        if mime.startswith("image/"):
            try:
                ocr = await ocr_image(file_bytes, mime)
                if ocr:
                    attachment["ocr_text"] = ocr
                    print(f"[legal/analyze] OCR'd {name}: {len(ocr)} chars")
                else:
                    print(f"[legal/analyze] OCR returned empty for {name}")
            except Exception as e:
                debug["errors"].append(f"{name}: OCR failed — {e}")
                print(f"[legal/analyze] OCR failed for {name}: {e}")

        out.append(attachment)
        total_bytes += len(file_bytes)
        total_pages += file_pages
        debug["loaded"].append({
            "name":      name,
            "bytes":     len(file_bytes),
            "mime":      mime,
            "pages":     file_pages,
            "ocr_chars": len(attachment["ocr_text"]) if attachment["ocr_text"] else 0,
        })
        print(f"[legal/analyze] attached: {name} ({len(file_bytes)} bytes, {file_pages}p, {mime})")

    debug["total_bytes"] = total_bytes
    debug["total_pages"] = total_pages
    if debug["skipped"]:
        print(f"[legal/analyze] {len(debug['skipped'])} file(s) skipped: {[s['name'] for s in debug['skipped']]}")
    return out, debug


def _complexity_score(case: CaseInput, attachments: list[dict]) -> int:
    """
    คะแนนความซับซ้อน 0-17+
    ─────────────────────────────────────────────────────────
    0-5   → Sonnet (คดีง่าย / เตรียมข้อมูล)
    6-10  → Opus   (คดีซับซ้อน / มีเอกสารมาก)
    11+   → Opus + Extended Thinking (คดียาก / ทุนทรัพย์สูงมาก)
    ─────────────────────────────────────────────────────────
    """
    s = 0
    total_text = (case.transcript or "") + (case.notes or "")
    total_bytes = sum(len(a.get("bytes", b"")) for a in attachments)

    # ── เนื้อหา / บันทึก ──────────────────────────────────────────────────────
    s += min(2, len(total_text) // 2000)            # บันทึกยาว > 2000 ตัวอักษร

    # ── เอกสารแนบ ─────────────────────────────────────────────────────────────
    s += min(3, len(attachments))                   # จำนวนเอกสาร (max +3)
    if total_bytes >= 10 * 1024 * 1024: s += 2      # ไฟล์รวม > 10 MB
    elif total_bytes >= 5 * 1024 * 1024:  s += 1    # ไฟล์รวม > 5 MB

    # ── ทุนทรัพย์ ─────────────────────────────────────────────────────────────
    amt = case.claim_amount or 0
    if   amt >= 100_000_000: s += 5   # > 100 ล้าน → สูงมาก
    elif amt >=  10_000_000: s += 3   # > 10 ล้าน
    elif amt >=   1_000_000: s += 2   # > 1 ล้าน
    elif amt >=     500_000: s += 1   # > 500,000

    # ── ประเภทคดี ─────────────────────────────────────────────────────────────
    HIGH_COMPLEX = {"ปกครอง", "ภาษี", "ล้มละลาย", "ฟื้นฟูกิจการ",
                    "ทรัพย์สินทางปัญญา", "อนุญาโตตุลาการ"}
    MED_COMPLEX  = {"อาญา", "ที่ดิน", "มรดก", "หุ้นส่วน", "บริษัท",
                    "แรงงาน", "ครอบครัว"}
    ct = case.case_type or ""
    if ct in HIGH_COMPLEX:  s += 3
    elif ct in MED_COMPLEX: s += 2

    # ── สัญญาณอื่น ────────────────────────────────────────────────────────────
    if case.plaintiff_name and case.defendant_name: s += 1  # ระบุคู่ความครบ
    if "ฎีกา" in total_text or "อุทธรณ์" in total_text: s += 1
    if re.search(r"มาตรา\s*\d", total_text): s += 1         # อ้างมาตรากฎหมายแล้ว

    return s


def _pick_model(case: CaseInput, attachments: list[dict]) -> tuple[str, bool]:
    """
    คืนค่า (model_name, use_extended_thinking)
    ────────────────────────────────────────────
    score  0-5  → Sonnet,     no thinking
    score  6-10 → Opus,       no thinking
    score 11+   → Opus,       extended thinking ON
    """
    score = _complexity_score(case, attachments)
    if score >= 11:
        return OPUS, True     # คดียาก — คิดลึกก่อนตอบ
    elif score >= 6:
        return OPUS, False    # คดีซับซ้อน — Opus ปกติ
    else:
        return SONNET, False  # คดีง่าย — Sonnet เร็วกว่า ถูกกว่า


def _resize_image_for_claude(image_bytes: bytes, mime: str, max_dim: int = 1800) -> tuple[bytes, str]:
    """Resize an image to fit Claude's multi-image API limit.

    Anthropic returns 400 invalid_request_error when ANY image in a many-image
    request exceeds 2000px on either axis. We resize anything above ~1800px
    (with a safety buffer) down, preserving aspect ratio. Returns (new_bytes,
    new_mime) — mime stays as input if Pillow recognises the format, else
    falls back to JPEG.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        # Quick exit: already within limits → don't recompress (preserves OCR quality)
        if img.width <= max_dim and img.height <= max_dim:
            return image_bytes, mime
        # Resize with aspect-ratio preserved
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        # Re-encode. JPEG keeps file size small for photos; PNG/WEBP keeps for graphics.
        buf = io.BytesIO()
        if mime == "image/png":
            # Preserve transparency
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), "image/png"
        if mime == "image/webp":
            img.save(buf, format="WEBP", quality=85, method=4)
            return buf.getvalue(), "image/webp"
        # JPEG (default) — also handles HEIC/etc. after Pillow conversion
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[image-resize] failed ({type(e).__name__}: {e}) — sending original")
        return image_bytes, mime


def _build_content_blocks(prompt: str, attachments: list[dict]) -> list:
    blocks = []
    # Count images so we know whether to enforce Claude's 2000px many-image cap.
    image_count = sum(1 for a in attachments if a.get("mime", "").startswith("image/"))
    needs_resize = image_count > 1   # Anthropic's "many images" cap kicks in immediately for >1

    for att in attachments:
        if att["mime"] == "application/pdf":
            b64 = base64.standard_b64encode(att["bytes"]).decode("ascii")
            blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
                "title": att["name"][:80],
            })
        elif att["mime"].startswith("image/"):
            # If we OCR'd the image first, prepend the verbatim text as a primary
            # source — Claude can then cross-check with the image itself.
            ocr_text = att.get("ocr_text") or ""
            if ocr_text:
                blocks.append({
                    "type": "text",
                    "text": (
                        f"=== ข้อความที่ OCR ได้จากภาพ \"{att['name']}\" "
                        f"(ดิบ verbatim — ใช้เป็นแหล่งข้อมูลหลัก) ===\n"
                        f"{ocr_text}\n"
                        f"=== จบข้อความ OCR ==="
                    ),
                })
            # Resize down to ≤1800px when there are many images (safety buffer
            # under the 2000px Anthropic cap). Single-image requests bypass the
            # resize so we keep maximum OCR-quality detail.
            img_bytes = att["bytes"]
            img_mime = att["mime"]
            if needs_resize:
                orig_size = len(img_bytes)
                img_bytes, img_mime = _resize_image_for_claude(img_bytes, img_mime, max_dim=1800)
                new_size = len(img_bytes)
                if new_size != orig_size:
                    print(f"[image-resize] {att['name']}: {orig_size:,}B → {new_size:,}B ({img_mime})")
            b64 = base64.standard_b64encode(img_bytes).decode("ascii")
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img_mime, "data": b64},
            })
        elif att["mime"] == "text/plain":
            try:
                text_content = att["bytes"].decode("utf-8", errors="replace")[:50_000]
                blocks.append({"type": "text", "text": f"=== เอกสาร: {att['name']} ===\n{text_content}\n=== จบเอกสาร ==="})
            except Exception:
                pass
    blocks.append({"type": "text", "text": prompt})
    return blocks


async def _claude_call(
    prompt: str,
    model: str,
    attachments: list[dict],
    max_tokens: int = 64000,
    extended_thinking: bool = False,
) -> str:
    """
    เรียก Claude วิเคราะห์คดี
    ─────────────────────────────────────────────────────────
    extended_thinking=True → เปิด Extended Thinking (Opus only)
      - Claude คิดลึก ~8,000 tokens ก่อนตอบ (ไม่นับใน max_tokens)
      - เหมาะกับคดียาก ทุนทรัพย์สูง หรืออ้างมาตรากฎหมายซับซ้อน
      - ช้าขึ้น ~2-4 นาที แต่ความแม่นยำสูงขึ้นมาก
    ─────────────────────────────────────────────────────────
    """
    content = _build_content_blocks(prompt, attachments)

    # Extended Thinking ไม่รองรับ streaming + temperature ต้องเป็น 1
    if extended_thinking and model == OPUS:
        print(f"[legal/analyze] 🧠 Extended Thinking ON — model={model} budget=8000 tokens")
        try:
            resp = await _claude.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=1,          # required for extended thinking
                thinking={
                    "type":         "enabled",
                    "budget_tokens": 8000,  # Claude คิดก่อนตอบ 8K tokens
                },
                messages=[{"role": "user", "content": content}],
            )
            # content อาจมี thinking blocks — เอาเฉพาะ text blocks
            return "".join(
                block.text for block in resp.content
                if hasattr(block, "text") and block.type == "text"
            ).strip()
        except Exception as e:
            print(f"[legal/analyze] Extended Thinking failed ({e}), falling back to normal Opus")
            # fallthrough to streaming below

    # Normal streaming call (Sonnet หรือ Opus ปกติ)
    async def _stream(mt: int) -> str:
        chunks: list[str] = []
        async with _claude.messages.stream(
            model=model,
            max_tokens=mt,
            temperature=0.3,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for text in stream.text_stream:
                chunks.append(text)
        return "".join(chunks).strip()

    try:
        return await _stream(max_tokens)
    except Exception:
        # Fallback if the model rejects max_tokens (e.g., Opus at 64K) —
        # retry at 32K which all current Claude 4.x models accept.
        return await _stream(min(max_tokens, 32000))


# ── Prompt ────────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_FRAME = """คุณคือทนายความไทยอาวุโสระดับเนติบัณฑิตไทย มีประสบการณ์ว่าความและที่ปรึกษากฎหมายของสำนักงานชั้นนำมากว่า 25 ปี
เชี่ยวชาญทุกแขนง — แพ่ง / อาญา / แรงงาน / ปกครอง / ภาษี / ครอบครัว / มรดก / IP / สัญญาธุรกิจ

ภารกิจ: เขียน "บทสรุปคดีและแนวทางต่อสู้ขั้นสุด" ในระดับเดียวกับ memo ภายในสำนักงานทนายความระดับ Tier-1
— ละเอียด ลึก ใช้งานได้จริงในศาล (ภาษาผลลัพธ์กำหนดโดย LANGUAGE INSTRUCTION ด้านล่าง)

🚨 **กฎเหล็ก — ห้ามฝ่าฝืน:**
1. **ต้องเขียนให้จบทุกหัวข้อ 1-14** — ห้ามตัดกลางคัน ห้ามจบกลางหัวข้อ ถ้ายาวต้องบีบเนื้อหาให้พอ แต่ห้ามขาดหัวข้อ
2. **หัวข้อ 9 (แนวต่อสู้) และ 14 (บทสรุป+โอกาสแพ้ชนะ) ต้องละเอียดที่สุด** — เป็นหัวใจของรายงาน
3. **อ่านเอกสารแนบก่อนตอบ** — ดึงข้อมูลทุกส่วน (วันที่ ตัวเลข ชื่อบุคคล/นิติบุคคล มาตรากฎหมาย คำพิพากษา เลขบัญชี)
4. **ใช้ชื่อจริงของทุกคนที่ปรากฏในเอกสาร** — ห้ามใช้ "โจทก์/จำเลย" ลอยๆ ต้องเขียนชื่อจริง
5. **อ้างอิงเลขมาตราเต็ม + ชื่อกฎหมาย + เลขฎีกา** ทุกครั้ง
6. **ภาษาผลลัพธ์: ดูคำสั่ง LANGUAGE INSTRUCTION ด้านล่าง** — ต้องใช้ภาษานั้นตลอดทั้งเอกสาร ห้ามสลับภาษา
7. **ความยาวรวม ≥ 5,000 คำ** — เขียนละเอียดเหมือน memo จริง
8. **ใช้ markdown** — `**bold**` / bullet `•` / `1.` `2.` `3.` / heading `## หัวข้อ`
"""


LANG_INSTRUCTION = {
    "th": "",   # Thai is the default — no extra instruction needed
    "en": "\n\n🌐 **LANGUAGE INSTRUCTION (HIGHEST PRIORITY — overrides all other rules): Write your ENTIRE response in English. Every section heading, analysis paragraph, recommendation, and legal term must be in English. Do NOT output any Thai text whatsoever.**\n",
    "zh": "\n\n🌐 **语言要求（最高优先级——覆盖所有其他规则）：请用中文（普通话）撰写全部回答。每个章节标题、分析段落、建议及法律术语均须使用中文。禁止输出任何泰语文字。**\n",
    "ja": "\n\n🌐 **言語指示（最優先 — 他のすべての規則を上書き）：回答全体を日本語で記述してください。すべてのセクション見出し、分析段落、推奨事項、法律用語は日本語で記述する必要があります。タイ語の出力は一切禁止します。**\n",
    "ko": "\n\n🌐 **언어 지침 (최우선 — 다른 모든 규칙을 무효화): 전체 응답을 한국어로 작성하십시오. 모든 섹션 제목, 분석 단락, 권장 사항 및 법률 용어는 한국어로 작성해야 합니다. 태국어는 출력하지 마십시오.**\n",
    "ru": "\n\n🌐 **ЯЗЫКОВАЯ ИНСТРУКЦИЯ (ВЫСШИЙ ПРИОРИТЕТ — заменяет все другие правила): Напишите ВЕСЬ ответ на русском языке. Каждый заголовок раздела, абзац анализа, рекомендация и юридический термин должны быть на русском языке. НЕ выводите тайский текст ни в коем случае.**\n",
    "fr": "\n\n🌐 **INSTRUCTION DE LANGUE (PRIORITÉ MAXIMALE — remplace toutes les autres règles) : Rédigez TOUTE votre réponse en français. Chaque titre de section, paragraphe d'analyse, recommandation et terme juridique doit être en français. NE produisez AUCUN texte en thaï.**\n",
    "ar": "\n\n🌐 **تعليمات اللغة (الأولوية القصوى — تتجاوز جميع القواعد الأخرى): اكتب الرد بالكامل باللغة العربية. يجب أن يكون كل عنوان قسم وفقرة تحليل وتوصية ومصطلح قانوني باللغة العربية. لا تُخرج أي نص باللغة التايلاندية.**\n",
}

# Final reminder appended at the END of the prompt — strongest position for
# instruction-following. Without this, the Thai template section headers
# ("# 📋 บทสรุปคดี" etc.) right before the model generates would override
# the language directive at the top of the prompt.
LANG_FINAL_REMINDER = {
    "en": """
═══════════════════════════════════════════════════════════════
🌐 **FINAL LANGUAGE OVERRIDE — READ BEFORE GENERATING**
═══════════════════════════════════════════════════════════════
Despite the Thai template above (section headers, instructions, examples), your **ENTIRE response MUST be in English**.

- Translate every Thai section heading into English. Examples:
  • "# 📋 บทสรุปคดี" → "# 📋 Case Summary"
  • "## 1. บทสรุปโดยย่อ (Executive Summary)" → "## 1. Executive Summary"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 Plaintiff"
  • "### 3.1 ตารางธุรกรรมการเงิน" → "### 3.1 Financial Transaction Table"
- Write all analysis, narrative, recommendations, table headers, and bullet points in English
- **DO NOT output any Thai characters** (Thai script ก-ฮ, ๐-๙) in your response — except for proper-noun names that have no English equivalent (e.g. court names, statute citations); keep those in Thai but translate every label around them
- The Thai content above is **structural guidance only** — translate it as you generate
═══════════════════════════════════════════════════════════════
""",
    "zh": """
═══════════════════════════════════════════════════════════════
🌐 **最终语言覆盖指令 — 生成前必读**
═══════════════════════════════════════════════════════════════
尽管上方模板使用泰语（章节标题、说明、示例），您的**全部回答必须使用简体中文**。

- 将每个泰语章节标题翻译成中文。示例：
  • "# 📋 บทสรุปคดี" → "# 📋 案件摘要"
  • "## 1. บทสรุปโดยย่อ (Executive Summary)" → "## 1. 执行摘要"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 原告"
  • "### 3.1 ตารางธุรกรรมการเงิน" → "### 3.1 金融交易明细表"
- 全部分析、叙述、建议、表格标题、项目符号均使用中文撰写
- **回答中不得出现任何泰语字符**（ก-ฮ、๐-๙）——专有名词（法院名称、法条引用）若无中文对应可保留泰语原文，但其周围的标签必须翻译为中文
- 上方泰语内容**仅作结构指引**——请边生成边翻译
═══════════════════════════════════════════════════════════════
""",
    "ja": """
═══════════════════════════════════════════════════════════════
🌐 **最終言語オーバーライド — 生成前に必読**
═══════════════════════════════════════════════════════════════
上記テンプレートはタイ語（章タイトル、指示、例）を使用していますが、**回答全体は日本語で記述する必要があります**。

- 各タイ語の章タイトルを日本語に翻訳してください。例：
  • "# 📋 บทสรุปคดี" → "# 📋 事案要旨"
  • "## 1. บทสรุปโดยย่อ" → "## 1. エグゼクティブサマリー"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 原告"
- すべての分析、推奨事項、表の見出しを日本語で記述してください
- **タイ文字（ก-ฮ、๐-๙）は出力しないこと** — 固有名詞（裁判所名、条文引用）に日本語訳がない場合のみタイ語原文を保持できますが、周辺のラベルは必ず日本語に翻訳してください
═══════════════════════════════════════════════════════════════
""",
    "ko": """
═══════════════════════════════════════════════════════════════
🌐 **최종 언어 오버라이드 — 생성 전 필독**
═══════════════════════════════════════════════════════════════
위의 템플릿이 태국어(섹션 제목, 지침, 예시)를 사용하지만, **전체 응답은 한국어로 작성해야 합니다**.

- 각 태국어 섹션 제목을 한국어로 번역하십시오. 예:
  • "# 📋 บทสรุปคดี" → "# 📋 사건 요지"
  • "## 1. บทสรุปโดยย่อ" → "## 1. 요약"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 원고"
- 모든 분석, 권장 사항, 표 제목을 한국어로 작성하십시오
- **태국 문자 (ก-ฮ, ๐-๙) 출력 금지** — 고유명사(법원명, 법조문 인용)에 한국어 번역이 없는 경우에만 태국어 원문 유지 가능하나 주변 라벨은 반드시 한국어로 번역
═══════════════════════════════════════════════════════════════
""",
    "ru": """
═══════════════════════════════════════════════════════════════
🌐 **ОКОНЧАТЕЛЬНОЕ ЯЗЫКОВОЕ ПЕРЕОПРЕДЕЛЕНИЕ — ПРОЧИТАЙТЕ ПЕРЕД ГЕНЕРАЦИЕЙ**
═══════════════════════════════════════════════════════════════
Несмотря на то, что приведённый выше шаблон использует тайский язык (заголовки разделов, инструкции, примеры), ваш **полный ответ должен быть написан на русском языке**.

- Переведите каждый тайский заголовок раздела на русский. Примеры:
  • "# 📋 บทสรุปคดี" → "# 📋 Резюме дела"
  • "## 1. บทสรุปโดยย่อ" → "## 1. Краткое резюме"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 Истец"
- Напишите весь анализ, рекомендации, заголовки таблиц на русском языке
- **НЕ выводите тайские символы** (ก-ฮ, ๐-๙) — для имён собственных (названия судов, ссылки на законы) можно сохранить тайский оригинал только при отсутствии русского эквивалента, но окружающие метки должны быть переведены на русский
═══════════════════════════════════════════════════════════════
""",
    "fr": """
═══════════════════════════════════════════════════════════════
🌐 **REMPLACEMENT LINGUISTIQUE FINAL — À LIRE AVANT DE GÉNÉRER**
═══════════════════════════════════════════════════════════════
Bien que le modèle ci-dessus utilise le thaï (titres de section, instructions, exemples), votre **réponse entière doit être rédigée en français**.

- Traduisez chaque titre de section thaï en français. Exemples :
  • "# 📋 บทสรุปคดี" → "# 📋 Résumé de l'affaire"
  • "## 1. บทสรุปโดยย่อ" → "## 1. Résumé exécutif"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 Demandeur"
- Rédigez toute l'analyse, les recommandations et les en-têtes de tableau en français
- **N'utilisez AUCUN caractère thaï** (ก-ฮ, ๐-๙) — pour les noms propres (noms de tribunaux, citations légales), vous pouvez conserver l'original thaï uniquement s'il n'existe pas d'équivalent français, mais les étiquettes environnantes doivent être traduites en français
═══════════════════════════════════════════════════════════════
""",
    "ar": """
═══════════════════════════════════════════════════════════════
🌐 **التجاوز اللغوي النهائي — اقرأ قبل التوليد**
═══════════════════════════════════════════════════════════════
على الرغم من أن النموذج أعلاه يستخدم اللغة التايلاندية (عناوين الأقسام، التعليمات، الأمثلة)، يجب كتابة **ردك بالكامل باللغة العربية**.

- ترجم كل عنوان قسم تايلاندي إلى العربية. أمثلة:
  • "# 📋 บทสรุปคดี" → "# 📋 ملخص القضية"
  • "## 1. บทสรุปโดยย่อ" → "## 1. الملخص التنفيذي"
  • "### 2.1 ผู้ฟ้อง / โจทก์" → "### 2.1 المدعي"
- اكتب كل التحليل والتوصيات وعناوين الجداول باللغة العربية
- **لا تخرج أي حروف تايلاندية** (ก-ฮ، ๐-๙) — بالنسبة لأسماء الأعلام (أسماء المحاكم، الاستشهادات القانونية)، يمكنك الاحتفاظ بالأصل التايلاندي فقط في حالة عدم وجود مكافئ عربي، لكن يجب ترجمة التسميات المحيطة إلى العربية
═══════════════════════════════════════════════════════════════
""",
}

def _analysis_prompt(case: CaseInput, perspective: str, has_attachments: bool, language: str = "th") -> str:
    p_label = _perspective_label(perspective)
    directive = _perspective_directive(perspective, case.plaintiff_name or "", case.defendant_name or "")
    lang_note = LANG_INSTRUCTION.get(language, "")
    lang_tail = LANG_FINAL_REMINDER.get(language, "")

    if not has_attachments:
        body = f"""{lang_note}{ANALYSIS_SYSTEM_FRAME}

⚠️ **ไม่มีเอกสารแนบในคดีนี้** — วิเคราะห์ตามข้อมูลที่ผู้ใช้กรอกเท่านั้น

ตอนต้นคำตอบ ให้แจ้งผู้ใช้ว่า: "ℹ️ คดีนี้ยังไม่มีเอกสารแนบ — การวิเคราะห์อิงข้อมูลที่กรอกเท่านั้น คุณภาพจะดีขึ้นมากถ้าอัปโหลดเอกสารคดีจริง (PDF คำฟ้อง / สัญญา / คำพิพากษา)"

== ข้อมูลคดี ==
{_case_block(case)}

== มุมมอง ==
{p_label}

{directive}

== โครงสร้างคำตอบ ==
ตอบครบทุกหัวข้อในเทมเพลตด้านล่าง — ใช้ชื่อจริงของ {case.plaintiff_name or '(ระบุโจทก์)'} และ {case.defendant_name or '(ระบุจำเลย)'} ตลอด
ทุกหัวข้อต้องสะท้อน **โหมดการวิเคราะห์ที่กำกับไว้ด้านบน**

🚨 **หัวข้อ Timeline ของคดี (บังคับมี — ห้ามข้าม)**
ในหัวข้อ "3. ข้อเท็จจริงโดยละเอียด" ให้สร้างตาราง markdown ของลำดับเหตุการณ์อย่างน้อย 6 เหตุการณ์
แม้ไม่มีเอกสารแนบ — ให้สร้างจากข้อมูลที่ user กรอก + ความรู้กฎหมายไทยทั่วไป (พร้อมระบุว่า "ประมาณการ — ต้องยืนยันกับเอกสารจริง" ในคอลัมน์หมายเหตุ)

| ลำดับ | วันที่ | เหตุการณ์ | ฝ่ายที่เกี่ยวข้อง | ความสำคัญ |
| --- | --- | --- | --- | --- |
| ๑ | dd/mm/yyyy หรือ "ประมาณ ม.ค. 2566" | สรุปสั้น | โจทก์/จำเลย | ⭐ critical / supporting |
| ๒ | ... | ... | ... | ... |

หลังตาราง: เขียน 2-3 บรรทัดสรุปว่า timeline เผยจุดสำคัญอะไรบ้าง

(ใช้โครงสร้าง 14 หัวข้อตามด้านล่าง)
"""
        return body + lang_tail

    if has_attachments:
        body = f"""{lang_note}{ANALYSIS_SYSTEM_FRAME}

🚨 **กฎสำคัญที่สุด:** เอกสารที่แนบมาด้านบนคือ **แหล่งความจริงเดียว** — ดึงทุกอย่างจากเอกสาร: ชื่อคู่ความ ประเภทคดี ศาล มาตราฎหมาย เลขฎีกา จำนวนเงิน วันที่ ทุกอย่าง

ฟิลด์ที่ผู้ใช้กรอกด้านล่างเป็นเพียง "เบาะแส" หรือ "ค่าเริ่มต้น" — **ห้ามเชื่อ** ถ้าขัดแย้งกับเอกสารแนบ ให้ใช้เอกสารเป็นหลัก
ถ้าผู้ใช้ใส่ "ประเภทคดี: แพ่ง" แต่เอกสารบอกเป็นคดีอาญา/ปกครอง/ฎีกา → ใช้ตามเอกสาร

== ข้อมูลที่ผู้ใช้กรอก (อ่านเป็นเบาะแสเท่านั้น) ==
{_case_block(case)}

== มุมมองที่ต้องวิเคราะห์ ==
{p_label}

{directive}

== โครงสร้างคำตอบที่ต้องการ ==
ตอบทุกหัวข้อข้างล่าง — ห้ามขาดหัวข้อใด ห้ามตอบสั้น
ดึงชื่อจริง วันที่จริง ตัวเลขจริง จากเอกสารแนบ ห้ามใช้ placeholder ห้ามใช้ตัวอย่างทั่วไป

# 📋 บทสรุปคดี
(ขึ้นต้นด้วยหัวข้อชื่อคดีตามที่ปรากฏในเอกสารจริง — ไม่ใช่ผู้ใช้กรอก)

## 1. บทสรุปโดยย่อ (Executive Summary)
สรุปคดีให้ลูกความเข้าใจในนาทีเดียว 6-10 บรรทัด — ใครฟ้องใคร เรื่องอะไร ทุนทรัพย์ คาดการณ์ผล กลยุทธ์หัวใจ

## 2. ข้อมูลคู่ความและรายละเอียดคดี
### 2.1 ผู้ฟ้อง / โจทก์
**(ระบุชื่อจริงตามเอกสาร)**
- สถานะทางกฎหมาย / ที่อยู่ / ผู้แทน / ทนายโจทก์ (เท่าที่ปรากฏในเอกสาร)
- คำขอท้ายฟ้อง — เรียกร้องอะไรบ้าง พร้อมจำนวนเงินครบทุกข้อ

### 2.2 ผู้ถูกฟ้อง / จำเลย
**(ระบุชื่อจริงตามเอกสาร)**
- สถานะ / ที่อยู่ / ผู้แทน / ทนายจำเลย (เท่าที่ปรากฏ)
- ท่าทีของจำเลยตามเอกสาร

### 2.3 ศาลและเขตอำนาจ
- ระบุศาลและคดีหมายเลขดำ/แดง (ถ้ามี)
- เหตุผลที่อยู่ในเขตอำนาจ + ประเด็นเขตอำนาจที่อาจโต้แย้ง

### 2.4 ทุนทรัพย์
- จำนวนรวม + โครงสร้าง (ค่าเสียหายจริง / ดอกเบี้ย / ค่าทนาย)

## 3. ข้อเท็จจริงโดยละเอียด (Timeline of Facts)

🚨 **บังคับมีในทุกคดี — ห้ามข้าม** : สร้างตาราง Timeline ด้วย markdown ดังนี้
**ขั้นต่ำ 8 เหตุการณ์** เรียงจากเก่าไปใหม่ — ดึงวันที่จริงจากเอกสารแนบ ถ้าไม่ระบุวันชัดให้ใช้ "ประมาณ {เดือน/ปี}" หรือ "วันที่ไม่ระบุ — ก่อน/หลังเหตุการณ์ X"

| ลำดับ | วันที่ | เหตุการณ์ | ฝ่ายที่เกี่ยวข้อง | ที่มา / แหล่งอ้างอิง | ความสำคัญทางคดี |
| --- | --- | --- | --- | --- | --- |
| ๑ | dd/mm/yyyy | สรุปเหตุการณ์เป็นประโยคเดียวชัดเจน | โจทก์ / จำเลย / พยาน / บุคคลที่สาม | คำฟ้องหน้า X / สัญญาข้อ Y / ใบเสร็จเลข Z | ⭐ critical / supporting / context |
| ๒ | ... | ... | ... | ... | ... |

**กฎเข้มงวด:**
- **ห้ามแต่งวันที่** — ถ้าไม่มีในเอกสาร ใส่ "ไม่ระบุ" หรือ "ประมาณ ม.ค. ๒๕๖๖"
- **ใช้เลขไทย** สำหรับลำดับ (๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘)
- **คอลัมน์ "ความสำคัญ":** ⭐ = critical (เป็นมูลคดีหลัก), supporting = สนับสนุน, context = ภูมิหลัง
- **เหตุการณ์สุดท้าย** ควรเป็นวันยื่นฟ้อง/วันที่ปัจจุบัน
- ถ้าคดีเกี่ยวกับการเงิน → ตาราง 3.1 ด้านล่างคือธุรกรรม **คนละชุดกับ timeline นี้**

หลังตาราง เขียน **คำอธิบายสั้น** (3-5 บรรทัด) สรุปจุดสำคัญที่ timeline เผยให้เห็น — เช่น "มีการเงียบหายของจำเลย 6 เดือนหลังการทวงถามครั้งแรก" หรือ "การโอนเกิดหลังประกาศกฎหมายใหม่"

### 3.1 ตารางธุรกรรมการเงิน / กระแสเงินสด (ถ้าคดีเกี่ยวกับการเงิน)
ถ้าเอกสารปรากฏการโอนเงิน รับเงิน จ่ายเงิน เลขบัญชี หรือธุรกรรมการเงินใดๆ — **ต้องสร้างตารางสรุปทุกธุรกรรมที่ปรากฏในเอกสาร** ในรูปแบบ markdown table ดังนี้:

| ลำดับ | วันที่ | จากบัญชี / ผู้โอน | ถึงบัญชี / ผู้รับ | จำนวนเงิน (บาท) | ประเภท | หมายเหตุ |
| --- | --- | --- | --- | --- | --- | --- |
| ๑ | dd/mm/yyyy | ชื่อ + เลขบัญชี + ธนาคาร | ชื่อ + เลขบัญชี + ธนาคาร | ๐๐๐,๐๐๐ | โอนค้ำประกัน / ชำระค่า ... | ... |
| ๒ | ... | ... | ... | ... | ... | ... |

**กฎ:**
- ใช้ข้อมูลจากเอกสารจริงเท่านั้น ห้ามแต่งเติม
- ถ้ามีการโอนเงินผ่านบัญชีม้าหลายทอด → ทำตารางแยกแต่ละทอด พร้อมหัวข้อย่อย "ทอดที่ ๑", "ทอดที่ ๒"
- รวมยอดเงินรวมท้ายตาราง
- ถ้าไม่มีธุรกรรมการเงินในเอกสาร → ข้ามหัวข้อ 3.1 นี้ได้

🚨 **กรณีธุรกรรมเกิน 5 รายการ — ต้องใช้รูปแบบตารางสรุปอย่างละเอียดและสวยงาม:**
1. **เรียงลำดับตามวันที่** จากเก่าไปใหม่ — ห้ามสลับ
2. **ใส่เลขลำดับเป็นตัวเลขไทย** (๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙ ๑๐) ทุกแถว
3. **จัดหมวดธุรกรรมเป็น Block ย่อย** (subsection) เช่น:
   - **3.1.A ธุรกรรมโอนต้นทาง** (จากผู้เสียหายส่งออก)
   - **3.1.B ธุรกรรมผ่านบัญชีม้าทอดที่ ๑**
   - **3.1.C ธุรกรรมผ่านบัญชีม้าทอดที่ ๒**
   - **3.1.D ธุรกรรมปลายทาง / ถอนเป็นเงินสด / แปลงเป็นสินทรัพย์**
   แต่ละ block มีตารางของตัวเอง + ยอดรวม block
4. **แถวรวมยอดท้ายตาราง** ต้องเขียน `**รวม:**` พร้อมจำนวนเงินตัวหนา
5. **เพิ่มคอลัมน์ "ยอดสะสม (Running Total)"** — ช่วยศาล/ทนายเห็นภาพการไหลของเงิน
6. **ใต้ตารางใหญ่** เพิ่ม "📊 บทวิเคราะห์เส้นทางการเงิน" 4-6 บรรทัด — เน้นจุดผิดปกติ จุดน่าสงสัย จุดที่จะใช้เป็นพยานหลัก
7. **ตารางผู้เกี่ยวข้องท้าย 3.1**:
   | ชื่อบัญชี | ธนาคาร | บทบาท | รวมรับ (บาท) | รวมจ่าย (บาท) | ยอดสุทธิ |
   | --- | --- | --- | --- | --- | --- |

## 4. ประเด็นข้อพิพาทหลัก (Issues)
ระบุ 4-6 ประเด็นที่ศาลต้องตัดสิน

## 5. กฎหมายที่เกี่ยวข้อง (Applicable Laws)
≥ 6-10 มาตรา/พรบ. พร้อม:
- เลขมาตราเต็ม + ชื่อกฎหมาย
- ใจความสำคัญ
- การประยุกต์ใช้กับข้อเท็จจริงคดีนี้

### 5.1 คำพิพากษาฎีกาที่เกี่ยวข้อง
≥ 3-5 ฎีกา ระบุเลข/ปี + สรุปประเด็น + ความเกี่ยวข้อง

## 6. ประเมินโอกาสชนะคดี
- ฝ่ายโจทก์: X% — เหตุผล
- ฝ่ายจำเลย: Y% — เหตุผล

## 7. จุดแข็ง{"ของทั้งสองฝ่าย (แยกฝ่ายโจทก์/ฝ่ายจำเลย)" if perspective == "both" else f"ของฝ่าย{p_label}"}
{("≥ 5 ข้อต่อฝ่าย — แยกหัวข้อย่อย 7.1 ฝ่ายโจทก์ และ 7.2 ฝ่ายจำเลย") if perspective == "both" else "≥ 5 ข้อ พร้อมเหตุผลสนับสนุน"}

## 8. จุดอ่อน/ความเสี่ยง{"ของทั้งสองฝ่าย (แยกฝ่ายโจทก์/ฝ่ายจำเลย)" if perspective == "both" else f"ของฝ่าย{p_label}"}
{("≥ 4 ข้อต่อฝ่าย — แยก 8.1 ฝ่ายโจทก์ และ 8.2 ฝ่ายจำเลย พร้อมวิธีบรรเทา") if perspective == "both" else f"≥ 4 ข้อ พร้อมวิธีบรรเทาแต่ละจุด — เน้น **คิดเหมือนคู่ความฝ่ายตรงข้าม** ว่าจะใช้ช่องไหนโจมตี แล้วเขียนวิธีอุดทุกช่อง"}

## 9. แนวทางต่อสู้คดีในชั้นศาล (Ultimate Court Strategy) — หัวใจของรายงาน
{("**หัวข้อนี้ต้องแยกเป็น 9.A กลยุทธ์ฝ่ายโจทก์ / 9.B กลยุทธ์ฝ่ายจำเลย / 9.C จุดปะทะหลัก** — แต่ละส่วนใช้โครงสร้าง 9.1-9.4 ด้านล่างของตัวเอง") if perspective == "both" else (f"**เขียนแบบทนายฝ่าย{p_label}เชิงรุก** — ทุก subsection (9.1-9.4) ต้องเขียนจากมุม{p_label}เป็นหลัก" if perspective == "plaintiff" else f"**เขียนแบบทนายฝ่าย{p_label}เชิงรับ + โต้กลับ** — เริ่มจากข้อต่อสู้เชิงรูปแบบ (motion ขอให้ยกฟ้อง / อายุความ / อำนาจฟ้อง / เขตอำนาจ) ใน 9.1-9.2 ก่อน ถ้ายกไม่ได้ค่อยสู้เนื้อหา")}

### 9.1 การเตรียมตัวก่อนยื่นฟ้อง / ก่อนขึ้นศาล
**หลักฐานเอกสารที่ต้องเตรียม** (จัดกลุ่มเป็นหมวดหมู่ พร้อมระบุว่าหาจากที่ไหน):
- หมวดที่ ๑: หลักฐานเกี่ยวกับ ... (รายการละเอียดอย่างน้อย 5-8 รายการ)
- หมวดที่ ๒: หลักฐานทางการเงิน ...
- หมวดที่ ๓: ...

**พยานบุคคลที่ต้องเชิญ** (ระบุชื่อ/บทบาท + ประเด็นที่จะถามแต่ละคน อย่างน้อย 4-6 พยาน):
- พยานปากที่ ๑ — ชื่อ/บทบาท: ... ประเด็นที่จะถาม: ...
- พยานปากที่ ๒ — ...

**เอกสารที่ต้องร่าง** (รายการละเอียด):
- คำฟ้อง / คำให้การ / คำร้อง / คำแถลง — ระบุเนื้อหาหลักของแต่ละฉบับ

**การจัดเตรียมพยานหลักฐานทางอิเล็กทรอนิกส์** (ถ้ามี):
- ภาพหน้าจอ / ข้อความแชต / สลิปโอนเงิน — วิธีรับรองความถูกต้อง

### 9.2 ขั้นตอนในชั้นพิจารณาคดี (Trial Tactics)
**ลำดับการนำสืบพยานโจทก์/จำเลย:**
- พยานปากแรกที่ควรเบิกความ — เหตุผล
- ลำดับการนำสืบที่ดีที่สุด — โครงสร้างเรื่องราว

**ประเด็นข้อสู้คดีหลัก (Legal Arguments)** ≥ 6 ประเด็น พร้อมเหตุผลและฐานทางกฎหมาย:
1. **ข้อสู้ที่ ๑:** (ระบุประเด็น) — เหตุผล + มาตรา/ฎีกาที่อ้างอิง
2. **ข้อสู้ที่ ๒:** ...
(ต่อไปจนครบ 6 ข้อ)

**แนวซักค้านพยานฝ่ายตรงข้าม (Cross-examination playbook):**
- พยานปากที่ ๑ ฝ่ายตรงข้าม — แนวคำถามซักค้านที่จะใช้ทำลายความน่าเชื่อถือ
- พยานปากที่ ๒ — ...

**การยกข้อต่อสู้ทางกฎหมาย:**
- อายุความ — อายุความฟ้องคดีนี้คือ ... ปี เริ่มนับเมื่อ ...
- อำนาจฟ้อง — โจทก์มีอำนาจฟ้องหรือไม่ เพราะอะไร
- ความชอบด้วยกฎหมาย — ฟ้องครบองค์ประกอบไหม
- นิติกรรมโมฆะ/โมฆียะ (ถ้าเกี่ยวข้อง)

**กลยุทธ์การจัดการพยานเอกสารฝ่ายตรงข้าม:**
- การโต้แย้งความถูกต้อง / ความน่าเชื่อถือ
- การขอให้ศาลมีคำสั่งให้ส่งเอกสารต้นฉบับ

### 9.3 ทางออกทางเลือก (Alternative Resolutions)
**ประนีประนอม / ยอมความ:**
- เงื่อนไขที่ลูกความควรยอมรับ
- เงื่อนไขที่ห้ามยอมเด็ดขาด
- ราคาเป้าหมายในการเจรจา

**ไกล่เกลี่ย:**
- ขั้นตอนและเวลา
- กลยุทธ์ในการประชุมไกล่เกลี่ย

**ถอนฟ้อง / ถอนคำให้การ:**
- เมื่อใดที่คุ้มค่า
- ผลทางกฎหมายของแต่ละทางเลือก

### 9.4 การเตรียมแผนสำรอง (Plan B & Plan C)
- ถ้าเกิดเหตุการณ์ A — แผนสำรองคือ ...
- ถ้าพยานหลักไม่มาเบิกความ — ...
- ถ้าศาลมีคำสั่งไม่ให้นำสืบ — ...

## 10. Timeline เชิงปฏิบัติ (Action Timeline)
| ช่วงเวลา | งานที่ต้องทำ |
| --- | --- |
| ทันที (0-7 วัน) | ... |
| ระยะสั้น (1-4 สัปดาห์) | ... |
| ระยะกลาง (1-3 เดือน) | ... |
| ระยะยาว (จนคำพิพากษา + บังคับคดี) | ... |

## 11. ประมาณการค่าใช้จ่าย
- ค่าฤชาธรรมเนียมศาล (อ้างอัตราจริง)
- ค่าทนาย: ช่วงต่ำ – สูง พร้อมเหตุผล
- ค่าใช้จ่ายอื่น (ผู้เชี่ยวชาญ / เดินทาง / เอกสาร)
- ระยะเวลารวมโดยประมาณ

## 12. ความเสี่ยงรอบด้าน (Comprehensive Risk Assessment)
- เสี่ยงทางกฎหมาย (อายุความ/อำนาจฟ้อง/โมฆะ)
- เสี่ยงทางการเงิน
- เสี่ยงด้านเวลา
- เสี่ยงทางชื่อเสียง / ผลกระทบทางธุรกิจ
- เสี่ยงทางอาญา/วินัย

## 13. คำแนะนำทางยุทธวิธี (Tactical Tips)
≥ 8 ข้อ จากประสบการณ์ทนายอาวุโส — เคล็ดลับชี้ขาดคดี

## 14. บทสรุปและการประเมินโอกาสแพ้ชนะขั้นสุด (Final Conclusion & Win-Loss Assessment)

### 14.1 สรุปประเด็นสำคัญทั้งคดี
สรุปจุดสำคัญทุกประเด็นจากหัวข้อ 1-13 ในย่อหน้าเดียว 8-12 บรรทัด

### 14.2 ประเมินโอกาสแพ้ชนะอย่างละเอียด
**เปรียบเทียบในตาราง:**

| ปัจจัย | ผลต่อโจทก์ | ผลต่อจำเลย |
| --- | --- | --- |
| น้ำหนักหลักฐานเอกสาร | + / – | + / – |
| ความน่าเชื่อถือพยานบุคคล | ... | ... |
| ฐานกฎหมายและฎีกา | ... | ... |
| พฤติการณ์แห่งคดี | ... | ... |
| ปัจจัยเฉพาะคดี | ... | ... |

**ตัวเลขสรุปรวม:**
- **โอกาสชนะของฝ่ายโจทก์: X%**
- **โอกาสชนะของฝ่ายจำเลย: Y%**
- เหตุผลประกอบ 4-6 บรรทัด — ทำไมประเมินเช่นนี้ ปัจจัยตัดสินสำคัญที่สุดคืออะไร

### 14.3 สถานการณ์ที่เป็นไปได้ (Scenario Analysis)
- **สถานการณ์ดีที่สุด (Best case):** ผลคำพิพากษาที่หวังได้
- **สถานการณ์ที่น่าจะเป็นไปได้มากที่สุด (Most likely):** ผลที่คาดว่าจะเกิด
- **สถานการณ์เลวร้ายที่สุด (Worst case):** ผลที่ต้องเตรียมรับมือ + แผนสำรอง

### 14.4 Action Items 7 วันแรก
1. ...
2. ...
3. ...
(ต่อจนครบ 7-10 รายการที่ทนายต้องทำทันที)

### 14.5 คำแนะนำขั้นสุดท้ายต่อลูกความ
สรุป 1 ย่อหน้าจากมุมมองทนายอาวุโสที่ให้คำแนะนำตรงๆ — ลูกความควรเลือกแนวทางใด เพราะอะไร

---

⚠️ **คำเตือน:** เอกสารนี้เป็นการประเมินจากระบบขั้นสูง ทนายผู้รับคดีต้องตรวจสอบและปรับใช้ตามรูปคดีจริง

🔚 **— จบรายงาน — กรุณาตรวจสอบว่าเขียนครบทุกหัวข้อ 1 ถึง 14 ก่อนหยุด —**
"""
        return body + lang_tail

    # Fallback (shouldn't reach here, but keep return type stable)
    return ""


async def _do_analyze(payload: "AnalyzeIn", user_id: str) -> dict:
    """Run the actual analysis. Pulled out so the job runner can call it."""
    # 1. Pull actual case PDFs from Supabase storage (skip Gemini summary)
    attachments, debug = ([], {"case_id": None})
    if payload.case_id:
        attachments, debug = await _load_case_attachments(payload.case_id, user_id, payload.doc_ids)

    print(f"[legal/analyze] case_id={payload.case_id} attachments={len(attachments)} bytes={sum(len(a['bytes']) for a in attachments)}")

    # 2. Thai Legal DB search — run in parallel with nothing yet (fast, ~2-3s)
    #    Build doc summaries string from attachment names for better queries
    doc_summary_str = " ".join(a["name"] for a in attachments)
    legal_search_results = []
    brave_key = settings.brave_search_api_key
    if brave_key:
        try:
            legal_search_results = await search_thai_legal(
                api_key=brave_key,
                case_type=payload.case.case_type or "",
                plaintiff=payload.case.plaintiff_name or "",
                defendant=payload.case.defendant_name or "",
                claim_amount=payload.case.claim_amount or 0,
                transcript=payload.case.transcript or "",
                notes=payload.case.notes or "",
                doc_summaries=doc_summary_str,
            )
            print(f"[legal/search] found {len(legal_search_results)} results from Thai legal DB")
        except Exception as e:
            print(f"[legal/search] search failed (non-fatal): {e}")
    else:
        print("[legal/search] BRAVE_SEARCH_API_KEY not set — skipping legal DB search")

    # 3. Pick model (Sonnet / Opus / Opus+ExtendedThinking ตามความซับซ้อน)
    chosen_model, use_thinking = _pick_model(payload.case, attachments)
    complexity = _complexity_score(payload.case, attachments)
    print(f"[legal/analyze] complexity={complexity} model={chosen_model} thinking={use_thinking}")

    # 3.5. RAG: pgvector retrieval for Thai immigration law (no-op for non-immigration cases)
    case_lang_b = getattr(payload, "language", "th")
    rag_query_b = " ".join(filter(None, [
        payload.case.title or "",
        payload.case.case_type or "",
        payload.case.notes or "",
        payload.case.transcript or "",
    ]))
    law_context_b = ""
    try:
        law_context_b = await get_law_context_for_prompt(rag_query_b, lang=case_lang_b)
        if law_context_b:
            print(f"[legal/analyze] RAG: injected {len(law_context_b)} chars of immigration-law context")
    except Exception as e:
        print(f"[legal/analyze] RAG retrieval failed (non-fatal): {e}")

    # 4. Build prompt — order: pgvector law → Brave search results → analysis template
    base_prompt = _analysis_prompt(payload.case, payload.perspective, has_attachments=bool(attachments), language=case_lang_b)
    legal_db_block = format_for_prompt(legal_search_results)
    prompt_parts_b = [p for p in [law_context_b, legal_db_block, base_prompt] if p]
    prompt = "\n\n".join(prompt_parts_b)

    # 5. Run Claude with PDFs as document blocks + enriched prompt
    analysis_text = await _claude_call(
        prompt, chosen_model, attachments,
        max_tokens=64000,
        extended_thinking=use_thinking,
    )

    # 6. Document drafts (in parallel, also with attachments for context)
    documents: dict[str, str] = {}
    if payload.document_types:
        results = await asyncio.gather(
            *[_draft_document(t, payload.case, chosen_model, attachments) for t in payload.document_types],
            return_exceptions=True,
        )
        for t, res in zip(payload.document_types, results):
            documents[t] = f"⚠️ ไม่สามารถร่างเอกสารได้: {str(res)}" if isinstance(res, Exception) else res

    return {
        "analysis": analysis_text,
        "model_used": chosen_model,
        "complexity_score": complexity,
        "extended_thinking_used": use_thinking,
        "attachments_count": len(attachments),
        "legal_search_results": legal_search_results,   # ส่งกลับ frontend ด้วย
        "debug": debug,
        "documents": documents,
    }


async def _run_job(job_id: str, payload: "AnalyzeIn", user_id: str):
    """Background worker — writes back into _jobs[job_id]."""
    try:
        result = await _do_analyze(payload, user_id)
        _jobs[job_id] = {
            **_jobs.get(job_id, {}),
            "status": "complete",
            "finished_at": time.time(),
            "result": result,
        }
    except Exception as e:
        _jobs[job_id] = {
            **_jobs.get(job_id, {}),
            "status": "failed",
            "finished_at": time.time(),
            "error": str(e),
        }


def _sse(event: dict) -> str:
    """Serialize a dict as a single SSE data line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _stream_analysis(payload: "AnalyzeIn", user_id: str) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted strings for the stream endpoint."""

    # ── 1. Load documents ────────────────────────────────────────────────────
    yield _sse({"type": "status", "message": "กำลังเชื่อมต่อและโหลดเอกสาร..."})
    attachments, debug = [], {"case_id": None}
    if payload.case_id:
        attachments, debug = await _load_case_attachments(payload.case_id, user_id, payload.doc_ids)
        n = len(attachments)
        pg = debug.get("total_pages", 0)
        mb = round(debug.get("total_bytes", 0) / 1024 / 1024, 2)
        yield _sse({"type": "status", "message": f"โหลดเอกสารแล้ว {n} ฉบับ · {pg} หน้า · {mb} MB"})

    # ── 2. Brave Search (parallel-ish — start before waiting on model pick) ─
    yield _sse({"type": "status", "message": "กำลังค้นหาฎีกาและกฎหมายที่เกี่ยวข้อง..."})
    legal_search_results = []
    doc_summary_str = " ".join(a["name"] for a in attachments)
    brave_key = settings.brave_search_api_key
    if brave_key:
        try:
            legal_search_results = await search_thai_legal(
                api_key=brave_key,
                case_type=payload.case.case_type or "",
                plaintiff=payload.case.plaintiff_name or "",
                defendant=payload.case.defendant_name or "",
                claim_amount=payload.case.claim_amount or 0,
                transcript=payload.case.transcript or "",
                notes=payload.case.notes or "",
                doc_summaries=doc_summary_str,
            )
            yield _sse({"type": "legal_search", "results": legal_search_results})
            print(f"[legal/stream] found {len(legal_search_results)} results from Thai legal DB")
        except Exception as e:
            print(f"[legal/stream] search failed (non-fatal): {e}")

    # ── 3. Pick model + send ETA ─────────────────────────────────────────────
    chosen_model, use_thinking = _pick_model(payload.case, attachments)
    complexity = _complexity_score(payload.case, attachments)

    # ETA estimate (seconds): base by tier + 10s per attachment
    if use_thinking:
        eta_sec = 480 + len(attachments) * 15   # ~8 min base
        status_msg = "กำลังวิเคราะห์เชิงลึก..."
    elif chosen_model == OPUS:
        eta_sec = 240 + len(attachments) * 12   # ~4 min base
        status_msg = "กำลังวิเคราะห์คดี..."
    else:
        eta_sec = 90 + len(attachments) * 8     # ~1.5 min base
        status_msg = "กำลังวิเคราะห์คดี..."

    yield _sse({"type": "status", "message": status_msg})
    yield _sse({"type": "eta", "seconds": eta_sec})
    print(f"[legal/stream] complexity={complexity} model={chosen_model} thinking={use_thinking} eta={eta_sec}s")

    # ── 3.5. RAG: pgvector retrieval for Thai immigration law ────────────────
    # No-op (returns "") when the case doesn't contain immigration keywords —
    # see app/services/law_retrieval.py:IMMIGRATION_KEYWORDS. When matched,
    # embeds the query and pulls top-6 law chunks from Supabase law_chunks.
    case_lang = getattr(payload, "language", "th")
    rag_query = " ".join(filter(None, [
        payload.case.title or "",
        payload.case.case_type or "",
        payload.case.notes or "",
        payload.case.transcript or "",
    ]))
    law_context = ""
    try:
        law_context = await get_law_context_for_prompt(rag_query, lang=case_lang)
        if law_context:
            yield _sse({"type": "status", "message": "📚 ดึงกฎหมาย ตม. ที่เกี่ยวข้องจากฐานข้อมูล..."})
            print(f"[legal/stream] RAG: injected {len(law_context)} chars of immigration-law context")
    except Exception as e:
        print(f"[legal/stream] RAG retrieval failed (non-fatal): {e}")

    # ── 4. Build prompt ──────────────────────────────────────────────────────
    base_prompt = _analysis_prompt(payload.case, payload.perspective, has_attachments=bool(attachments), language=case_lang)
    legal_db_block = format_for_prompt(legal_search_results)
    # Order: pgvector law (most authoritative) → Brave search → analysis template.
    # Claude reads top-down so the more concrete, retrieved law goes first.
    prompt_parts = [p for p in [law_context, legal_db_block, base_prompt] if p]
    prompt = "\n\n".join(prompt_parts)
    content = _build_content_blocks(prompt, attachments)

    # ── 5. Stream Claude ──────────────────────────────────────────────────────
    analysis_text = ""
    stop_reason: str = "unknown"   # end_turn | max_tokens | stop_sequence | error | unknown
    error_message: str = ""

    if use_thinking and chosen_model == OPUS:
        # Extended Thinking doesn't support streaming — wait, then chunk-send
        yield _sse({"type": "status", "message": "กำลังวิเคราะห์เชิงลึก อาจใช้เวลาสักครู่..."})
        try:
            resp = await _claude.messages.create(
                model=chosen_model,
                max_tokens=64000,
                temperature=1,
                thinking={"type": "enabled", "budget_tokens": 8000},
                messages=[{"role": "user", "content": content}],
            )
            analysis_text = "".join(
                block.text for block in resp.content
                if hasattr(block, "text") and block.type == "text"
            ).strip()
            stop_reason = getattr(resp, "stop_reason", "unknown") or "unknown"
            # Fake-stream in 150-char chunks so the UI updates progressively
            chunk_size = 150
            for i in range(0, len(analysis_text), chunk_size):
                chunk = analysis_text[i:i + chunk_size]
                yield _sse({"type": "token", "text": chunk})
                await asyncio.sleep(0)   # yield event loop
        except Exception as e:
            print(f"[legal/stream] Extended Thinking failed ({e}), falling back to Opus streaming")
            use_thinking = False   # fall through to streaming below

    if not (use_thinking and chosen_model == OPUS):
        # True streaming — Sonnet or Opus (no thinking).
        # If we hit max_tokens, automatically continue via assistant-prefill
        # pattern: feed the partial response back as the last assistant message
        # and Claude resumes mid-token without re-greeting or re-recapping.
        MAX_CONTINUATIONS = 3  # safety cap (3 = up to 4×32K tokens of output)
        language = getattr(payload, "language", "th")
        cont_status = {
            "th": "กำลังเขียนต่อจากที่ค้างไว้... ({i}/{n})",
            "en": "Continuing from where it stopped... ({i}/{n})",
            "zh": "正在从中断处继续... ({i}/{n})",
        }.get(language, "Continuing... ({i}/{n})")

        messages = [{"role": "user", "content": content}]

        try:
            for attempt in range(MAX_CONTINUATIONS + 1):
                async with _claude.messages.stream(
                    model=chosen_model,
                    max_tokens=64000,
                    temperature=0.3,
                    messages=messages,
                ) as stream:
                    async for text in stream.text_stream:
                        analysis_text += text
                        yield _sse({"type": "token", "text": text})
                    final_msg = await stream.get_final_message()
                    stop_reason = getattr(final_msg, "stop_reason", "unknown") or "unknown"

                if stop_reason != "max_tokens":
                    break  # natural finish (end_turn) or stop_sequence — done

                if attempt >= MAX_CONTINUATIONS:
                    print(f"[legal/stream] hit MAX_CONTINUATIONS cap; accepting truncation")
                    break

                # Prefill with what we have so far. Anthropic continues the
                # assistant message instead of starting a new turn — no recap,
                # no greeting, just resumes the next token.
                # Strip trailing whitespace so Claude doesn't get confused.
                prefill = analysis_text.rstrip()
                if not prefill:
                    break  # nothing to continue from
                messages = [
                    {"role": "user",      "content": content},
                    {"role": "assistant", "content": prefill},
                ]
                yield _sse({"type": "status", "message": cont_status.format(i=attempt + 1, n=MAX_CONTINUATIONS)})
                print(f"[legal/stream] max_tokens hit at {len(analysis_text)} chars — continuing (attempt {attempt + 1}/{MAX_CONTINUATIONS})")
        except Exception as e:
            error_message = str(e)
            err_msg = f"\n\n❌ เกิดข้อผิดพลาด: {e}"
            analysis_text += err_msg
            yield _sse({"type": "token", "text": err_msg})
            stop_reason = "error"

    print(f"[legal/stream] final stop_reason={stop_reason} chars={len(analysis_text)}")

    # ── 6. Done ───────────────────────────────────────────────────────────────
    yield _sse({
        "type":                  "done",
        "model_used":            chosen_model,
        "complexity_score":      complexity,
        "extended_thinking_used": use_thinking,
        "attachments_count":     len(attachments),
        "legal_search_results":  legal_search_results,
        "debug":                 debug,
        "stop_reason":           stop_reason,         # end_turn = complete; anything else = problem
        "error_message":         error_message,
        "char_count":            len(analysis_text),
    })
    yield "data: [DONE]\n\n"


@router.post("/analyze/stream")
async def legal_analyze_stream(
    payload: AnalyzeIn,
    user_id: str = Depends(get_current_user_id),
):
    """SSE streaming endpoint — tokens arrive in real-time instead of polling.

    Event types emitted:
      status  — progress message (UI status bar)
      token   — Claude text chunk (append to result)
      legal_search — Brave search results (sent early, before Claude finishes)
      done    — final metadata (model, debug, legal_search_results)
    """
    return StreamingResponse(
        _stream_analysis(payload, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # tells nginx/Cloudflare not to buffer
            "Connection":       "keep-alive",
        },
    )


@router.post("/analyze")
async def legal_analyze_start(
    payload: AnalyzeIn,
    user_id: str = Depends(get_current_user_id),
):
    """Start an analysis job in the background. Returns {job_id} immediately.

    Clients poll GET /api/legal/analyze/jobs/{job_id} for completion.
    A finished result remains available for ~1 hour after completion.
    """
    _evict_stale_jobs()
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status":     "pending",
        "started_at": time.time(),
        "user_id":    user_id,
    }
    # asyncio.create_task is sufficient — the analysis is purely async I/O
    # against Anthropic + Supabase. Survives the request lifecycle.
    asyncio.create_task(_run_job(job_id, payload, user_id))
    return {"job_id": job_id, "status": "pending"}


@router.get("/analyze/jobs/{job_id}")
async def legal_analyze_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return current status of an analysis job. 404 if unknown / evicted."""
    _evict_stale_jobs()
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")
    return {
        "job_id":      job_id,
        "status":      job.get("status"),
        "started_at":  job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "result":      job.get("result"),
        "error":       job.get("error"),
    }


async def _draft_document(doc_type: str, case: CaseInput, model: str, attachments: list[dict]) -> str:
    plaintiff = case.plaintiff_name or "(ระบุชื่อโจทก์ตามเอกสาร)"
    defendant = case.defendant_name or "(ระบุชื่อจำเลยตามเอกสาร)"
    court = case.court or "(ระบุศาลตามเอกสาร)"

    common_intro = f"""คุณคือทนายความไทยอาวุโสที่มีประสบการณ์ว่าความในศาลมามากกว่า 20 ปี
อ่านเอกสารแนบ (ถ้ามี) อย่างละเอียด ดึงข้อเท็จจริงจริงทุกชิ้น (ชื่อ วันที่ ตัวเลข เลขบัญชี เลขมาตรา เลขฎีกา)
ใช้ชื่อจริงของคู่ความตามที่ปรากฏในเอกสาร — ห้ามใช้ placeholder ห้ามใช้คำว่า "โจทก์/จำเลย" ลอยๆ
ภาษาที่ใช้: ภาษากฎหมายไทยทางการระดับศาล — "อันเป็นเหตุให้" "ย่อม" "พึงต้อง" "ต้องด้วย" ฯลฯ

ข้อมูลคดี: {_case_block(case)}
"""

    if doc_type == "complaint":
        prompt = common_intro + f"""

🎯 **ภารกิจ: ร่างคำฟ้องในมุมมองทนาย{plaintiff} (ฝ่ายโจทก์)**

⚠️ **ข้อสำคัญ:** เอกสารฉบับนี้คือ "คำฟ้อง" ดังนั้นต้องเขียนจากมุมมองของทนายโจทก์เสมอ
**ไม่ว่าผู้ใช้จะเลือกมุมมองคดีเป็นโจทก์ จำเลย หรือทั้งสองฝ่าย** — คำฟ้องคือเอกสารของฝ่ายโจทก์เท่านั้น

🔑 **กรอบความคิดที่ต้องใช้ตลอดร่าง:**
1. เขียนเชิงรุก — บีบให้จำเลยต้องตอบทุกประเด็น
2. ระบุข้อเท็จจริงให้แน่น **อุดทุกช่องที่จำเลยจะใช้ขอยกฟ้อง** (อายุความ / อำนาจฟ้อง / เขตอำนาจ / องค์ประกอบครบ)
3. คำขอท้ายฟ้องต้องครบ — เผื่อ "คำขอประธาน" และ "คำขอรอง" ในกรณีศาลให้ไม่ครบ
4. ทุนทรัพย์คำนวณให้ละเอียด พร้อมดอกเบี้ย ค่าเสียหายต่อเนื่อง ค่าทนาย ค่าฤชา

# คำฟ้อง

ศาล{court}
คดีหมายเลขดำที่ ..../25.. (ศาลกรอก)

ระหว่าง
{plaintiff} ............................ โจทก์
{defendant} .......................... จำเลย

เรื่อง  (ระบุประเภทคดีและสาระสำคัญ เช่น "เรียกค่าเสียหายตามสัญญาซื้อขาย" / "ฉ้อโกง เรียกเงินคืน")

ข้าพเจ้า {plaintiff} โจทก์ ขอฟ้อง {defendant} จำเลย ต่อศาลที่เคารพ ดังต่อไปนี้

## ข้อ 1. ฐานะของคู่ความและอำนาจฟ้อง
- สถานะของโจทก์ (บุคคลธรรมดา/นิติบุคคล + ที่อยู่ + เลขทะเบียน — ตามเอกสาร)
- สถานะของจำเลย
- **อำนาจฟ้อง** — ระบุว่าโจทก์เป็นผู้เสียหายโดยตรงตามกฎหมายอย่างไร อ้างอิงมาตราที่ให้สิทธิ์ฟ้อง

## ข้อ 2. ข้อเท็จจริงโดยละเอียด
เขียนเป็นย่อหน้าตามลำดับเวลา ≥ 6-8 ย่อหน้า — แต่ละย่อหน้าต้องมีวัน/เดือน/ปี และพยานหลักฐานอ้างอิง
- ย่อหน้า ๑: ความสัมพันธ์ระหว่างโจทก์-จำเลยก่อนเกิดเหตุ
- ย่อหน้า ๒-๕: ลำดับเหตุการณ์ที่นำมาสู่การฟ้อง (ตามเอกสาร)
- ย่อหน้า ๖-๗: การทวงถาม / การติดตาม / ความเสียหายที่เกิดขึ้น
- ย่อหน้า ๘: เหตุที่ต้องนำคดีมาสู่ศาล

## ข้อ 3. มูลเหตุฟ้อง / การกระทำของจำเลย
- พฤติการณ์ของ {defendant} ที่ผิดสัญญา / ผิดกฎหมาย / ทำละเมิด — ระบุชัดเจน
- เชื่อมโยงข้อเท็จจริงกับองค์ประกอบความผิดทางแพ่ง/อาญาที่อ้าง
- **ห้ามขาดองค์ประกอบใด** — ตรวจให้ครบทุกองค์ประกอบของฐานความผิดที่ฟ้อง

## ข้อ 4. ความเสียหาย
- ความเสียหายที่เกิดขึ้นจริง (Actual damages) — รายการละเอียด พร้อมหลักฐานอ้างอิง
- ความเสียหายต่อเนื่อง (Consequential damages) ถ้ามี
- ดอกเบี้ย — อัตราและจุดเริ่มนับ ตามมาตรา 7 / มาตรา 224 ป.พ.พ. (ตามคดี)
- การคำนวณรวมขั้นสุดท้าย — แสดงเป็นตาราง
| รายการ | จำนวนเงิน (บาท) | ฐานทางกฎหมาย |
| --- | --- | --- |

### ข้อ 4.1 ตารางสรุปธุรกรรมการเงิน (ถ้าคดีเกี่ยวกับการเงิน — สำคัญมาก)
ถ้าเอกสารปรากฏการโอนเงิน/รับเงิน/จ่ายเงิน — **ต้องสร้างตาราง markdown ครบทุกธุรกรรม**:

| ลำดับ | วันที่ | จากบัญชี/ผู้โอน | ถึงบัญชี/ผู้รับ | จำนวนเงิน (บาท) | หมายเหตุ |
| --- | --- | --- | --- | --- | --- |

**กฎ:** ถ้ามีธุรกรรม **เกิน 5 รายการ** ต้องใช้ตารางสวยงามสมบูรณ์ — รวมยอดท้ายตาราง + แยกหมวด (โอนค้ำประกัน / ชำระค่า ... / โอนคืน) + ถ้ามีบัญชีม้าหลายทอด แยกหัวข้อย่อย "ทอดที่ ๑", "ทอดที่ ๒"

## ข้อ 5. กฎหมายและคำพิพากษาฎีกาที่อ้างอิง
- ระบุมาตราเต็มอย่างน้อย 4-6 มาตรา + ชื่อกฎหมาย + ใจความ + การประยุกต์
- ฎีกาที่สนับสนุนฝ่ายโจทก์ ≥ 2-3 เลข พร้อมเลข/ปี + ประเด็น

## ข้อ 6. ปิดช่องที่จำเลยอาจใช้ยกฟ้อง (Pre-empt Defenses)
ระบุชัดเจนในคำฟ้องเลยว่า:
- **อายุความ** — ฟ้องในเวลาที่กฎหมายกำหนด เพราะเริ่มนับเมื่อ ... (ระบุเหตุการณ์)
- **อำนาจฟ้อง** — โจทก์มีอำนาจฟ้องเพราะ ...
- **เขตอำนาจศาล** — ศาลนี้มีเขตอำนาจเพราะ ...
- (อื่นๆ ที่จำเลยอาจยก)

## ข้อ 7. คำขอท้ายฟ้อง
**คำขอประธาน:**
1. ให้ {defendant} ชำระเงิน ... บาท พร้อมดอกเบี้ยอัตรา ...% ต่อปี นับแต่วันที่ ... จนกว่าจะชำระเสร็จ
2. ให้ {defendant} ชำระค่าฤชาธรรมเนียมและค่าทนายความแทนโจทก์
3. (ขอเฉพาะคดี เช่น เพิกถอนนิติกรรม / ส่งมอบทรัพย์ / ห้ามทำการ ...)

**คำขอรอง (เผื่อศาลให้ไม่ครบ):**
1. (ระบุคำขอที่ลดหย่อนกว่าแต่ยังเป็นประโยชน์ต่อโจทก์)

ขอศาลที่เคารพได้โปรดพิจารณาพิพากษาให้เป็นไปตามคำขอท้ายฟ้องของโจทก์ทุกประการ
ควรมิควรแล้วแต่จะโปรด

ลงชื่อ ............................ โจทก์
ลงชื่อ ............................ ทนายโจทก์

⚠ ทนายผู้รับคดีต้องตรวจสอบก่อนยื่นจริง — ปรับข้อความเข้ารูปคดี
"""
    elif doc_type == "defense":
        prompt = common_intro + f"""

🛡️ **ภารกิจ: ร่างคำให้การจำเลยในมุมมองทนาย{defendant} (ฝ่ายจำเลย)**

⚠️ **ข้อสำคัญ:** เอกสารฉบับนี้คือ "คำให้การจำเลย" ดังนั้นต้องเขียนจากมุมมองของทนายจำเลยเสมอ
**ไม่ว่าผู้ใช้จะเลือกมุมมองคดีเป็นโจทก์ จำเลย หรือทั้งสองฝ่าย** — คำให้การคือเอกสารของฝ่ายจำเลยเท่านั้น

🔑 **กรอบความคิดที่ต้องใช้ตลอดร่าง:**
1. **เป้าหมายสูงสุด: ขอให้ศาลยกฟ้องโจทก์** — โจมตีคำฟ้องเชิงรูปแบบก่อนเข้าเนื้อหา
2. ปฏิเสธข้อกล่าวหาทุกข้ออย่างมีเหตุผล + พยานหลักฐานสนับสนุน
3. ยกข้อต่อสู้ทางกฎหมายให้ครบ (Procedural + Substantive Defenses)
4. ถ้ามีสิทธิเรียกร้องกลับ → **ฟ้องแย้ง** ในคดีเดียวกัน
5. แม้แพ้คดี ก็ต้องลดทุนทรัพย์ให้น้อยที่สุด

# คำให้การจำเลย

ศาล{court} · คดีหมายเลขดำที่ ..../25..
ระหว่าง
{plaintiff} ........................... โจทก์
{defendant} ......................... จำเลย

ข้าพเจ้า {defendant} จำเลย ขอยื่นคำให้การต่อสู้คดีนี้ ดังต่อไปนี้

## ข้อ 1. ฐานะของจำเลยและการยอมรับเขตอำนาจศาล
- สถานะของจำเลย (พร้อมที่อยู่ตามเอกสาร)
- **เรื่องเขตอำนาจศาล** — ถ้าโจทก์ฟ้องผิดศาล ให้คัดค้านที่นี่ + ขอให้โอนคดี/ยกฟ้อง

## ข้อ 2. ปฏิเสธข้ออ้างของโจทก์ทีละข้อ
{defendant} ขอเรียนต่อศาลที่เคารพว่า ปฏิเสธข้อกล่าวหาของโจทก์ทุกข้อทั้งสิ้น
- **ข้อ 1 (ของฟ้อง):** ปฏิเสธว่า ... ความจริงคือ ... (พร้อมพยานหลักฐาน)
- **ข้อ 2 (ของฟ้อง):** ปฏิเสธว่า ... ความจริงคือ ...
- (ปฏิเสธทุกข้อในคำฟ้องอย่างละเอียด — อย่าตอบรวมๆ)

## ข้อ 3. ข้อต่อสู้เชิงรูปแบบ (Procedural Defenses) — โจมตีคำฟ้อง
**ถ้าข้อใดข้อหนึ่งสำเร็จ ศาลต้องยกฟ้องโดยไม่ต้องเข้าเนื้อหา:**

### 3.1 อายุความ
- คดีนี้มีอายุความตามมาตรา ... ป.พ.พ. ระยะเวลา ... ปี
- เริ่มนับเมื่อ ... (เหตุการณ์ที่กฎหมายกำหนด)
- โจทก์ฟ้องเมื่อ ... ซึ่งเลยอายุความแล้ว → **ขอให้ยกฟ้อง**

### 3.2 อำนาจฟ้อง / Standing
- โจทก์ไม่ใช่ผู้เสียหายโดยตรง / ไม่มีนิติสัมพันธ์ / ไม่มีอำนาจดำเนินคดีแทน ...
- ขอให้ศาลตัดสินยกฟ้องเพราะโจทก์ไม่มีอำนาจฟ้อง

### 3.3 ความครบขององค์ประกอบมูลคดี
- คำฟ้องของโจทก์ขาดข้อเท็จจริงสำคัญ ... ทำให้องค์ประกอบความผิดไม่ครบ

### 3.4 โมฆะ/โมฆียะของนิติกรรม (ถ้าเกี่ยวข้อง)
- นิติกรรมที่โจทก์อ้าง โมฆะ/โมฆียะ ตามมาตรา ... เพราะ ...

## ข้อ 4. ข้อต่อสู้เชิงเนื้อหา (Substantive Defenses)
ถ้าข้อต่อสู้ในข้อ 3 ไม่สำเร็จ จำเลยขอต่อสู้เนื้อหาดังนี้:

### 4.1 จำเลยไม่ได้กระทำตามที่โจทก์กล่าวอ้าง
- พยานหลักฐานที่จำเลยมี: เอกสาร / พยานบุคคล / พยานวัตถุ
- ลำดับเหตุการณ์ที่ถูกต้องตามจริง

### 4.2 ผู้เสียหายมีส่วนผิดเอง (Contributory Fault)
- โจทก์มีพฤติการณ์ที่ทำให้เกิดความเสียหายเอง ตามมาตรา 442 ป.พ.พ.

### 4.3 การชำระหนี้/การปลดหนี้/การหักกลบลบหนี้
- จำเลยได้ชำระแล้วเมื่อ ... (พร้อมหลักฐาน)

### 4.4 เหตุสุดวิสัย / เหตุพ้นวิสัย / ความยินยอม
- ระบุเหตุการณ์ตามมาตรา 8, 219, 437 ป.พ.พ. (ตามคดี)

### 4.5 ลดทุนทรัพย์ — แม้แพ้ก็ขอให้ค่าเสียหายน้อยที่สุด
- ค่าเสียหายที่โจทก์เรียกเกินจริงเพราะ ...
- ค่าเสียหายที่ควรจะเป็นจริง: ... บาท

## ข้อ 5. ฟ้องแย้ง (Counterclaim) — ถ้ามี
ถ้าจำเลยมีสิทธิเรียกร้องกลับโจทก์ ฟ้องแย้งในคดีนี้:
- มูลเหตุของฟ้องแย้ง
- จำนวนทุนทรัพย์
- คำขอท้ายฟ้องแย้ง

## ข้อ 6. กฎหมายและคำพิพากษาฎีกาที่อ้างอิง
- มาตราเต็มที่จำเลยอ้าง ≥ 4-6 มาตรา + ใจความ + การประยุกต์
- ฎีกาที่สนับสนุนฝ่ายจำเลย ≥ 2-3 เลข

## ข้อ 7. คำขอท้ายคำให้การ
1. **ขอให้ศาลยกฟ้องโจทก์** เพราะ ... (สรุปเหตุหลัก)
2. ให้โจทก์ชำระค่าฤชาธรรมเนียมและค่าทนายความแทนจำเลย
3. (ถ้ามีฟ้องแย้ง) ให้โจทก์ชำระเงิน ... บาท ตามฟ้องแย้ง

ขอศาลที่เคารพได้โปรดพิจารณาพิพากษาตามคำขอของจำเลยทุกประการ
ควรมิควรแล้วแต่จะโปรด

ลงชื่อ ............................ จำเลย
ลงชื่อ ............................ ทนายจำเลย

⚠ ทนายผู้รับคดีต้องตรวจก่อนยื่น — ปรับข้อความเข้ารูปคดี
"""
    else:
        return f"ประเภทเอกสารไม่รองรับ: {doc_type}"

    return await _claude_call(prompt, model, attachments, max_tokens=16000)


# ── Standalone Thai Legal Search endpoint ────────────────────────────────────

@router.get("/search")
async def thai_legal_search(
    q: str = Query(..., description="คำค้นหา เช่น 'ฉ้อโกง มาตรา 341' หรือ 'ผิดสัญญาเช่า'"),
    case_type: str = Query("", description="ประเภทคดี เช่น แพ่ง อาญา แรงงาน"),
    user_id: str = Depends(get_current_user_id),
):
    """
    ค้นหาฎีกา / กฎหมายไทยจากฐานข้อมูลภายนอก
    ใช้ Brave Search กรองเฉพาะแหล่งกฎหมายไทยที่น่าเชื่อถือ
    """
    brave_key = settings.brave_search_api_key
    if not brave_key:
        raise HTTPException(
            status_code=503,
            detail="Thai legal search ยังไม่พร้อมใช้งาน — กรุณาตั้งค่า BRAVE_SEARCH_API_KEY",
        )
    try:
        results = await search_thai_legal(
            api_key=brave_key,
            case_type=case_type,
            transcript=q,
        )
        return {
            "query": q,
            "results": results,
            "total": len(results),
            "sources": list({r["source_label"] for r in results}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ค้นหาไม่สำเร็จ: {str(e)}")


# ── Server-side PDF export ────────────────────────────────────────────────────
class ExportPdfRequest(BaseModel):
    markdown: str
    filename: str = "thai-law-analysis.pdf"
    lang: str = "th"
    perspective: str = ""
    case_title: str = ""
    plaintiff: str = ""
    defendant: str = ""
    court: str = ""
    case_number: str = ""
    claim_amount: float = 0.0

@router.post("/export-pdf")
async def export_pdf(
    payload: ExportPdfRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    รับ HTML string → fpdf2 + Noto fonts (embedded) → PDF blob
    Pure Python, no system-level dependencies.
    """
    import io
    import asyncio
    import functools
    from app.services.pdf_generator import generate_pdf

    lang = payload.lang
    case_meta = {
        "title":          payload.case_title,
        "plaintiff_name": payload.plaintiff,
        "defendant_name": payload.defendant,
        "court":          payload.court,
        "case_number":    payload.case_number,
        "claim_amount":   payload.claim_amount,
    }

    try:
        loop = asyncio.get_event_loop()
        pdf_bytes: bytes = await loop.run_in_executor(
            None,
            functools.partial(generate_pdf, payload.markdown, lang, case_meta, payload.perspective),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    safe_name = payload.filename.replace('"', "'")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "Content-Length":      str(len(pdf_bytes)),
        },
    )
