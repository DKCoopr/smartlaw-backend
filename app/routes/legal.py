"""Legal analysis — Claude reads PDFs natively (no Gemini middleman)."""
import asyncio
import base64
import re
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings
from app.database import get_supabase
from app.services.documents import ocr_image, _extract_docx_text, _extract_doc_text, DOCX_MIME, DOC_MIME
from app.services.brave_search import search_thai_legal, format_for_prompt

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

MAX_DOC_FILES   = 100
MAX_DOC_BYTES   = 32 * 1024 * 1024


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
    debug = {"case_id": case_id, "user_id": user_id, "rows_found": 0, "loaded": [], "errors": [], "subset": bool(doc_ids)}
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
        if total_bytes + len(file_bytes) > MAX_DOC_BYTES:
            debug["errors"].append(f"{name}: skipped — total size limit reached")
            break

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
        debug["loaded"].append({
            "name": name, "bytes": len(file_bytes), "mime": mime,
            "ocr_chars": len(attachment["ocr_text"]) if attachment["ocr_text"] else 0,
        })
        print(f"[legal/analyze] attached: {name} ({len(file_bytes)} bytes, {mime})")

    debug["total_bytes"] = total_bytes
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


def _build_content_blocks(prompt: str, attachments: list[dict]) -> list:
    blocks = []
    for att in attachments:
        b64 = base64.standard_b64encode(att["bytes"]).decode("ascii")
        if att["mime"] == "application/pdf":
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
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": att["mime"], "data": b64},
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
    max_tokens: int = 32000,
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
        return await _stream(min(max_tokens, 24000))


# ── Prompt ────────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_FRAME = """คุณคือทนายความไทยอาวุโสระดับเนติบัณฑิตไทย มีประสบการณ์ว่าความและที่ปรึกษากฎหมายของสำนักงานชั้นนำมากว่า 25 ปี
เชี่ยวชาญทุกแขนง — แพ่ง / อาญา / แรงงาน / ปกครอง / ภาษี / ครอบครัว / มรดก / IP / สัญญาธุรกิจ

ภารกิจ: เขียน "บทสรุปคดีและแนวทางต่อสู้ขั้นสุด" ในระดับเดียวกับ memo ภายในสำนักงานทนายความระดับ Tier-1
— ละเอียด ลึก เป็นภาษากฎหมายไทยทางการ ใช้งานได้จริงในศาล

🚨 **กฎเหล็ก — ห้ามฝ่าฝืน:**
1. **ต้องเขียนให้จบทุกหัวข้อ 1-14** — ห้ามตัดกลางคัน ห้ามจบกลางหัวข้อ ถ้ายาวต้องบีบเนื้อหาให้พอ แต่ห้ามขาดหัวข้อ
2. **หัวข้อ 9 (แนวต่อสู้) และ 14 (บทสรุป+โอกาสแพ้ชนะ) ต้องละเอียดที่สุด** — เป็นหัวใจของรายงาน
3. **อ่านเอกสารแนบก่อนตอบ** — ดึงข้อมูลทุกส่วน (วันที่ ตัวเลข ชื่อบุคคล/นิติบุคคล มาตรากฎหมาย คำพิพากษา เลขบัญชี)
4. **ใช้ชื่อจริงของทุกคนที่ปรากฏในเอกสาร** — ห้ามใช้ "โจทก์/จำเลย" ลอยๆ ต้องเขียนชื่อจริง
5. **อ้างอิงเลขมาตราเต็ม + ชื่อกฎหมาย + เลขฎีกา** ทุกครั้ง
6. **ภาษากฎหมายไทยทางการ** — "อันเป็นเหตุให้" / "ย่อม" / "พึง" / "ต้องด้วย" ฯลฯ
7. **ความยาวรวม ≥ 5,000 คำ** — เขียนละเอียดเหมือน memo จริง
8. **ใช้ markdown** — `**bold**` / bullet `•` / `1.` `2.` `3.` / heading `## หัวข้อ`
"""


def _analysis_prompt(case: CaseInput, perspective: str, has_attachments: bool) -> str:
    p_label = _perspective_label(perspective)
    directive = _perspective_directive(perspective, case.plaintiff_name or "", case.defendant_name or "")

    if not has_attachments:
        return f"""{ANALYSIS_SYSTEM_FRAME}

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

(ใช้โครงสร้าง 14 หัวข้อตามด้านล่าง)
"""

    if has_attachments:
        return f"""{ANALYSIS_SYSTEM_FRAME}

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
ไทม์ไลน์ครบถ้วน ≥ 8-12 เหตุการณ์ ระบุวัน/เดือน/ปี (ตามเอกสาร) พร้อมรายละเอียดเชิงลึกแต่ละเหตุการณ์

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

    # 4. Build prompt — inject legal DB search results at the top
    base_prompt = _analysis_prompt(payload.case, payload.perspective, has_attachments=bool(attachments))
    legal_db_block = format_for_prompt(legal_search_results)
    # Inject search results BEFORE the main analysis template so Claude sees them first
    prompt = f"{legal_db_block}\n\n{base_prompt}" if legal_db_block else base_prompt

    # 5. Run Claude with PDFs as document blocks + enriched prompt
    analysis_text = await _claude_call(
        prompt, chosen_model, attachments,
        max_tokens=32000,
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
