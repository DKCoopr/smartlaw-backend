"""Legal analysis — Claude reads PDFs natively (no Gemini middleman)."""
import asyncio
import base64
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings
from app.database import get_supabase
from app.services.documents import ocr_image

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

MAX_DOC_FILES   = 8
MAX_DOC_BYTES   = 18 * 1024 * 1024


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


def _content_type(file_type: str) -> str:
    return {
        "pdf":  "application/pdf",
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "txt":  "text/plain",
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
    s = 0
    total_text = (case.transcript or "") + (case.notes or "")
    s += min(3, len(total_text) // 1500)
    s += min(3, len(attachments))
    if (case.claim_amount or 0) >= 5_000_000: s += 1
    if (case.claim_amount or 0) >= 50_000_000: s += 1
    if case.case_type in ("ปกครอง", "อาญา", "ภาษี"): s += 1
    if "ฎีกา" in total_text or "อุทธรณ์" in total_text: s += 1
    return s


def _pick_model(case: CaseInput, attachments: list[dict]) -> str:
    return OPUS if _complexity_score(case, attachments) >= 4 else SONNET


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


async def _claude_call(prompt: str, model: str, attachments: list[dict], max_tokens: int = 32000) -> str:
    """Stream Claude response so we can use very high max_tokens without timeouts."""
    content = _build_content_blocks(prompt, attachments)

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
        # Retry with smaller token budget
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

    if not has_attachments:
        return f"""{ANALYSIS_SYSTEM_FRAME}

⚠️ **ไม่มีเอกสารแนบในคดีนี้** — วิเคราะห์ตามข้อมูลที่ผู้ใช้กรอกเท่านั้น

ตอนต้นคำตอบ ให้แจ้งผู้ใช้ว่า: "ℹ️ คดีนี้ยังไม่มีเอกสารแนบ — การวิเคราะห์อิงข้อมูลที่กรอกเท่านั้น คุณภาพจะดีขึ้นมากถ้าอัปโหลดเอกสารคดีจริง (PDF คำฟ้อง / สัญญา / คำพิพากษา)"

== ข้อมูลคดี ==
{_case_block(case)}

== มุมมอง ==
{p_label}

== โครงสร้างคำตอบ ==
ตอบครบทุกหัวข้อในเทมเพลตด้านล่าง — ใช้ชื่อจริงของ {case.plaintiff_name or '(ระบุโจทก์)'} และ {case.defendant_name or '(ระบุจำเลย)'} ตลอด

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

== โครงสร้างคำตอบที่ต้องการ ==
ตอบทุกหัวข้อข้างล่าง — ห้ามขาดหัวข้อใด ห้ามตอบสั้น
ดึงชื่อจริง วันที่จริง ตัวเลขจริง จากเอกสารแนบ ห้ามใช้ placeholder ห้ามใช้ตัวอย่างทั่วไป

# 📋 บทสรุปคดี
(ขึ้นต้นด้วยหัวข้อชื่อคดีตามที่ปรากฏในเอกสารจริง — ไม่ใช่ผู้ใช้กรอก)

## 1. บทสรุปผู้บริหาร (Executive Summary)
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

## 7. จุดแข็งของฝ่าย{p_label}
≥ 5 ข้อ พร้อมเหตุผลสนับสนุน

## 8. จุดอ่อน/ความเสี่ยงของฝ่าย{p_label}
≥ 4 ข้อ พร้อมวิธีบรรเทาแต่ละจุด

## 9. แนวทางต่อสู้คดีในชั้นศาล (Ultimate Court Strategy) — หัวใจของรายงาน

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

    # 2. Pick model (Sonnet for simple, Opus for complex/heavy attachments)
    chosen_model = _pick_model(payload.case, attachments)

    # 3. Run Claude with PDFs as document blocks + final prompt
    prompt = _analysis_prompt(payload.case, payload.perspective, has_attachments=bool(attachments))

    analysis_text = await _claude_call(prompt, chosen_model, attachments, max_tokens=32000)

    # 4. Document drafts (in parallel, also with attachments for context)
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
        "complexity_score": _complexity_score(payload.case, attachments),
        "attachments_count": len(attachments),
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

    common_intro = f"""คุณคือทนายความไทยอาวุโส อ่านเอกสารแนบ (ถ้ามี) อย่างละเอียด แล้วร่างเอกสารต่อไปนี้
ใช้ชื่อจริงของคู่ความตามที่ปรากฏในเอกสาร — ห้ามใช้ placeholder
ข้อมูลคดี: {_case_block(case)}
"""

    if doc_type == "complaint":
        prompt = common_intro + f"""

ภารกิจ: ร่าง **คำฟ้อง** ที่สมบูรณ์แบบ ใช้ในศาลได้จริง ตามรูปแบบทางการของศาลไทย

# คำฟ้อง

ศาล{court}
คดีหมายเลขดำที่ ..../25.. (ศาลกรอก)

ระหว่าง
{plaintiff} ............................ โจทก์
{defendant} .......................... จำเลย

## ข้อ 1. ฐานะของคู่ความและอำนาจฟ้อง
(ความเป็นนิติบุคคล/บุคคล + อำนาจฟ้อง — ดูจากเอกสารแนบ)

## ข้อ 2. ข้อเท็จจริง
(เขียนข้อเท็จจริงเป็นย่อหน้าๆ ≥ 5 ย่อหน้า — เหตุการณ์ที่นำมาสู่การฟ้อง)

## ข้อ 3. มูลเหตุฟ้อง / การกระทำของจำเลย
(พฤติการณ์ของ {defendant} ที่ผิดสัญญา/ผิดกฎหมาย โดยละเอียด)

## ข้อ 4. ความเสียหาย
(จำนวน + การคำนวณ + ฐานทางกฎหมาย)

## ข้อ 5. กฎหมายที่อ้างอิง

## ข้อ 6. คำขอท้ายฟ้อง
1. ให้ {defendant} ชำระเงิน ... บาท พร้อมดอกเบี้ย ...
2. ค่าฤชาธรรมเนียม + ค่าทนาย
3. (ขออื่นๆ)

(ลงท้ายตามแบบราชการ)

⚠ ทนายผู้รับคดีต้องตรวจสอบก่อนยื่นจริง
"""
    elif doc_type == "defense":
        prompt = common_intro + f"""

ภารกิจ: ร่าง **คำให้การจำเลย** ที่สมบูรณ์ ใช้ในศาลได้

# คำให้การจำเลย

ศาล{court} · คดีหมายเลขดำที่ ..../25..
ระหว่าง
{plaintiff} ........................... โจทก์
{defendant} ......................... จำเลย

## ข้อ 1. ฐานะของจำเลย

## ข้อ 2. ปฏิเสธข้ออ้างของโจทก์
{defendant} ขอเรียนต่อศาลที่เคารพว่า ปฏิเสธข้อกล่าวหาทั้งสิ้น โดยมีเหตุผลโดยละเอียด:
(ปฏิเสธทีละข้อกล่าวหา)

## ข้อ 3. ข้อต่อสู้ของจำเลย
≥ 4-6 ข้อ พร้อมเหตุผลและกฎหมายอ้างอิง — เช่น ขาดอายุความ / ไม่มีอำนาจฟ้อง / ชำระแล้ว / นิติกรรมโมฆะ

## ข้อ 4. กฎหมายและคำพิพากษาฎีกาที่อ้างอิง

## ข้อ 5. คำขอท้ายคำให้การ
1. ให้ยกฟ้องโจทก์
2. ค่าฤชาและค่าทนาย

⚠ ทนายต้องตรวจก่อนยื่น
"""
    elif doc_type == "contract":
        prompt = common_intro + f"""

ภารกิจ: ตรวจสัญญาในคดี ออกรายงานความเสี่ยง

# รายงานตรวจสัญญาและความเสี่ยง
คู่สัญญา: {plaintiff} กับ {defendant}

## 1. สรุปสัญญาโดยย่อ
## 2. 🔴 Clauses ที่ต้องระวังที่สุด (≥ 5 จุด)
ข้อที่ / ปัญหา / ผลกระทบ / คำแนะนำ
## 3. 🟡 ความเสี่ยงโมฆะ/ขัดต่อกฎหมาย (≥ 3 จุด)
## 4. 🟢 จุดแข็งของสัญญาที่ต้องรักษา (≥ 3 จุด)
## 5. 📋 ข้อเสนอแนะในการเจรจา (≥ 7 ข้อ พร้อมภาษาทางเลือก)
## 6. ภาพรวมความเสี่ยง 0-10 + คำแนะนำสุดท้าย

⚠ ต้องผ่านทนายก่อนใช้
"""
    else:
        return f"ประเภทเอกสารไม่รองรับ: {doc_type}"

    return await _claude_call(prompt, model, attachments, max_tokens=16000)
