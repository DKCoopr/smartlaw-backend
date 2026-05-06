"""Legal analysis — Claude reads PDFs natively (no Gemini middleman)."""
import asyncio
import base64
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings
from app.database import get_supabase

router = APIRouter(prefix="/api/legal", tags=["legal"])

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


async def _load_case_attachments(case_id: str, user_id: str) -> tuple[list[dict], dict]:
    """Pull case docs from Supabase Storage. Returns (attachments, debug_info)."""
    debug = {"case_id": case_id, "user_id": user_id, "rows_found": 0, "loaded": [], "errors": []}
    if not case_id:
        debug["errors"].append("no case_id provided")
        return [], debug

    db = get_supabase()
    try:
        rows = (
            db.table("documents")
            .select("*")
            .eq("case_id", case_id)
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
    except Exception as e:
        debug["errors"].append(f"db query failed: {e}")
        print(f"[legal/analyze] DB query failed for case_id={case_id}: {e}")
        return [], debug

    rows_data = rows.data or []
    debug["rows_found"] = len(rows_data)
    print(f"[legal/analyze] case_id={case_id} → {len(rows_data)} document rows")

    out = []
    total_bytes = 0
    for d in rows_data[:MAX_DOC_FILES]:
        path = d.get("storage_path")
        name = d.get("original_name") or d.get("doc_label") or "document"
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
        out.append({
            "name": name,
            "mime": _content_type(d.get("file_type")),
            "bytes": file_bytes,
        })
        total_bytes += len(file_bytes)
        debug["loaded"].append({"name": name, "bytes": len(file_bytes), "mime": _content_type(d.get("file_type"))})
        print(f"[legal/analyze] attached: {name} ({len(file_bytes)} bytes, {_content_type(d.get('file_type'))})")

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
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": att["mime"], "data": b64},
            })
        elif att["mime"] == "text/plain":
            try:
                text_content = att["bytes"].decode("utf-8", errors="replace")[:50_000]
                blocks.append({"type": "text", "text": f"=== ไฟล์: {att['name']} ===\n{text_content}\n=== จบไฟล์ ==="})
            except Exception:
                pass
    blocks.append({"type": "text", "text": prompt})
    return blocks


async def _claude_call(prompt: str, model: str, attachments: list[dict], max_tokens: int = 16000) -> str:
    content = _build_content_blocks(prompt, attachments)
    try:
        msg = await _claude.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        # Retry with smaller max_tokens
        msg = await _claude.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.3,
            messages=[{"role": "user", "content": content}],
        )
    return (msg.content[0].text or "").strip()


# ── Prompt ────────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_FRAME = """คุณคือทนายความไทยอาวุโสระดับเนติบัณฑิตไทย มีประสบการณ์ว่าความและที่ปรึกษากฎหมายของสำนักงานชั้นนำมากว่า 25 ปี
เชี่ยวชาญทุกแขนง — แพ่ง / อาญา / แรงงาน / ปกครอง / ภาษี / ครอบครัว / มรดก / IP / สัญญาธุรกิจ

ภารกิจ: เขียน "บทสรุปคดีและแนวทางต่อสู้ขั้นสุด" ในระดับเดียวกับ memo ภายในสำนักงานทนายความระดับ Tier-1
— ละเอียด ลึก เป็นภาษากฎหมายไทยทางการ ใช้งานได้จริงในศาล

หลักสำคัญ:
1. **อ่านเอกสารแนบทั้งหมดก่อนตอบ** — ดึงข้อมูลทุกส่วนที่จำเป็น (วันที่ ตัวเลข ชื่อบุคคล/นิติบุคคล มาตรากฎหมาย คำพิพากษา)
2. **ใช้ชื่อจริงของทุกคนที่ปรากฏในเอกสาร** — ห้ามใช้คำว่า "โจทก์" หรือ "จำเลย" ลอยๆ ต้องเขียนชื่อจริงตามเอกสาร
3. **อ้างอิงเลขมาตราเต็ม + ชื่อกฎหมาย + เลขฎีกา** ทุกครั้งที่กล่าวถึงตัวบทกฎหมายหรือบรรทัดฐาน
4. **เขียนเป็นภาษากฎหมายไทยทางการ** ไม่ใช่ภาษาพูด ใช้คำว่า "อันเป็นเหตุให้" / "ย่อม" / "พึง" / "ต้องด้วย" ฯลฯ
5. **ความยาวไม่ต่ำกว่า 4,000 คำ** — เขียนละเอียดเหมือน memo จริง ไม่สรุปสั้น
6. ใช้ markdown: **bold** / bullet `•` / ลำดับเลข `1.` `2.` `3.` / heading `## หัวข้อ`
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

## 9. แนวทางต่อสู้คดีขั้นสุด (Ultimate Strategy)
### 9.1 ก่อนยื่นฟ้อง / ก่อนขึ้นศาล
- หลักฐานที่ต้องเตรียม (รายการละเอียด)
- พยานบุคคลที่ต้องเชิญ + ประเด็นที่จะถาม
- เอกสารที่ต้องร่าง

### 9.2 ในชั้นพิจารณาคดี
- ประเด็นข้อสู้คดีหลัก ≥ 5 ประเด็น พร้อมเหตุผล
- แนวซักค้านพยานฝ่ายตรงข้าม
- การยกข้อต่อสู้ทางกฎหมาย (อายุความ / อำนาจฟ้อง / โมฆะ ฯลฯ)

### 9.3 ทางออกทางเลือก
- ประนีประนอม (เงื่อนไขที่รับ-ไม่รับ)
- ไกล่เกลี่ย
- ถอนฟ้อง

## 10. Timeline เชิงปฏิบัติ
- ทันที (0-7 วัน)
- ระยะสั้น (1-4 สัปดาห์)
- ระยะกลาง (1-3 เดือน)
- ระยะยาว (จนคำพิพากษา + บังคับคดี)

## 11. ประมาณการค่าใช้จ่าย
- ค่าฤชาธรรมเนียม
- ค่าทนาย (ช่วงต่ำ-สูง)
- ค่าใช้จ่ายอื่น (ผู้เชี่ยวชาญ/เดินทาง/เอกสาร)
- ระยะเวลารวม

## 12. ความเสี่ยงรอบด้าน
- กฎหมาย / การเงิน / เวลา / ชื่อเสียง / อาญา-วินัย

## 13. คำแนะนำทางยุทธวิธี
≥ 6 ข้อ จากประสบการณ์ทนายอาวุโส

## 14. บทสรุปและขั้นตอนถัดไป
สรุปย่อหน้าเดียว + Action Items 7 วันแรก เป็นเลข 1, 2, 3, ...

---

⚠️ **คำเตือน:** เอกสารนี้เป็นการประเมินจากระบบขั้นสูง ทนายผู้รับคดีต้องตรวจสอบและปรับใช้ตามรูปคดีจริง
"""


@router.post("/analyze")
async def legal_analyze(
    payload: AnalyzeIn,
    user_id: str = Depends(get_current_user_id),
):
    # 1. Pull actual case PDFs from Supabase storage (skip Gemini summary)
    attachments, debug = ([], {"case_id": None})
    if payload.case_id:
        attachments, debug = await _load_case_attachments(payload.case_id, user_id)

    print(f"[legal/analyze] case_id={payload.case_id} attachments={len(attachments)} bytes={sum(len(a['bytes']) for a in attachments)}")

    # 2. Pick model (Sonnet for simple, Opus for complex/heavy attachments)
    chosen_model = _pick_model(payload.case, attachments)

    # 3. Run Claude with PDFs as document blocks + final prompt
    prompt = _analysis_prompt(payload.case, payload.perspective, has_attachments=bool(attachments))

    try:
        analysis_text = await _claude_call(prompt, chosen_model, attachments, max_tokens=16000)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analysis failed ({chosen_model}): {str(e)}")

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

    return await _claude_call(prompt, model, attachments, max_tokens=8192)
