"""Legal analysis routes — Claude (Sonnet/Opus auto-selected) + GPT-4o polish."""
import asyncio
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings
from app.services.gpt4o import polish_thai_legal

router = APIRouter(prefix="/api/legal", tags=["legal"])

settings = get_settings()
_claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

# Models
SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-7"

# Heuristic: anything over this many chars of input → Opus (deeper reasoning).
COMPLEX_CHAR_THRESHOLD = 4000


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
    documents_summary: Optional[str] = ""   # joined ai_summary of docs in this case


class AnalyzeIn(BaseModel):
    case: CaseInput
    perspective: str = "both"          # plaintiff | defendant | both
    document_types: List[str] = []     # complaint | defense | contract


def _case_block(case: CaseInput) -> str:
    plaintiff = case.plaintiff_name or "(ยังไม่ระบุ)"
    defendant = case.defendant_name or "(ยังไม่ระบุ)"
    our = "ฝ่ายโจทก์" if case.our_client == "plaintiff" else ("ฝ่ายจำเลย" if case.our_client == "defendant" else "ทั้งสองฝ่าย")
    block = (
        f"ชื่อคดี: {case.title or '—'}\n"
        f"ประเภทคดี: {case.case_type or '—'}\n"
        f"ศาล: {case.court or '—'}\n"
        f"โจทก์: {plaintiff}\n"
        f"จำเลย: {defendant}\n"
        f"ลูกความที่เรารับ: {our}\n"
        f"ทุนทรัพย์: {case.claim_amount or 0:,.0f} บาท\n"
        f"รายละเอียด/คำบอกเล่า:\n{case.transcript or case.notes or '—'}"
    )
    if case.documents_summary:
        block += f"\n\n== สรุปเอกสารในคดี ==\n{case.documents_summary}"
    return block


def _perspective_label(p: str) -> str:
    return {"plaintiff": "ฝ่ายโจทก์", "defendant": "ฝ่ายจำเลย"}.get(p, "ทั้งสองฝ่าย")


def _pick_model(input_text: str) -> str:
    return OPUS if len(input_text) >= COMPLEX_CHAR_THRESHOLD else SONNET


async def _claude_call(prompt: str, model: str, max_tokens: int = 4000) -> str:
    msg = await _claude.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return (msg.content[0].text or "").strip()


@router.post("/analyze")
async def legal_analyze(
    payload: AnalyzeIn,
    user_id: str = Depends(get_current_user_id),
):
    case_block = _case_block(payload.case)
    p_label = _perspective_label(payload.perspective)
    plaintiff = payload.case.plaintiff_name or "(โจทก์ยังไม่ระบุ)"
    defendant = payload.case.defendant_name or "(จำเลยยังไม่ระบุ)"

    main_prompt = f"""คุณคือทนายความไทยอาวุโสผู้เชี่ยวชาญทุกแขนงกฎหมาย ระดับเนติบัณฑิตและที่ปรึกษากฎหมายของสำนักงานชั้นนำ
วิเคราะห์คดีต่อไปนี้ให้ละเอียดที่สุดเท่าที่ทำได้ — ให้คำตอบเชิงลึก ครบถ้วน ครอบคลุม โดยอ้างอิงตัวบทจริง

== ข้อมูลคดี ==
{case_block}

== มุมมองที่ต้องวิเคราะห์ ==
{p_label} (ในคดีนี้: โจทก์ = {plaintiff} / จำเลย = {defendant})

== ข้อกำหนดสำคัญ ==
1. **ระบุชื่อโจทก์และจำเลยจริงในการวิเคราะห์ทุกครั้ง** — ห้ามใช้คำว่า "โจทก์" หรือ "จำเลย" ลอยๆ ต้องเขียน "{plaintiff}" หรือ "{defendant}" ตรงๆ
2. ตอบเป็นภาษาไทยทางกฎหมาย กระชับแต่ครบถ้วน
3. ตอบทุกหัวข้อข้างล่างนี้แบบละเอียดสูงสุด — ห้ามละเว้นหัวข้อใด
4. ใช้ markdown: **bold**, bullet `•`, ลำดับเลข `1.`

== โครงสร้างคำตอบที่ต้องการ ==

🔍 **บทสรุปผู้บริหาร (Executive Summary)**
สรุปคดีทั้งหมดในรูปแบบที่ผู้บริหารหรือลูกความเข้าใจในนาทีเดียว — ต้องครอบคลุม:
- คดีคืออะไร เกิดอะไรขึ้น ใครฟ้องใคร ฟ้องอะไร
- ผลลัพธ์คาดการณ์ + แนวทางสำคัญที่สุด

📚 **ข้อเท็จจริงโดยละเอียด (Detailed Facts)**
รายงานข้อเท็จจริงของคดีอย่างเป็นระบบ ครบทุกมิติ (อย่างน้อย 8-10 ข้อ พร้อมรายละเอียดเชิงลึก)

⚖️ **กฎหมายที่เกี่ยวข้อง (Applicable Laws)**
ระบุมาตรา/พรบ./ประมวลกฎหมายที่เกี่ยวข้องครบทุกข้อ (อย่างน้อย 5-8 ข้อ) พร้อม:
- เลขมาตราและชื่อกฎหมาย
- ใจความสำคัญของมาตรา (สรุปเป็นภาษาคนเข้าใจง่าย)
- การประยุกต์ใช้กับข้อเท็จจริงในคดีนี้
อ้างอิงคำพิพากษาฎีกาที่เกี่ยวข้องด้วย (ถ้านึกออก) อย่างน้อย 2-3 ฎีกา

🎯 **ประเมินโอกาสชนะคดี (Probability of Success)**
- ฝ่าย{plaintiff} (โจทก์): X%
- ฝ่าย{defendant} (จำเลย): Y%
- เหตุผลประกอบการประเมินอย่างละเอียด

💪 **จุดแข็งของฝ่าย{p_label}**
อย่างน้อย 5 ข้อพร้อมเหตุผลสนับสนุน

🛑 **จุดอ่อน/ความเสี่ยงของฝ่าย{p_label}**
อย่างน้อย 4 ข้อพร้อมแนวทางบรรเทาความเสี่ยง

🛡️ **แนวทางการต่อสู้ขั้นสุด (Ultimate Defense/Prosecution Strategy)**
นี่คือหัวใจของรายงาน — ต้องละเอียดที่สุด:

**ก่อนยื่น/ก่อนขึ้นศาล:**
- หลักฐานที่ต้องเตรียม (รายการละเอียด)
- พยานบุคคลที่ต้องเชิญ + ประเด็นที่จะถาม
- เอกสารที่ต้องร่าง

**ในชั้นศาล:**
- แนวประเด็นข้อสู้คดี (legal arguments) ทุกประเด็น
- แนวซักค้านพยานฝ่ายตรงข้าม
- การยกข้อต่อสู้ทางกฎหมาย (เช่น อายุความ, อำนาจฟ้อง, ความชอบด้วยกฎหมาย)

**ข้อต่อรอง/ทางออกทางเลือก:**
- ประนีประนอม
- ไกล่เกลี่ย
- แนวทางลด/หยุดความเสียหาย

📅 **Timeline เชิงปฏิบัติ (Action Timeline)**
ระบุระยะเวลาแต่ละขั้นตอนพร้อมรายละเอียด:
- ทันที (0-7 วัน)
- ระยะสั้น (1-4 สัปดาห์)
- ระยะกลาง (1-3 เดือน)
- ระยะยาว (3+ เดือนถึงคำพิพากษา)

💰 **ประเมินค่าใช้จ่ายและเวลา**
- ค่าฤชาธรรมเนียมศาล
- ค่าทนาย (ประมาณการ)
- ระยะเวลารวมที่คาดว่าใช้

⚠️ **ความเสี่ยงรอบด้าน**
- เสี่ยงต่อการขาดอายุความ
- เสี่ยงต่อค่าฤชาธรรมเนียมที่อาจเสียเพิ่ม
- เสี่ยงทางชื่อเสียง/ผลกระทบทางธุรกิจ
- เสี่ยงทางอาญา/วินัย (ถ้ามี)

🎓 **คำแนะนำทางยุทธวิธี (Tactical Recommendations)**
จากประสบการณ์ของทนายอาวุโส — Tips & Tricks ที่อาจชี้ขาดคดี อย่างน้อย 5 ข้อ

📝 **บทสรุปและขั้นตอนถัดไป**
สรุปทุกประเด็น + Action Items ที่ต้องทำต่อทันที

ลงท้ายด้วย:
"⚠️ ผลการวิเคราะห์นี้ใช้ AI ขั้นสูงและฐานความรู้กฎหมายไทย แต่เพื่อความปลอดภัยสูงสุด ทนายความที่รับคดีต้องตรวจสอบและปรับใช้ตามรูปคดีจริงก่อนใช้งาน"
"""

    chosen_model = _pick_model(case_block)

    try:
        analysis_text = await _claude_call(main_prompt, chosen_model, max_tokens=8000)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude analysis failed ({chosen_model}): {str(e)}")

    # Polish through GPT-4o (preserves all content, refines wording)
    polished = await polish_thai_legal(analysis_text)

    # Document drafts in parallel
    documents: dict[str, str] = {}
    if payload.document_types:
        results = await asyncio.gather(
            *[_draft_document(t, payload.case, chosen_model) for t in payload.document_types],
            return_exceptions=True,
        )
        for t, res in zip(payload.document_types, results):
            if isinstance(res, Exception):
                documents[t] = f"⚠️ ไม่สามารถร่างเอกสารได้: {str(res)}"
            else:
                documents[t] = res

    return {
        "analysis": polished,
        "model_used": chosen_model,
        "documents": documents,
    }


async def _draft_document(doc_type: str, case: CaseInput, model: str) -> str:
    case_block = _case_block(case)
    plaintiff = case.plaintiff_name or "(โจทก์ยังไม่ระบุ)"
    defendant = case.defendant_name or "(จำเลยยังไม่ระบุ)"
    court = case.court or "(ศาลยังไม่ระบุ)"

    if doc_type == "complaint":
        prompt = f"""คุณคือทนายความไทย ร่าง **คำฟ้อง** ให้ครบถ้วนสมบูรณ์ที่สุด ใช้ในศาลได้จริง

ข้อมูลคดี:
{case_block}

ใช้ชื่อจริงของคู่ความในคำฟ้อง:
- โจทก์: {plaintiff}
- จำเลย: {defendant}
- ศาล: {court}

ร่างคำฟ้องที่มี:
**คำฟ้อง**

ข้าพเจ้า {plaintiff} โจทก์ ขอฟ้อง {defendant} จำเลย
ต่อ{court}

**ข้อ 1. ข้อเท็จจริง**
(เขียนข้อเท็จจริงเชิงลึกอย่างน้อย 5 ย่อหน้า ครอบคลุมที่มาที่ไปของคดี)

**ข้อ 2. มูลเหตุฟ้อง**
(พฤติการณ์ของจำเลย {defendant} โดยละเอียด + ความเสียหายที่เกิดต่อ {plaintiff})

**ข้อ 3. กฎหมายที่อ้างอิง**
(มาตราที่อ้างอิงพร้อมเหตุผลประกอบ)

**ข้อ 4. คำขอท้ายฟ้อง**
ลำดับเลขครบทุกคำขอ

ลงท้าย: "⚠ ร่างคำฟ้องนี้ผ่าน AI ต้องให้ทนายความตรวจและปรับสำนวนก่อนยื่นศาล"

ตอบยาวพอที่จะใช้งานได้จริง ใช้ชื่อจริงของคู่ความตลอดทั้งเอกสาร ห้ามใช้ placeholder
"""
    elif doc_type == "defense":
        prompt = f"""คุณคือทนายความไทย ร่าง **คำให้การจำเลย** ให้ครบถ้วน ใช้ในศาลได้

ข้อมูลคดี:
{case_block}

ใช้ชื่อจริงของคู่ความ:
- โจทก์: {plaintiff}
- จำเลย: {defendant}

ร่างคำให้การที่มีโครงสร้าง:
**คำให้การจำเลย**

ข้าพเจ้า {defendant} จำเลย ขอให้การต่อสู้คดีดังนี้

**ข้อ 1. ปฏิเสธข้ออ้างของ {plaintiff}**
(เขียนเหตุผลปฏิเสธโดยละเอียด)

**ข้อ 2. ข้อต่อสู้หลัก**
(แสดงข้อต่อสู้ครบทุกประเด็น เช่น ขาดอายุความ / ไม่มีอำนาจฟ้อง / ชำระแล้ว / ความชอบด้วยกฎหมาย)

**ข้อ 3. คำขอท้ายคำให้การ**
(ขอให้ยกฟ้อง + เรียกร้องค่าฤชาธรรมเนียม)

ใช้ชื่อจริงตลอด ห้ามใช้ placeholder
ลงท้าย: "⚠ ร่างนี้ต้องให้ทนายความตรวจและปรับสำนวนก่อนยื่นศาล"
"""
    elif doc_type == "contract":
        prompt = f"""คุณคือทนายความไทย ตรวจสอบ **ความเสี่ยงสัญญา** ในคดี

ข้อมูลคดี:
{case_block}
คู่ความ: {plaintiff} (โจทก์) vs {defendant} (จำเลย)

ออกรายงาน:
**รายงานตรวจสัญญา / ความเสี่ยง**

**🔴 Clause ที่ต้องระวัง (อย่างน้อย 4-5 จุด)**

**🟡 ความเสี่ยงโมฆะ/บกพร่อง (อย่างน้อย 3 จุด)**

**🟢 จุดแข็งที่ต้องรักษา**

**📋 ข้อแนะนำในการเจรจา/แก้ไข (อย่างน้อย 5 ข้อ)**

ใช้ชื่อจริงคู่สัญญาตลอด
ลงท้าย: "⚠ ต้องผ่านการพิจารณาจากทนายความก่อนนำไปใช้"
"""
    else:
        return f"ประเภทเอกสารไม่รองรับ: {doc_type}"

    raw = await _claude_call(prompt, model, max_tokens=4000)
    polished = await polish_thai_legal(raw)
    return polished
