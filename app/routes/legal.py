"""Legal analysis routes — Claude-powered case analysis + draft documents."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import anthropic
from app.auth import get_current_user_id
from app.config import get_settings

router = APIRouter(prefix="/api/legal", tags=["legal"])

settings = get_settings()
_claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


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


class AnalyzeIn(BaseModel):
    case: CaseInput
    perspective: str = "both"          # plaintiff | defendant | both
    document_types: List[str] = []     # complaint | defense | contract


def _case_block(case: CaseInput) -> str:
    return f"""ชื่อคดี: {case.title or "—"}
ประเภทคดี: {case.case_type or "—"}
ศาล: {case.court or "—"}
โจทก์: {case.plaintiff_name or "—"}
จำเลย: {case.defendant_name or "—"}
ลูกความของเรา: {"ฝ่ายโจทก์" if case.our_client == "plaintiff" else "ฝ่ายจำเลย"}
ทุนทรัพย์: {case.claim_amount or 0:,.0f} บาท
รายละเอียด/คำบอกเล่า: {case.transcript or case.notes or "—"}"""


def _perspective_label(p: str) -> str:
    return {"plaintiff": "ฝ่ายโจทก์", "defendant": "ฝ่ายจำเลย"}.get(p, "ทั้งสองฝ่าย")


async def _claude_call(prompt: str, max_tokens: int = 1500) -> str:
    msg = await _claude.messages.create(
        model="claude-sonnet-4-6",
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

    main_prompt = f"""คุณคือทนายความไทยผู้เชี่ยวชาญ วิเคราะห์คดีต่อไปนี้จากมุมมอง{p_label}

== ข้อมูลคดี ==
{case_block}

ตอบในหัวข้อต่อไปนี้ ด้วยภาษาไทย กระชับและตรงประเด็น:

🔍 **วิเคราะห์คดีภาพรวม — {p_label}**

**ข้อเท็จจริงสำคัญ** (3-4 ข้อ)

**กฎหมายที่เกี่ยวข้อง**
- อ้างอิงมาตรา/พรบ. ที่เกี่ยวข้อง 3-5 ข้อ พร้อมเลขมาตรา

**โอกาสชนะคดี**
- ประเมินเป็นเปอร์เซ็นต์: โจทก์ X% / จำเลย Y%

**จุดแข็งของคดี**
- 3-4 ข้อ

**จุดอ่อน / ความเสี่ยง**
- 2-3 ข้อ

**กลยุทธ์แนะนำ**
1. ระยะสั้น (0-30 วัน):
2. ระยะกลาง (30-90 วัน):
3. ระยะยาว:

**ประเมินความเสี่ยง**
- สูง 🔴: ...
- กลาง 🟡: ...
- ต่ำ 🟢: ...

**Timeline สำคัญ**
- (ระบุช่วงเวลา + เหตุการณ์ที่คาดว่าจะเกิด)

ลงท้ายด้วยข้อความ: "⚠️ ผลการวิเคราะห์นี้เป็นการประมาณการเบื้องต้น ต้องผ่านการพิจารณาจากทนายความก่อนนำไปใช้"
"""

    try:
        analysis_text = await _claude_call(main_prompt, max_tokens=2200)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude analysis failed: {str(e)}")

    documents: dict[str, str] = {}
    for doc_type in payload.document_types or []:
        try:
            documents[doc_type] = await _draft_document(doc_type, payload.case)
        except Exception as e:
            documents[doc_type] = f"⚠️ ไม่สามารถร่างเอกสารได้: {str(e)}"

    return {
        "analysis": analysis_text,
        "documents": documents,
    }


async def _draft_document(doc_type: str, case: CaseInput) -> str:
    case_block = _case_block(case)

    if doc_type == "complaint":
        prompt = f"""คุณคือทนายความไทย ร่าง **คำฟ้อง** สำหรับคดีต่อไปนี้

{case_block}

ร่างคำฟ้องในรูปแบบที่ใช้ในศาลไทย ประกอบด้วย:
**คำฟ้อง**

ข้าพเจ้า [โจทก์] ขอฟ้อง [จำเลย] ต่อ[ศาล]

**ข้อ 1. ข้อเท็จจริง**
(ระบุข้อเท็จจริงพื้นฐาน 3-5 บรรทัด)

**ข้อ 2. มูลเหตุฟ้อง**
(พฤติการณ์ของจำเลยและความเสียหาย)

**ข้อ 3. คำขอท้ายฟ้อง**
1. (ขอให้จำเลย...)
2. (ขอให้จำเลย...)

ลงท้ายด้วย: "⚠ ร่างนี้เป็นตัวอย่าง ต้องให้ทนายความตรวจสอบก่อนยื่นศาล"
"""
    elif doc_type == "defense":
        prompt = f"""คุณคือทนายความไทย ร่าง **คำให้การจำเลย** สำหรับคดีต่อไปนี้

{case_block}

ร่างคำให้การประกอบด้วย:
**คำให้การจำเลย**

ข้าพเจ้า [จำเลย] ขอให้การต่อสู้คดีดังนี้

**ข้อ 1. ปฏิเสธข้ออ้างโจทก์**

**ข้อ 2. ข้อต่อสู้**
(เช่น โจทก์ไม่มีอำนาจฟ้อง / ขาดอายุความ / ชำระแล้ว)

**ข้อ 3. คำขอท้ายคำให้การ**

ลงท้ายด้วย: "⚠ ร่างนี้เป็นตัวอย่าง ต้องให้ทนายความตรวจสอบก่อนยื่นศาล"
"""
    elif doc_type == "contract":
        prompt = f"""คุณคือทนายความไทย ตรวจสอบ **ความเสี่ยงสัญญา** ในคดีต่อไปนี้

{case_block}

ออกรายงานในรูปแบบ:
**รายงานตรวจสัญญา / ความเสี่ยง**

**Clause ที่ต้องระวัง 🔴**
- ข้อ X — ...

**ความเสี่ยงโมฆะ 🟡**
- ข้อ Y — ...

**ข้อแนะนำ**
1.
2.
3.

ลงท้ายด้วย: "⚠ ต้องผ่านการพิจารณาจากทนายความก่อนนำไปใช้"
"""
    else:
        return f"ประเภทเอกสารไม่รองรับ: {doc_type}"

    return await _claude_call(prompt, max_tokens=1500)
