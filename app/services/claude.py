"""
Claude analysis service — Step 2 of the AI pipeline
Takes Gemini's extracted data + original transcript and performs
deep Thai legal analysis: applicable laws, recommended steps, risk assessment.
Uses Claude Sonnet 4.6 (fast + accurate for legal reasoning).
"""
import anthropic
from app.config import get_settings

settings = get_settings()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

LEGAL_ANALYSIS_PROMPT = """
คุณคือที่ปรึกษากฎหมายไทยผู้เชี่ยวชาญ ทำงานในระบบสมาร์ทลอว์
วิเคราะห์คดีต่อไปนี้และให้คำแนะนำทางกฎหมายที่ครบถ้วนและถูกต้อง

== ข้อมูลคดี ==
{case_summary}

== คำถามที่ต้องวิเคราะห์ ==

1. **มาตราและกฎหมายที่เกี่ยวข้อง**
   - ระบุมาตรา ชื่อกฎหมาย พระราชบัญญัติ หรือประมวลกฎหมายที่ใช้บังคับ
   - อ้างอิงให้ครบถ้วน เช่น "ประมวลกฎหมายอาญา มาตรา 341"

2. **ขั้นตอนที่แนะนำ** (เรียงลำดับความสำคัญ)
   - สิ่งที่ผู้แจ้งควรทำทันที
   - การรวบรวมหลักฐาน
   - ขั้นตอนทางกฎหมาย

3. **ระดับความเสี่ยงของคดี**
   - ประเมิน: low / medium / high
   - เหตุผล

4. **คำแนะนำเพิ่มเติม**

ตอบเป็นภาษาไทย กระชับ ชัดเจน และถูกต้องตามกฎหมายไทย
"""

VERIFICATION_PROMPT = """
ตรวจสอบความถูกต้องของการวิเคราะห์ทางกฎหมายต่อไปนี้:

== การวิเคราะห์ที่ต้องตรวจสอบ ==
{analysis}

== คำถามตรวจสอบ ==
1. มาตราที่อ้างถึงมีอยู่จริงในกฎหมายไทยหรือไม่?
2. การตีความกฎหมายถูกต้องหรือไม่?
3. มีข้อผิดพลาดหรือข้อมูลที่ขาดหายไปหรือไม่?

ตอบในรูปแบบ JSON:
{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "issues": ["รายการปัญหา (ถ้ามี)"],
  "corrections": "การแก้ไข (ถ้ามี)"
}
"""


async def analyze_case(case_summary: str) -> dict:
    """
    Run Claude legal analysis on the extracted case data.
    Returns { legal_text, steps, risk, raw_response }
    """
    prompt = LEGAL_ANALYSIS_PROMPT.format(case_summary=case_summary)

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    analysis_text = message.content[0].text

    # Parse out steps and risk from the response
    steps = _extract_steps(analysis_text)
    risk = _extract_risk(analysis_text)

    return {
        "legal_text": analysis_text,
        "steps": steps,
        "risk": risk,
        "raw_response": analysis_text,
    }


async def verify_analysis(analysis: str) -> dict:
    """
    Self-verification pass — Claude checks its own output for legal accuracy.
    Returns { is_correct, confidence, issues, corrections }
    """
    prompt = VERIFICATION_PROMPT.format(analysis=analysis)

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0.0,    # Zero temp for verification — deterministic
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    try:
        result = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        result = {"is_correct": True, "confidence": 0.7, "issues": [], "corrections": ""}

    return result


def _extract_steps(text: str) -> list:
    """Extract numbered steps from Claude's response"""
    steps = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        # Match lines starting with numbers or Thai bullet patterns
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            clean = line.lstrip("0123456789.-•) ").strip()
            if len(clean) > 10:   # Skip very short lines
                steps.append(clean)
    return steps[:6]   # Max 6 steps


def _extract_risk(text: str) -> str:
    """Extract risk level from Claude's response"""
    text_lower = text.lower()
    if "high" in text_lower or "สูง" in text:
        return "high"
    elif "medium" in text_lower or "ปานกลาง" in text:
        return "medium"
    return "low"
