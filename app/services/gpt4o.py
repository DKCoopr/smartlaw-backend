"""
GPT-4o writing service — Step 3 of the AI pipeline
Takes the extracted data and Claude's analysis, writes the formal Thai
police report body in official บันทึกประจำวัน language.
"""
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

FORMAL_WRITING_PROMPT = """
คุณคือผู้เชี่ยวชาญด้านการเขียนเอกสารราชการไทย โดยเฉพาะบันทึกประจำวันของสำนักงานตำรวจแห่งชาติ

จากข้อมูลต่อไปนี้ กรุณาเขียนพฤติการณ์แห่งคดีในรูปแบบราชการไทยที่ถูกต้อง:

== ข้อมูล ==
ผู้แจ้ง: {name} อายุ {age} ปี อาชีพ {occupation}
สถานที่เกิดเหตุ: {location}
วันเวลา: {incident_date} เวลา {incident_time} น.
สาระสำคัญ: {body}
ข้อหา: {charge}

== รูปแบบที่ต้องการ ==
- ใช้ภาษาราชการไทยที่ถูกต้อง
- ขึ้นต้นด้วย "ผู้แจ้งให้การว่า..."
- เขียนเป็นร้อยแก้วต่อเนื่อง ไม่ใช้ข้อๆ
- ใช้คำว่า "ผู้แจ้ง" แทนชื่อ
- ระบุสถานที่ วันเวลา และพฤติการณ์ให้ครบถ้วน
- ปิดท้ายด้วยความประสงค์ของผู้แจ้ง
- ความยาวประมาณ 3-5 ย่อหน้า

ตอบเฉพาะเนื้อหาพฤติการณ์แห่งคดีเท่านั้น ห้ามมีคำอธิบายเพิ่มเติม
"""


async def write_formal_report(extracted: dict) -> str:
    """
    Use GPT-4o to write the formal Thai police report body.
    Returns the formal Thai text string.
    """
    prompt = FORMAL_WRITING_PROMPT.format(
        name=extracted.get("name", "ผู้แจ้ง"),
        age=extracted.get("age", ""),
        occupation=extracted.get("occupation", ""),
        location=extracted.get("location", ""),
        incident_date=extracted.get("incident_date", ""),
        incident_time=extracted.get("incident_time", ""),
        body=extracted.get("body", ""),
        charge=extracted.get("charge", ""),
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "คุณเป็นผู้เชี่ยวชาญด้านเอกสารราชการไทย เขียนได้ถูกต้องตามรูปแบบบันทึกประจำวันตำรวจ",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
