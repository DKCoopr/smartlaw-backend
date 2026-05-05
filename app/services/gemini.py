"""
Gemini extraction service — Step 1 of the AI pipeline
Reads the raw Thai transcript and extracts structured fields.
Fast, cheap, high context window — perfect for initial extraction.
"""
import json
import google.generativeai as genai
from app.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.google_api_key)

model = genai.GenerativeModel("gemini-1.5-flash")   # Fast + cheap for extraction


EXTRACTION_PROMPT = """
คุณคือผู้ช่วย AI ของระบบสมาร์ทลอว์ ทำหน้าที่สกัดข้อมูลจากคำบอกเล่าของผู้แจ้งความ

จากข้อความต่อไปนี้ กรุณาสกัดข้อมูลในรูปแบบ JSON ที่มีโครงสร้างดังนี้:

{
  "name": "ชื่อ-นามสกุลผู้แจ้ง",
  "age": "อายุ (ตัวเลขเท่านั้น)",
  "id_card": "เลขบัตรประจำตัวประชาชน 13 หลัก",
  "occupation": "อาชีพ",
  "address": "ที่อยู่",
  "phone": "เบอร์โทรศัพท์",
  "incident_date": "วันที่เกิดเหตุ (พ.ศ.)",
  "incident_time": "เวลาเกิดเหตุ",
  "location": "สถานที่เกิดเหตุ",
  "body": "พฤติการณ์แห่งคดีโดยละเอียด",
  "charge": "ข้อหาที่เกิดขึ้น",
  "intent": "prosecute หรือ mediate หรือ none",
  "suspect_name": "ชื่อผู้ต้องสงสัย (ถ้ามี)",
  "witness": "พยาน (ถ้ามี)",
  "evidence": "หลักฐาน (ถ้ามี)"
}

หากไม่มีข้อมูลส่วนใด ให้ใส่ค่าว่าง "" แทน
ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น

ข้อความ:
{transcript}
"""


async def extract_from_transcript(transcript: str) -> dict:
    """
    Use Gemini Flash to extract structured data from Thai transcript.
    Returns a dict of extracted fields.
    """
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)

    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,         # Low temp = more deterministic extraction
            response_mime_type="application/json",
        ),
    )

    try:
        extracted = json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback: return empty structure if Gemini returns bad JSON
        extracted = {}

    return extracted
