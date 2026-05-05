"""Document AI: extract summary from PDF/image/DOCX using Gemini."""
import google.generativeai as genai
from app.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.google_api_key)

_model = genai.GenerativeModel("gemini-1.5-flash")

SUMMARY_PROMPT = """คุณคือผู้ช่วยกฎหมายไทย กรุณาวิเคราะห์เอกสารนี้แล้วตอบเป็นภาษาไทย:

1. สรุปเนื้อหาเอกสารโดยรวม (1-2 ประโยค)
2. ระบุประเภทเอกสาร (เช่น สัญญา / คำฟ้อง / หนังสือเลิกจ้าง / ใบเสร็จ / พินัยกรรม ฯลฯ)
3. ชี้จุดสำคัญ/ความเสี่ยงที่ทนายควรตรวจสอบ (ถ้ามี)

ตอบให้กระชับ ไม่เกิน 4 บรรทัด"""

# Mime types Gemini can read inline
SUPPORTED_INLINE_MIMES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "text/plain",
}

MAX_INLINE_BYTES = 18 * 1024 * 1024   # 18MB safety margin under Gemini 20MB limit


async def summarize_document(file_bytes: bytes, mime_type: str) -> str:
    """
    Returns a Thai-language summary of the document, or "" if unsupported/too large.
    Errors are swallowed and return "" — summary is best-effort.
    """
    if mime_type not in SUPPORTED_INLINE_MIMES:
        return ""
    if len(file_bytes) > MAX_INLINE_BYTES:
        return ""
    try:
        response = await _model.generate_content_async([
            {"mime_type": mime_type, "data": file_bytes},
            SUMMARY_PROMPT,
        ])
        return (response.text or "").strip()
    except Exception as e:
        return f""
