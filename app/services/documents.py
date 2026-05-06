"""Document AI: extract summary from PDF/image/DOCX using Gemini."""
import google.generativeai as genai
from app.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.google_api_key)

# Pro model handles Thai legal documents more accurately than Flash.
_model = genai.GenerativeModel("gemini-1.5-pro-latest")

SUMMARY_PROMPT = """คุณเป็นทนายความไทยผู้เชี่ยวชาญ อ่านเอกสารนี้ (เป็นภาษาไทย) แล้วเขียนสรุปเชิงกฎหมายอย่างละเอียดในภาษาไทยทางการ

โครงสร้างคำตอบ:

**ประเภทเอกสาร:** (เช่น คำฟ้อง / คำให้การ / สัญญา / หนังสือบอกกล่าว / พินัยกรรม / คำสั่งศาล / รายงานแพทย์ / ใบรับรองแพทย์ / ใบเสร็จ ฯลฯ)

**คู่ความ/ผู้เกี่ยวข้อง:**
- ชื่อ-นามสกุลของบุคคลและนิติบุคคลทุกฝ่ายที่ปรากฏในเอกสาร (เขียนชื่อจริง)
- บทบาทของแต่ละฝ่าย (โจทก์/จำเลย/ผู้ให้/ผู้รับ ฯลฯ)

**สาระสำคัญ:**
- ใจความหลักของเอกสาร 3-5 บรรทัด
- วันที่/เลขเอกสาร/เลขคดีที่ปรากฏ (ถ้ามี)
- จำนวนเงิน/ทรัพย์สินที่ระบุ (ถ้ามี)

**ข้อกฎหมายที่อ้าง:**
- มาตรา / พระราชบัญญัติ / คำพิพากษาฎีกา ที่เอกสารกล่าวถึง

**จุดสำคัญ/ความเสี่ยงทางกฎหมาย:**
- ประเด็นที่ทนายต้องสนใจเป็นพิเศษ (เงื่อนไขเสี่ยง / ข้อสัญญาผิดปกติ / ข้อมูลที่ขาดหาย / ความขัดแย้ง)

ตอบเฉพาะเนื้อหาที่ปรากฏจริงในเอกสาร ห้ามแต่งเติม ถ้าข้อมูลส่วนใดไม่มีให้ใส่ "—" """

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
