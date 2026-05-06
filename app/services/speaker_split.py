"""
Speaker diarization via GPT-4o
──────────────────────────────────────────────────────────────────
Whisper API does not natively diarize. For Thai police-interview
recordings we use GPT-4o to assign each Whisper segment to one
of two roles based on content cues:

  • officer       — asks structured questions, requests info
  • complainant   — narrates events, gives personal info

The output is:
  { "turns": [ { "speaker": "officer"|"complainant", "text": "...",
                 "start": 0.0, "end": 4.2 }, ... ],
    "complainant_text": "concatenated narration suitable for form extraction",
    "tagged_transcript": "[ตำรวจ] ...\n[ผู้ร้องทุกข์] ..." }
──────────────────────────────────────────────────────────────────
"""
import json
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

# Cap segments sent to GPT to keep costs predictable. Long sessions are
# truncated; first N segments cover most use cases.
_MAX_SEGMENTS = 200


_SYSTEM_PROMPT = """คุณคือผู้ช่วยถอดบันทึกการแจ้งความที่ภาษาไทย
ภารกิจ: ดูช่วงเสียง (segments) แต่ละช่วงจาก Whisper แล้วแยกว่าเป็นเสียงของใคร

มี 2 บทบาทเท่านั้น:
- "officer"     = เจ้าหน้าที่ตำรวจ / ผู้สอบปากคำ — มักถามคำถาม สอบยืนยันข้อมูล (ชื่อ อายุ ที่อยู่) ขอ ID
- "complainant" = ผู้ร้องทุกข์ / ผู้แจ้ง — เล่าเรื่องราว บรรยายเหตุการณ์ ตอบคำถาม

หลักการตัดสินใจ:
1. ใครถาม → officer  (เช่น "คุณชื่ออะไร", "เกิดเหตุที่ไหน", "เห็นหน้าผู้ก่อเหตุไหม")
2. ใครเล่าเหตุการณ์/ตอบคำถามยาว → complainant
3. ถ้าเป็นการบันทึกเดี่ยว (ผู้แจ้งบันทึกเอง) → ทุก segment เป็น complainant
4. ถ้าไม่ชัด → เลือก complainant เป็น default

ตอบเป็น JSON เท่านั้น ในรูปแบบ:
{
  "turns": [
    {"id": <segment id>, "speaker": "officer" | "complainant"}
  ]
}
ตอบครบทุก segment id ที่ส่งไป — ห้ามขาด"""


async def split_speakers(segments: list[dict]) -> dict:
    """
    Tag each Whisper segment with a speaker role.

    Returns a dict containing:
      - turns:               [{speaker, text, start, end, id}]
      - complainant_text:    concatenated complainant narration
      - tagged_transcript:   pretty Thai-labeled transcript for display
    """
    if not segments:
        return {"turns": [], "complainant_text": "", "tagged_transcript": ""}

    # Single-utterance recordings or very short statements: skip GPT call
    if len(segments) <= 2:
        turns = [
            {**s, "speaker": "complainant"}
            for s in segments
        ]
        return _assemble(turns)

    work = segments[:_MAX_SEGMENTS]
    user_payload = json.dumps(
        [{"id": s.get("id"), "text": s["text"]} for s in work],
        ensure_ascii=False,
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",         # diarization is a content-classification task; mini is plenty
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Segments:\n{user_payload}"},
        ],
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        labelled = {t["id"]: t["speaker"] for t in parsed.get("turns", []) if "id" in t}
    except (json.JSONDecodeError, KeyError, TypeError):
        labelled = {}

    turns = []
    for s in segments:
        speaker = labelled.get(s.get("id")) or "complainant"
        if speaker not in ("officer", "complainant"):
            speaker = "complainant"
        turns.append({**s, "speaker": speaker})

    return _assemble(turns)


def _assemble(turns: list[dict]) -> dict:
    """Build complainant_text and tagged_transcript from labelled turns."""
    complainant_chunks = [t["text"] for t in turns if t["speaker"] == "complainant"]
    complainant_text = " ".join(complainant_chunks).strip()

    label_th = {"officer": "[ตำรวจ]", "complainant": "[ผู้ร้องทุกข์]"}
    pretty_lines = []
    last_speaker = None
    buffer = []
    for t in turns:
        if t["speaker"] != last_speaker and last_speaker is not None:
            pretty_lines.append(f"{label_th[last_speaker]} {' '.join(buffer)}")
            buffer = []
        buffer.append(t["text"])
        last_speaker = t["speaker"]
    if buffer and last_speaker is not None:
        pretty_lines.append(f"{label_th[last_speaker]} {' '.join(buffer)}")

    return {
        "turns":             turns,
        "complainant_text":  complainant_text,
        "tagged_transcript": "\n".join(pretty_lines),
    }
