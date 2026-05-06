"""
Whisper transcription service
Sends audio to OpenAI Whisper Large v3 optimised for Thai language.
"""
import base64
import tempfile
import os
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """
    Transcribe audio bytes using Whisper Large v3.
    Returns: { transcript, duration_seconds, language_detected }
    """
    # Write to temp file (Whisper needs a file object)
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",          # whisper-1 = Whisper Large v3 via API
                file=audio_file,
                language="th",              # Force Thai — faster and more accurate
                response_format="verbose_json",   # gives us duration + language + segments
                prompt="บันทึกประจำวันตำรวจ แจ้งความ ผู้เสียหาย ผู้ต้องหา",  # Thai legal context hint
            )

        # Pull segments — each is one Whisper-detected utterance with timing.
        # Used downstream for speaker diarization.
        raw_segments = getattr(response, "segments", []) or []
        segments = []
        for seg in raw_segments:
            # Library returns either pydantic model or dict depending on SDK version
            get = (lambda k: getattr(seg, k, None)) if hasattr(seg, "text") else (lambda k: seg.get(k))
            text = (get("text") or "").strip()
            if not text:
                continue
            segments.append({
                "id":    get("id"),
                "start": float(get("start") or 0.0),
                "end":   float(get("end") or 0.0),
                "text":  text,
            })

        return {
            "transcript":        response.text,
            "duration_seconds":  getattr(response, "duration", 0.0),
            "language_detected": getattr(response, "language", "th"),
            "segments":          segments,
        }
    finally:
        os.unlink(tmp_path)   # Always clean up temp file


async def transcribe_base64(audio_base64: str, filename: str = "audio.webm") -> dict:
    """Convenience wrapper for base64-encoded audio from frontend"""
    audio_bytes = base64.b64decode(audio_base64)
    return await transcribe_audio(audio_bytes, filename)
