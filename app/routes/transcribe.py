"""
POST /api/transcribe
Accepts: audio file upload OR base64 audio
Returns: Thai transcript text
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from app.models.case import TranscribeRequest, TranscribeResponse
from app.services.whisper import transcribe_audio, transcribe_base64
from app.services.speaker_split import split_speakers
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["transcribe"])
settings = get_settings()


@router.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    diarize: bool = Query(True, description="Tag each segment with officer/complainant speaker label"),
):
    """
    Upload audio file directly (from browser MediaRecorder).
    Accepts: webm, mp4, wav, mp3, m4a
    Max size: configured via MAX_AUDIO_MB

    With diarize=True (default), each Whisper segment is tagged as
    [ตำรวจ] or [ผู้ร้องทุกข์] so the form-extraction pipeline can
    use only the complainant's narration.
    """
    # Validate file type
    allowed_types = ["audio/webm", "audio/mp4", "audio/wav", "audio/mpeg", "audio/m4a", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {file.content_type}. Use webm, mp4, wav, or mp3.",
        )

    # Validate file size
    audio_bytes = await file.read()
    max_bytes = settings.max_audio_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum {settings.max_audio_mb}MB.",
        )

    try:
        result = await transcribe_audio(audio_bytes, file.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Speaker diarization (best-effort; never fail the request on tagging error)
    if diarize and result.get("segments"):
        try:
            split = await split_speakers(result["segments"])
            result["turns"] = split["turns"]
            result["tagged_transcript"] = split["tagged_transcript"]
            result["complainant_text"] = split["complainant_text"]
        except Exception as e:
            # Log but don't fail — return raw transcript
            print(f"[transcribe] speaker_split failed: {e}")

    # Strip internal-only fields before returning
    result.pop("segments", None)
    return TranscribeResponse(**result)


@router.post("/transcribe/base64", response_model=TranscribeResponse)
async def transcribe_base64_endpoint(request: TranscribeRequest):
    """
    Accept base64-encoded audio (for small recordings sent inline from frontend).
    """
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 field is required")

    try:
        result = await transcribe_base64(request.audio_base64)
        return TranscribeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
