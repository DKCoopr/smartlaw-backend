"""
POST /api/analyze
Accepts: Thai transcript text
Returns: Fully populated FormData + legal analysis
Runs the full Gemini → Claude → GPT-4o pipeline
"""
from fastapi import APIRouter, HTTPException
from app.models.case import AnalyzeRequest, AnalyzeResponse
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_transcript(request: AnalyzeRequest):
    """
    Full AI pipeline endpoint.
    1. Gemini extracts structured fields
    2. Claude performs legal analysis (parallel with step 3)
    3. GPT-4o writes formal police report body (parallel with step 2)
    4. Claude verifies the analysis

    Typical latency: 8–15 seconds
    """
    if not request.transcript or len(request.transcript.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Transcript is too short. Please provide a full statement.",
        )

    try:
        result = await run_pipeline(
            transcript=request.transcript,
            station=request.station,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {str(e)}")


@router.post("/transcribe-and-analyze")
async def transcribe_and_analyze(
    transcript: str,
    station: str = "สมาร์ทลอว์",
):
    """
    Convenience endpoint: runs pipeline directly on a transcript.
    Frontend can call this after getting the transcript from /transcribe.
    """
    try:
        result = await run_pipeline(transcript=transcript, station=station)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
