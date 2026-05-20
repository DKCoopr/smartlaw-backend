"""
Thai.Law AI Pipeline Orchestrator
──────────────────────────────────────────────────────────────────
Pipeline stages:
  1. Gemini Flash  → extract structured fields from Thai transcript
  2. Claude Sonnet → deep legal analysis + verification pass
  3. GPT-4o        → write formal police report body

Parallel architecture:
  - Stage 2 (Claude) and Stage 3 (GPT-4o) run concurrently
  - Claude also runs a self-verification pass after main analysis
  - Total latency ≈ max(Claude, GPT-4o) instead of Claude + GPT-4o
──────────────────────────────────────────────────────────────────
"""
import asyncio
import time
from app.services.gemini import extract_from_transcript
from app.services.claude import analyze_case, verify_analysis
from app.services.gpt4o import write_formal_report
from app.models.case import FormData, AnalyzeResponse


async def run_pipeline(transcript: str, station: str = "สมาร์ทลอว์") -> AnalyzeResponse:
    """
    Full AI pipeline: transcript → structured FormData + legal analysis.

    Flow:
      [Gemini extraction]
             ↓
      [Claude analysis] ──parallel── [GPT-4o formal writing]
             ↓
      [Claude verification]
             ↓
      [Merge into FormData]
    """
    start_ms = int(time.time() * 1000)

    # ── STAGE 1: Gemini extraction ────────────────────────────────────────────
    extracted = await extract_from_transcript(transcript)

    # ── STAGE 2+3: Claude analysis + GPT-4o writing in PARALLEL ─────────────
    case_summary = _build_case_summary(extracted, transcript)

    claude_task = asyncio.create_task(analyze_case(case_summary))
    gpt4o_task = asyncio.create_task(write_formal_report(extracted))

    claude_result, formal_body = await asyncio.gather(claude_task, gpt4o_task)

    # ── STAGE 4: Verification pass (Claude checks itself) ────────────────────
    verification = await verify_analysis(claude_result["legal_text"])

    # If verification found issues and confidence is low, re-run Claude once
    if not verification.get("is_correct") and verification.get("confidence", 1.0) < 0.6:
        corrections = verification.get("corrections", "")
        if corrections:
            retry_prompt = f"{case_summary}\n\n== หมายเหตุ: พบข้อผิดพลาดดังนี้ ==\n{corrections}\nกรุณาวิเคราะห์ใหม่โดยแก้ไขข้อผิดพลาดดังกล่าว"
            claude_result = await analyze_case(retry_prompt)

    # ── Merge results into FormData ───────────────────────────────────────────
    form_data = _merge_to_form_data(extracted, claude_result, formal_body, station)

    end_ms = int(time.time() * 1000)

    return AnalyzeResponse(
        form_data=form_data,
        raw_extraction=extracted,
        legal_analysis=claude_result["legal_text"],
        formal_body=formal_body,
        verification_passed=verification.get("is_correct", True),
        confidence_score=verification.get("confidence", 0.85),
        processing_ms=end_ms - start_ms,
    )


def _build_case_summary(extracted: dict, transcript: str) -> str:
    """Build a concise case summary for Claude from extracted fields"""
    return f"""
ผู้แจ้ง: {extracted.get('name', 'ไม่ระบุ')} อายุ {extracted.get('age', '')} ปี
อาชีพ: {extracted.get('occupation', '')}
สถานที่เกิดเหตุ: {extracted.get('location', '')}
วันเวลา: {extracted.get('incident_date', '')} เวลา {extracted.get('incident_time', '')} น.
พฤติการณ์: {extracted.get('body', '')}
ข้อหาที่กล่าวหา: {extracted.get('charge', '')}
ความประสงค์: {extracted.get('intent', 'prosecute')}
ผู้ต้องสงสัย: {extracted.get('suspect_name', 'ไม่ระบุ')}
หลักฐาน: {extracted.get('evidence', 'ไม่มี')}

คำบอกเล่าต้นฉบับ: {transcript[:500]}...
""".strip()


def _merge_to_form_data(
    extracted: dict,
    claude_result: dict,
    formal_body: str,
    station: str,
) -> FormData:
    """Combine all pipeline outputs into the FormData structure"""

    # Use GPT-4o's formal writing as the body if available, else extracted body
    body = formal_body if formal_body else extracted.get("body", "")

    # Map intent string to enum value
    intent_map = {"prosecute": "prosecute", "mediate": "mediate", "none": "none"}
    raw_intent = extracted.get("intent", "prosecute").lower()
    intent = intent_map.get(raw_intent, "prosecute")

    return FormData(
        name=extracted.get("name", ""),
        age=extracted.get("age", ""),
        id_card=extracted.get("id_card", ""),
        occupation=extracted.get("occupation", ""),
        address=extracted.get("address", ""),
        phone=extracted.get("phone", ""),
        station=station,
        incident_date=extracted.get("incident_date", ""),
        incident_time=extracted.get("incident_time", ""),
        location=extracted.get("location", ""),
        body=body,
        charge=extracted.get("charge", ""),
        intent=intent,
        legal=claude_result.get("legal_text", ""),
        steps=claude_result.get("steps", []),
        risk=claude_result.get("risk", "low"),
    )
