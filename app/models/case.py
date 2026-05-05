from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class IntentEnum(str, Enum):
    prosecute = "prosecute"
    mediate = "mediate"
    none = "none"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ── Input: raw transcript from Whisper ──────────────────────────────────────

class TranscribeRequest(BaseModel):
    """Sent from frontend after recording stops"""
    audio_base64: Optional[str] = None   # for small files sent inline
    language: str = "th"                  # Thai by default


class TranscribeResponse(BaseModel):
    transcript: str
    duration_seconds: float
    language_detected: str


# ── Input: run the full AI pipeline on a transcript ──────────────────────────

class AnalyzeRequest(BaseModel):
    transcript: str
    station: str = "สมาร์ทลอว์"


# ── The form data structure (mirrors frontend fd state) ──────────────────────

class FormData(BaseModel):
    # Reporter info
    name: str = ""
    age: str = ""
    id_card: str = ""
    occupation: str = ""
    address: str = ""
    phone: str = ""
    station: str = "สมาร์ทลอว์"

    # Incident
    incident_date: str = ""
    incident_time: str = ""
    location: str = ""

    # Details
    body: str = ""
    charge: str = ""
    intent: IntentEnum = IntentEnum.prosecute

    # AI analysis (filled by pipeline)
    legal: str = ""
    steps: List[str] = []
    risk: Optional[RiskLevel] = None


class AnalyzeResponse(BaseModel):
    form_data: FormData
    raw_extraction: dict        # Gemini output
    legal_analysis: str         # Claude output
    formal_body: str            # GPT-4o output
    verification_passed: bool
    confidence_score: float     # 0.0 – 1.0
    processing_ms: int


# ── Case persistence ─────────────────────────────────────────────────────────

class CaseCreate(BaseModel):
    title: str
    transcript: str
    form_data: FormData
    analysis: Optional[AnalyzeResponse] = None


class CaseOut(BaseModel):
    id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    form_data: FormData
    user_id: str


class CaseListOut(BaseModel):
    cases: List[CaseOut]
    total: int


# ── Document generation ──────────────────────────────────────────────────────

class DocumentRequest(BaseModel):
    case_id: str
    form_data: FormData
    document_type: str = "police_daily_log"   # police_daily_log | complaint | power_of_attorney
