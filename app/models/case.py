from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, date
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
    audio_base64: Optional[str] = None
    language: str = "th"


class SpeakerTurn(BaseModel):
    speaker: str           # "officer" | "complainant"
    text: str
    start: float
    end: float


class TranscribeResponse(BaseModel):
    transcript: str
    duration_seconds: float
    language_detected: str
    # Diarization (when available — empty list / empty string for very short audio)
    turns: list[SpeakerTurn] = []
    tagged_transcript: str = ""
    complainant_text: str = ""


# ── Input: run the full AI pipeline on a transcript ──────────────────────────

class AnalyzeRequest(BaseModel):
    transcript: str
    station: str = "สมาร์ทลอว์"


# ── The form data structure (mirrors frontend fd state) ──────────────────────

class FormData(BaseModel):
    name: str = ""
    age: str = ""
    id_card: str = ""
    occupation: str = ""
    address: str = ""
    phone: str = ""
    station: str = "สมาร์ทลอว์"

    incident_date: str = ""
    incident_time: str = ""
    location: str = ""

    body: str = ""
    charge: str = ""
    intent: IntentEnum = IntentEnum.prosecute

    legal: str = ""
    steps: List[str] = []
    risk: Optional[RiskLevel] = None


class AnalyzeResponse(BaseModel):
    form_data: FormData
    raw_extraction: dict
    legal_analysis: str
    formal_body: str
    verification_passed: bool
    confidence_score: float
    processing_ms: int


# ── Case persistence ─────────────────────────────────────────────────────────

class CaseCreate(BaseModel):
    title: str
    case_type: Optional[str] = "แพ่ง"
    court: Optional[str] = ""
    plaintiff_name: Optional[str] = ""
    defendant_name: Optional[str] = ""
    our_client: Optional[str] = "plaintiff"
    claim_amount: Optional[float] = 0
    assigned_lawyer: Optional[str] = ""
    next_hearing: Optional[str] = None
    status: Optional[str] = "active"
    transcript: Optional[str] = ""
    form_data: Optional[FormData] = None
    analysis: Optional[Any] = None
    ai_strength_score: Optional[int] = None


class CaseOut(BaseModel):
    id: str
    user_id: str
    title: str
    status: str
    case_number: Optional[str] = None
    case_type: Optional[str] = None
    court: Optional[str] = None
    plaintiff_name: Optional[str] = None
    defendant_name: Optional[str] = None
    our_client: Optional[str] = None
    claim_amount: Optional[float] = 0
    ai_strength_score: Optional[int] = None
    assigned_lawyer: Optional[str] = None
    next_hearing: Optional[date] = None
    created_at: datetime
    updated_at: datetime
    form_data: Optional[Any] = None
    transcript: Optional[str] = None
    analysis: Optional[Any] = None


class CaseListOut(BaseModel):
    cases: List[CaseOut]
    total: int


# ── Document generation ──────────────────────────────────────────────────────

class DocumentRequest(BaseModel):
    case_id: str
    form_data: FormData
    document_type: str = "police_daily_log"
