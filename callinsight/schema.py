"""Pydantic v2 schemas for Call Center Insights extraction and reporting."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

SentimentValue = Literal["satisfied", "neutral", "upset", "angry"]
ResolutionValue = Literal["resolved", "unresolved", "unclear"]


class EvidenceItem(BaseModel):
    """An evidence quote with its timestamp."""
    quote: str = Field(description="Exact quote from the transcript supporting the finding.")
    timestamp: Optional[float] = Field(default=None, description="Start timestamp in seconds of the quoted segment.")


class CallExtractionSchema(BaseModel):
    """Structured extraction output required from LLM or baseline provider."""
    customer_sentiment_start: SentimentValue = Field(
        description="Customer sentiment at the beginning of the call."
    )
    customer_sentiment_end: SentimentValue = Field(
        description="Customer sentiment at the end of the call."
    )
    agent_handled_well: bool = Field(
        description="Whether the representative handled the customer and situation well."
    )
    resolution: ResolutionValue = Field(
        description="Whether the customer's issue was resolved, unresolved, or unclear."
    )
    reason: str = Field(
        description="One sentence summarizing the reasoning behind the extraction."
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="List of quotes and timestamps from the transcript supporting the conclusions."
    )


class Segment(BaseModel):
    """A single transcript segment."""
    start: float
    end: float
    speaker: Literal["agent", "customer", "unknown", "inferred"]
    text: str
    confidence: Optional[float] = 1.0


class CallAnalysisResult(BaseModel):
    """Full analysis result for a single call recording."""
    call_id: str
    audio_file: Optional[str] = None
    duration_seconds: float
    resolution_time_minutes: Optional[float] = None
    grounded: bool = True
    ungrounded_quotes: List[str] = Field(default_factory=list)
    extraction: CallExtractionSchema
    segments: List[Segment]
    is_stereo: bool = False
    speaker_diarization_confidence: float = 1.0
    prosody_stats: Optional[dict] = None
    llm_latency_seconds: Optional[float] = None
    llm_tokens: Optional[dict] = None
    provider: str = "baseline"
