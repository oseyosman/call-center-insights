"""LLM structured extraction module with Pydantic v2 validation and grounding check.

Providers:
- baseline (default, offline): VADER sentiment on customer segments + keyword rules.
- ollama: local open-source model via http://localhost:11434 with JSON-schema format.
- anthropic: Claude API via forced tool use.

⚠️  SIMULATED DATA WARNING: Demo and evaluation transcripts are synthetic/simulated.
    No real personal data is ever passed to any provider.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from callinsight.redact import redact_segments
from callinsight.schema import (
    CallAnalysisResult,
    CallExtractionSchema,
    EvidenceItem,
    Segment,
    SentimentValue,
    ResolutionValue,
)
from callinsight.timing import calculate_resolution_time, compute_call_duration, normalize_text_for_search


def verify_grounding(evidence: List[EvidenceItem], transcript_text: str) -> Tuple[bool, List[str]]:
    """Grounding check: verify that every evidence quote appears in the transcript.

    Grounding rules (applied in order):
    1. Normalised direct substring match – the entire normalised quote must appear
       as a substring of the normalised transcript.  Punctuation and case are ignored.
    2. Short quotes (< 4 words) that fail rule 1 are immediately ungrounded.
    3. Longer quotes (≥ 4 words) that fail rule 1 are allowed only if ≥ 80 % of
       their 4-gram windows appear IN ORDER within the transcript. A single matching
       4-gram is NOT enough (fixes the original bug).

    Returns (is_grounded, list_of_ungrounded_quotes).
    """
    if not evidence:
        return True, []

    norm_transcript = normalize_text_for_search(transcript_text)
    ungrounded = []

    for item in evidence:
        norm_quote = normalize_text_for_search(item.quote)
        if not norm_quote:
            continue

        # Rule 1: full normalised substring match
        if norm_quote in norm_transcript:
            continue

        # Rule 2 / 3: fallback for longer quotes
        quote_words = norm_quote.split()
        if len(quote_words) >= 4:
            windows = [" ".join(quote_words[i:i + 4]) for i in range(len(quote_words) - 3)]
            # Ordered search: scan transcript left-to-right
            pos = 0
            matched_windows = 0
            for w in windows:
                idx = norm_transcript.find(w, pos)
                if idx != -1:
                    matched_windows += 1
                    pos = idx  # advance but allow overlaps
            if matched_windows / len(windows) < 0.80:
                ungrounded.append(item.quote)
        else:
            # Short quote failed direct match – ungrounded
            ungrounded.append(item.quote)

    is_grounded = len(ungrounded) == 0
    return is_grounded, ungrounded


class BaseExtractor(ABC):
    """Abstract base class for extraction providers."""

    @abstractmethod
    def extract(self, segments: List[Segment]) -> Tuple[CallExtractionSchema, Dict[str, Any]]:
        """Extract structured insights from transcript segments.

        Returns (CallExtractionSchema, metadata_dict).
        """
        pass


class BaselineExtractor(BaseExtractor):
    """Offline reference baseline using VADER sentiment and simple keyword rules.

    NOTE: The baseline copies quotes verbatim from segments, so its grounding
    rate is trivially 100%.  That number only has meaning for LLM providers
    that must generate quotes from memory.
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def _score_to_sentiment(self, compound: float) -> SentimentValue:
        if compound >= 0.25:
            return "satisfied"
        elif compound > -0.15:
            return "neutral"
        elif compound > -0.5:
            return "upset"
        else:
            return "angry"

    def extract(self, segments: List[Segment]) -> Tuple[CallExtractionSchema, Dict[str, Any]]:
        start_time = time.time()

        if not segments:
            return CallExtractionSchema(
                customer_sentiment_start="neutral",
                customer_sentiment_end="neutral",
                agent_handled_well=True,
                resolution="unclear",
                reason="No audio segments available.",
                evidence=[]
            ), {"provider": "baseline", "latency_seconds": 0.0, "tokens": None}

        # Filter customer segments for sentiment
        customer_segments = [s for s in segments if s.speaker == "customer"]
        all_inferred = not customer_segments

        # Bug 5 fix: when no customer-labeled segments exist, use all segments
        # but annotate the reason clearly.
        if not customer_segments:
            customer_segments = segments

        n = len(customer_segments)
        first_third_idx = max(1, n // 3)
        last_third_idx = max(first_third_idx, 2 * n // 3)

        start_text = " ".join(s.text for s in customer_segments[:first_third_idx])
        end_text = " ".join(s.text for s in customer_segments[last_third_idx:])

        score_start = self.vader.polarity_scores(start_text)["compound"]
        score_end = self.vader.polarity_scores(end_text)["compound"]

        sentiment_start = self._score_to_sentiment(score_start)
        sentiment_end = self._score_to_sentiment(score_end)

        # Keyword heuristics for resolution and agent handling
        full_text_lower = " ".join(s.text.lower() for s in segments)
        last_half = " ".join(s.text.lower() for s in segments[len(segments) // 2:])

        resolved_keywords = [
            "fixed", "resolved", "solved", "working now", "sorted out",
            "refund processed", "account unlocked", "all set", "take care of that for you"
        ]
        unresolved_keywords = [
            "cannot help", "can't help", "unable to assist", "not fixed",
            "still broken", "escalate to a supervisor", "speak to a manager", "file a complaint",
            "nothing i can do", "system is down"
        ]

        resolution: ResolutionValue = "unclear"
        evidence_items: List[EvidenceItem] = []

        if any(kw in last_half for kw in resolved_keywords):
            resolution = "resolved"
        elif any(kw in full_text_lower for kw in unresolved_keywords):
            resolution = "unresolved"
        elif "thank" in last_half:
            resolution = "resolved"

        # Agent handling
        bad_agent_phrases = ["not my problem", "calm down", "hurry up", "i don't care", "whatever"]
        agent_handled_well = not any(phrase in full_text_lower for phrase in bad_agent_phrases)

        # Pick evidence segment
        if resolution == "resolved":
            for seg in reversed(segments):
                seg_lower = seg.text.lower()
                if any(kw in seg_lower for kw in resolved_keywords) or "thank" in seg_lower:
                    evidence_items.append(EvidenceItem(quote=seg.text, timestamp=seg.start))
                    break
        elif resolution == "unresolved":
            for seg in segments:
                seg_lower = seg.text.lower()
                if any(kw in seg_lower for kw in unresolved_keywords):
                    evidence_items.append(EvidenceItem(quote=seg.text, timestamp=seg.start))
                    break

        if not evidence_items and segments:
            # Add at least one representative quote
            evidence_items.append(EvidenceItem(quote=segments[-1].text, timestamp=segments[-1].start))

        reason = (
            f"Baseline rule classified resolution as {resolution} with customer sentiment shifting "
            f"from {sentiment_start} to {sentiment_end}."
        )

        # Bug 5 fix: annotate mono/inferred audio in the reason field
        if all_inferred:
            reason += " (speaker attribution inferred; sentiment computed on all speech)"

        extraction = CallExtractionSchema(
            customer_sentiment_start=sentiment_start,
            customer_sentiment_end=sentiment_end,
            agent_handled_well=agent_handled_well,
            resolution=resolution,
            reason=reason,
            evidence=evidence_items
        )

        latency = time.time() - start_time
        return extraction, {"provider": "baseline", "latency_seconds": round(latency, 4), "tokens": None}


class OllamaExtractor(BaseExtractor):
    """Local LLM extractor via Ollama's HTTP API (http://localhost:11434)."""

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("CALL_OLLAMA_MODEL", "llama3")

    def _build_prompt(self, segments: List[Segment]) -> str:
        lines = []
        for s in segments:
            lines.append(f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker.upper()}: {s.text}")
        transcript = "\n".join(lines)

        return (
            "You are an expert quality assurance analyst for call center operations.\n"
            "Analyze the following call transcript and extract structured insights in exact JSON.\n"
            "Requirements:\n"
            "- customer_sentiment_start & customer_sentiment_end: one of 'satisfied', 'neutral', 'upset', 'angry'.\n"
            "- agent_handled_well: boolean (true if agent remained polite, professional, and helpful; false if rude, dismissive, or hostile).\n"
            "- resolution: 'resolved', 'unresolved', or 'unclear'.\n"
            "- reason: exactly one clear sentence explaining the resolution & agent evaluation.\n"
            "- evidence: list of objects with 'quote' (EXACT string copied verbatim from the transcript) and 'timestamp' (float seconds).\n"
            "CRITICAL: Do NOT invent or paraphrase evidence quotes. Every quote MUST be an exact verbatim substring from the transcript.\n\n"
            f"Transcript:\n{transcript}\n"
        )

    def extract(self, segments: List[Segment]) -> Tuple[CallExtractionSchema, Dict[str, Any]]:
        import requests

        prompt = self._build_prompt(segments)
        schema_json = CallExtractionSchema.model_json_schema()

        url = f"{self.host.rstrip('/')}/api/chat"
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()

        def _call_api(msgs: List[dict]):
            payload = {
                "model": self.model,
                "messages": msgs,
                "format": schema_json,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        try:
            res_data = _call_api(messages)
            raw_content = res_data.get("message", {}).get("content", "{}")
            extraction = CallExtractionSchema.model_validate_json(raw_content)
        except Exception as first_err:
            # Retry once with error message prompt
            messages.append({"role": "assistant", "content": raw_content if "raw_content" in locals() else ""})
            messages.append({
                "role": "user",
                "content": f"The previous output failed validation: {str(first_err)}. Please respond with valid JSON strictly conforming to the schema."
            })
            res_data = _call_api(messages)
            raw_content = res_data.get("message", {}).get("content", "{}")
            extraction = CallExtractionSchema.model_validate_json(raw_content)

        latency = time.time() - start_time
        tokens = {
            "prompt_eval_count": res_data.get("prompt_eval_count"),
            "eval_count": res_data.get("eval_count")
        }

        return extraction, {"provider": f"ollama ({self.model})", "latency_seconds": round(latency, 4), "tokens": tokens}


class AnthropicExtractor(BaseExtractor):
    """Claude API extractor using forced tool use for guaranteed structured JSON."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for AnthropicExtractor.")
        # Bug 7: default changed from stale claude-3-5-sonnet-20241022 to claude-sonnet-5-5
        self.model = model or os.environ.get("CALL_ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def _build_prompt(self, segments: List[Segment]) -> str:
        lines = []
        for s in segments:
            lines.append(f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker.upper()}: {s.text}")
        transcript = "\n".join(lines)

        return (
            "You are an expert QA auditor for customer service call centers.\n"
            "Analyze the transcript and call the record_call_insights tool with your structured analysis.\n"
            "Requirements:\n"
            "- customer_sentiment_start and customer_sentiment_end: 'satisfied', 'neutral', 'upset', or 'angry'.\n"
            "- agent_handled_well: boolean.\n"
            "- resolution: 'resolved', 'unresolved', or 'unclear'.\n"
            "- reason: exactly one sentence.\n"
            "- evidence: list of verbatim quotes from the transcript and start timestamps.\n"
            "CRITICAL: Evidence quotes must be verbatim matches from the transcript.\n\n"
            f"Transcript:\n{transcript}\n"
        )

    def extract(self, segments: List[Segment]) -> Tuple[CallExtractionSchema, Dict[str, Any]]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        tool_schema = {
            "name": "record_call_insights",
            "description": "Record structured call center extraction findings.",
            "input_schema": CallExtractionSchema.model_json_schema()
        }

        prompt = self._build_prompt(segments)
        start_time = time.time()

        def _call_claude(msgs):
            return client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=[tool_schema],
                tool_choice={"type": "auto"},
                messages=msgs
            )

        messages = [{"role": "user", "content": prompt}]
        resp = None
        try:
            resp = _call_claude(messages)
            tool_input = None
            tool_use_block = None
            for block in resp.content:
                if block.type == "tool_use" and block.name == "record_call_insights":
                    tool_input = block.input
                    tool_use_block = block
                    break
            if tool_input is None:
                raise ValueError("Anthropic API did not return tool_use content block.")
            extraction = CallExtractionSchema.model_validate(tool_input)
        except Exception as first_err:
            # Bug 7 fix: build a proper Anthropic multi-turn retry with real content blocks
            # The assistant turn must contain the actual content blocks from the response,
            # and we must add a tool_result for the failed tool_use block.
            if resp is not None and resp.content:
                assistant_content = [b.model_dump() for b in resp.content]
                # Find the tool_use block id for the tool_result
                tool_use_id = None
                for b in resp.content:
                    if b.type == "tool_use":
                        tool_use_id = b.id
                        break
                messages.append({"role": "assistant", "content": assistant_content})
                if tool_use_id:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": f"Validation error: {str(first_err)}",
                                "is_error": True,
                            },
                            {
                                "type": "text",
                                "text": "Please call record_call_insights again with valid arguments strictly matching the schema.",
                            }
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"The previous tool call failed validation: {str(first_err)}. Please call record_call_insights with valid arguments strictly matching the schema."
                    })
            else:
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": f"The previous tool call failed validation: {str(first_err)}. Please call record_call_insights with valid arguments strictly matching the schema."
                })

            resp = _call_claude(messages)
            tool_input = None
            for block in resp.content:
                if block.type == "tool_use" and block.name == "record_call_insights":
                    tool_input = block.input
                    break
            if tool_input is None:
                raise ValueError(f"Retry failed to return record_call_insights tool use: {str(first_err)}")
            extraction = CallExtractionSchema.model_validate(tool_input)

        latency = time.time() - start_time
        tokens = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None)
        }

        return extraction, {"provider": f"anthropic ({self.model})", "latency_seconds": round(latency, 4), "tokens": tokens}


class FakeExtractor(BaseExtractor):
    """Test-only extractor that always returns a fabricated (hallucinated) quote.

    Used to verify that analyze_transcript correctly flags grounded=False when
    the evidence quote does not appear in the transcript.
    """

    def extract(self, segments: List[Segment]) -> Tuple[CallExtractionSchema, Dict[str, Any]]:
        extraction = CallExtractionSchema(
            customer_sentiment_start="neutral",
            customer_sentiment_end="neutral",
            agent_handled_well=True,
            resolution="resolved",
            reason="Fake extractor always invents quotes for testing purposes.",
            evidence=[EvidenceItem(
                quote="This fabricated quote never appeared in the original transcript at all.",
                timestamp=0.0
            )]
        )
        return extraction, {"provider": "fake", "latency_seconds": 0.0, "tokens": None}


def get_extractor(provider_name: Optional[str] = None) -> BaseExtractor:
    """Factory to get the extractor instance based on provider name or CALL_LLM_PROVIDER."""
    provider = (provider_name or os.environ.get("CALL_LLM_PROVIDER", "baseline")).lower().strip()

    if provider == "baseline":
        return BaselineExtractor()
    elif provider == "ollama":
        return OllamaExtractor()
    elif provider == "anthropic":
        return AnthropicExtractor()
    else:
        raise ValueError(
            f"Unknown extractor provider '{provider}'. Expected one of: 'baseline', 'ollama', 'anthropic'."
        )


def analyze_transcript(
    call_id: str,
    segments: List[Segment],
    provider: Optional[str] = None,
    extractor: Optional[BaseExtractor] = None,
    is_stereo: bool = False,
    speaker_diarization_confidence: float = 1.0,
    audio_file: Optional[str] = None,
    prosody_stats: Optional[dict] = None
) -> CallAnalysisResult:
    """Run full analysis pipeline on transcript segments:

    1. Redact PII from segments.
    2. Run extractor (baseline / ollama / anthropic / custom).
    3. Perform grounding verification on evidence quotes.
    4. Compute call duration and resolution time.
    5. Return validated CallAnalysisResult.

    Args:
        extractor: If provided, overrides the `provider` string lookup.
                   Used for dependency-injection in tests (e.g. FakeExtractor).
    """
    # 1. PII Redaction
    raw_seg_dicts = [s.model_dump() for s in segments]
    redacted_seg_dicts = redact_segments(raw_seg_dicts)
    redacted_segments = [Segment(**d) for d in redacted_seg_dicts]

    # Full transcript text for grounding check
    full_transcript = " ".join(s.text for s in redacted_segments)

    # 2. Extraction
    ext = extractor if extractor is not None else get_extractor(provider)
    extraction, meta = ext.extract(redacted_segments)

    # 3. Grounding Check
    is_grounded, ungrounded_quotes = verify_grounding(extraction.evidence, full_transcript)

    # 4. Timing
    duration = compute_call_duration(redacted_segments)
    resolution_time_min = calculate_resolution_time(
        extraction.resolution, extraction.evidence, redacted_segments
    )

    return CallAnalysisResult(
        call_id=call_id,
        audio_file=audio_file,
        duration_seconds=duration,
        resolution_time_minutes=resolution_time_min,
        grounded=is_grounded,
        ungrounded_quotes=ungrounded_quotes,
        extraction=extraction,
        segments=redacted_segments,
        is_stereo=is_stereo,
        speaker_diarization_confidence=speaker_diarization_confidence,
        prosody_stats=prosody_stats,
        llm_latency_seconds=meta.get("latency_seconds"),
        llm_tokens=meta.get("tokens"),
        provider=meta.get("provider", "baseline"),
    )
