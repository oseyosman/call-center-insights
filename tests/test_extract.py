"""Regression tests for callinsight/extract.py.

Covers: verify_grounding (bug 3), FakeExtractor grounding flag (bug 4),
BaselineExtractor, mono/inferred speaker annotation (bug 5),
analyze_transcript PII redaction, and Ollama retry.
"""
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from callinsight.schema import CallExtractionSchema, EvidenceItem, Segment
from callinsight.extract import (
    verify_grounding,
    BaselineExtractor,
    FakeExtractor,
    analyze_transcript,
    get_extractor,
    OllamaExtractor,
    AnthropicExtractor,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_schema_validation_valid():
    data = {
        "customer_sentiment_start": "upset",
        "customer_sentiment_end": "satisfied",
        "agent_handled_well": True,
        "resolution": "resolved",
        "reason": "Agent issued a refund and customer thanked them.",
        "evidence": [{"quote": "I have processed the refund.", "timestamp": 45.0}],
    }
    schema = CallExtractionSchema.model_validate(data)
    assert schema.customer_sentiment_start == "upset"
    assert schema.customer_sentiment_end == "satisfied"
    assert schema.agent_handled_well is True
    assert schema.resolution == "resolved"


def test_schema_validation_invalid_sentiment():
    data = {
        "customer_sentiment_start": "delighted",  # invalid sentiment
        "customer_sentiment_end": "satisfied",
        "agent_handled_well": True,
        "resolution": "resolved",
        "reason": "Invalid test",
        "evidence": [],
    }
    with pytest.raises(ValidationError):
        CallExtractionSchema.model_validate(data)


# ---------------------------------------------------------------------------
# verify_grounding (bug 3: ordered 80% 4-gram window requirement)
# ---------------------------------------------------------------------------

def test_grounding_check_grounded():
    transcript = (
        "Agent: Hello how can I help you today? Customer: My bill has an unexpected charge. "
        "Agent: I see it here and I have refunded that fee for you right away. Customer: Thank you so much!"
    )
    evidence = [
        EvidenceItem(quote="I have refunded that fee for you right away.", timestamp=12.0),
        EvidenceItem(quote="Thank you so much!", timestamp=18.0),
    ]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is True
    assert ungrounded == []


def test_grounding_check_punctuation_difference_is_grounded():
    """Bug 3: a quote that differs only in punctuation must still be grounded."""
    transcript = "Agent: I have refunded that fee for you right away. Thank you for calling."
    evidence = [
        EvidenceItem(quote="I have refunded that fee for you right away", timestamp=5.0)
    ]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is True, f"Expected grounded, got ungrounded: {ungrounded}"


def test_grounding_check_ungrounded():
    transcript = "Agent: Hello. Customer: My internet is down. Agent: Please restart your router."
    evidence = [
        EvidenceItem(quote="I have credited your account 50 dollars.", timestamp=10.0)
    ]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is False
    assert len(ungrounded) == 1
    assert "I have credited your account" in ungrounded[0]


def test_grounding_check_half_fabricated_is_ungrounded():
    """Bug 3 regression: a quote where only one 4-gram matches (< 80%) must be ungrounded."""
    transcript = (
        "Agent: Your account balance is twenty dollars. "
        "Customer: Thank you very much."
    )
    # Real first half, fabricated second half – only ~1/4 windows will match
    fabricated_quote = "Your account balance is twenty dollars plus a fictitious credit of one hundred."
    evidence = [EvidenceItem(quote=fabricated_quote, timestamp=0.0)]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is False, "Half-fabricated quote should be ungrounded"
    assert len(ungrounded) == 1


def test_grounding_check_real_quote_fully_grounded():
    """Bug 3: a quote fully present in transcript must be grounded regardless of length."""
    transcript = (
        "Agent: I will apply the promotional discount of fifteen percent to your next order."
    )
    evidence = [EvidenceItem(
        quote="I will apply the promotional discount of fifteen percent to your next order.",
        timestamp=3.0
    )]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is True


# ---------------------------------------------------------------------------
# Baseline extractor
# ---------------------------------------------------------------------------

def test_baseline_extractor_runs():
    extractor = BaselineExtractor()
    segments = [
        Segment(start=0.0, end=4.0, speaker="customer", text="I am very angry and frustrated with this service!"),
        Segment(start=4.5, end=8.0, speaker="agent", text="I apologize for the trouble, let me fix this right now."),
        Segment(start=8.5, end=12.0, speaker="customer", text="Thank you, it is resolved and working now."),
    ]
    res, meta = extractor.extract(segments)
    assert res.customer_sentiment_start in ["upset", "angry"]
    assert res.customer_sentiment_end in ["satisfied", "neutral"]
    assert res.resolution == "resolved"
    assert meta["provider"] == "baseline"


def test_baseline_extractor_inferred_speaker_annotates_reason():
    """Bug 5: when all segments are 'inferred', reason must contain the inferred-speaker notice."""
    extractor = BaselineExtractor()
    segments = [
        Segment(start=0.0, end=5.0, speaker="inferred", text="Hello I need some help with my bill."),
        Segment(start=5.0, end=10.0, speaker="inferred", text="Let me look that up for you right away."),
    ]
    res, meta = extractor.extract(segments)
    assert "speaker attribution inferred" in res.reason.lower() or "inferred" in res.reason.lower(), \
        f"Expected inferred annotation in reason, got: {res.reason}"


# ---------------------------------------------------------------------------
# FakeExtractor grounding test (bug 4)
# ---------------------------------------------------------------------------

def test_fake_extractor_produces_ungrounded_result():
    """Bug 4: analyze_transcript must flag grounded=False when FakeExtractor is used."""
    segments = [
        Segment(start=0.0, end=5.0, speaker="customer", text="My router keeps disconnecting."),
        Segment(start=5.0, end=10.0, speaker="agent", text="I will reset your connection now."),
    ]
    result = analyze_transcript("call-fake-001", segments, extractor=FakeExtractor())
    assert result.grounded is False, "FakeExtractor's fabricated quote should be flagged as ungrounded"
    assert len(result.ungrounded_quotes) >= 1


# ---------------------------------------------------------------------------
# End-to-end with PII redaction
# ---------------------------------------------------------------------------

def test_analyze_transcript_end_to_end_with_pii_redaction():
    segments = [
        Segment(
            start=0.0,
            end=4.0,
            speaker="customer",
            text="My name is Bob, email bob@example.com, card 4532 0151 1283 0366."
        ),
        Segment(
            start=4.5,
            end=8.0,
            speaker="agent",
            text="I found your card and resolved the dispute."
        ),
    ]
    result = analyze_transcript("call-001", segments, provider="baseline")
    assert result.call_id == "call-001"
    # Verify PII is redacted in output segments
    assert "bob@example.com" not in result.segments[0].text
    assert "[REDACTED_EMAIL]" in result.segments[0].text
    assert "4532 0151 1283 0366" not in result.segments[0].text
    assert "[REDACTED_CARD]" in result.segments[0].text
    assert result.grounded is True


# ---------------------------------------------------------------------------
# Ollama retry
# ---------------------------------------------------------------------------

def test_ollama_retry_on_invalid_json():
    extractor = OllamaExtractor()

    valid_payload = {
        "message": {
            "content": (
                '{"customer_sentiment_start":"upset","customer_sentiment_end":"satisfied",'
                '"agent_handled_well":true,"resolution":"resolved","reason":"Issue resolved.",'
                '"evidence":[{"quote":"fixed","timestamp":1.0}]}'
            )
        },
        "prompt_eval_count": 50,
        "eval_count": 40
    }
    invalid_payload = {
        "message": {"content": "Not a valid json response"}
    }

    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = invalid_payload
    mock_resp1.raise_for_status.return_value = None

    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = valid_payload
    mock_resp2.raise_for_status.return_value = None

    with patch("requests.post", side_effect=[mock_resp1, mock_resp2]) as mock_post:
        segments = [Segment(start=0.0, end=2.0, speaker="customer", text="fixed")]
        extraction, meta = extractor.extract(segments)
        assert mock_post.call_count == 2
        assert extraction.resolution == "resolved"
        assert meta["tokens"]["eval_count"] == 40


# ---------------------------------------------------------------------------
# Anthropic extractor & retry (Bug 7)
# ---------------------------------------------------------------------------

def test_anthropic_extractor_success():
    extractor = AnthropicExtractor(api_key="test-key-mock")

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "record_call_insights"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {
        "customer_sentiment_start": "upset",
        "customer_sentiment_end": "satisfied",
        "agent_handled_well": True,
        "resolution": "resolved",
        "reason": "Agent solved the issue quickly.",
        "evidence": [{"quote": "All fixed now", "timestamp": 10.0}]
    }

    mock_resp = MagicMock()
    mock_resp.content = [mock_tool_block]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_resp

        segments = [Segment(start=0.0, end=12.0, speaker="customer", text="All fixed now")]
        extraction, meta = extractor.extract(segments)

        assert extraction.resolution == "resolved"
        assert extraction.agent_handled_well is True
        assert meta["tokens"]["input_tokens"] == 100
        # Check messages.create kwargs: temperature must not be present
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["max_tokens"] == 1024

