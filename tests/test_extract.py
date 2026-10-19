import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from callinsight.schema import CallExtractionSchema, EvidenceItem, Segment
from callinsight.extract import (
    verify_grounding,
    BaselineExtractor,
    analyze_transcript,
    get_extractor,
    OllamaExtractor,
    AnthropicExtractor,
)


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


def test_grounding_check_ungrounded():
    transcript = "Agent: Hello. Customer: My internet is down. Agent: Please restart your router."
    evidence = [
        EvidenceItem(quote="I have credited your account 50 dollars.", timestamp=10.0)
    ]
    is_grounded, ungrounded = verify_grounding(evidence, transcript)
    assert is_grounded is False
    assert len(ungrounded) == 1
    assert "I have credited your account" in ungrounded[0]


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


def test_ollama_retry_on_invalid_json():
    extractor = OllamaExtractor()

    # Simulate first response returning invalid json, then second response returning valid json
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
