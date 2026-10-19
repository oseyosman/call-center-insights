import pytest
from callinsight.schema import Segment, EvidenceItem
from callinsight.timing import (
    compute_call_duration,
    find_first_customer_segment,
    calculate_resolution_time,
)


def test_call_duration():
    segments = [
        Segment(start=5.0, end=10.0, speaker="agent", text="Hello"),
        Segment(start=10.5, end=65.0, speaker="customer", text="I need help"),
    ]
    assert compute_call_duration(segments) == 60.0
    assert compute_call_duration([]) == 0.0


def test_find_first_customer_segment():
    assert find_first_customer_segment([]) is None

    segments = [
        Segment(start=0.0, end=2.0, speaker="agent", text="Hi there"),
        Segment(start=2.5, end=6.0, speaker="customer", text="My card is blocked"),
        Segment(start=6.5, end=10.0, speaker="agent", text="Let me check"),
    ]
    first_cust = find_first_customer_segment(segments)
    assert first_cust is not None
    assert first_cust.start == 2.5
    assert first_cust.speaker == "customer"

    # Fallback when speaker is unknown
    mono_segments = [
        Segment(start=1.0, end=4.0, speaker="unknown", text="Hello caller"),
    ]
    assert find_first_customer_segment(mono_segments).start == 1.0


def test_resolution_time_resolved():
    segments = [
        Segment(start=0.0, end=3.0, speaker="agent", text="Thank you for calling support."),
        Segment(start=3.0, end=10.0, speaker="customer", text="My account is locked out."),
        Segment(start=12.0, end=30.0, speaker="agent", text="I have unlocked your account right now."),
        Segment(start=31.0, end=35.0, speaker="customer", text="Awesome, it works!"),
    ]
    evidence = [
        EvidenceItem(quote="I have unlocked your account right now.", timestamp=12.0)
    ]
    res_time = calculate_resolution_time("resolved", evidence, segments)
    # (12.0 - 3.0) / 60.0 = 9.0 / 60.0 = 0.15 minutes
    assert res_time == 0.15


def test_resolution_time_unresolved_returns_none():
    segments = [
        Segment(start=0.0, end=3.0, speaker="customer", text="Still broken."),
    ]
    evidence = [
        EvidenceItem(quote="Still broken.", timestamp=0.0)
    ]
    assert calculate_resolution_time("unresolved", evidence, segments) is None
    assert calculate_resolution_time("unclear", evidence, segments) is None


def test_resolution_time_empty_evidence_returns_none():
    segments = [
        Segment(start=0.0, end=10.0, speaker="customer", text="Issue fixed."),
    ]
    assert calculate_resolution_time("resolved", [], segments) is None


def test_resolution_time_hallucinated_evidence_returns_none():
    segments = [
        Segment(start=0.0, end=5.0, speaker="customer", text="Hello there."),
        Segment(start=5.0, end=10.0, speaker="agent", text="We will see."),
    ]
    evidence = [
        EvidenceItem(quote="A quote that never occurred in the call whatsoever", timestamp=None)
    ]
    # If quote doesn't match and timestamp is None, resolution time is null (not guessed)
    assert calculate_resolution_time("resolved", evidence, segments) is None
