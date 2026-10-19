"""Timing calculations for call-center audio and transcripts.

All timing is computed purely from segment timestamps, never estimated or guessed.
"""

from typing import List, Optional, Tuple
import re
from callinsight.schema import Segment, EvidenceItem


def normalize_text_for_search(text: str) -> str:
    """Normalize text by lowercasing and stripping non-alphanumeric chars for substring matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def compute_call_duration(segments: List[Segment]) -> float:
    """Calculate total call duration in seconds from the first to the last segment.

    Returns 0.0 if segments list is empty.
    """
    if not segments:
        return 0.0
    start_time = min(s.start for s in segments)
    end_time = max(s.end for s in segments)
    return max(0.0, round(end_time - start_time, 2))


def find_first_customer_segment(segments: List[Segment]) -> Optional[Segment]:
    """Find the first segment spoken by the customer.

    If speaker labels are unavailable or all unknown, falls back to the very first segment.
    """
    if not segments:
        return None
    for seg in segments:
        if seg.speaker == "customer":
            return seg
    # Fallback if speaker diarization is mono/unknown
    return segments[0]


def find_matching_segment(quote: str, segments: List[Segment]) -> Optional[Segment]:
    """Find the transcript segment that contains the given quote."""
    norm_quote = normalize_text_for_search(quote)
    if not norm_quote:
        return None

    # First attempt: direct normalized substring in segment
    for seg in segments:
        norm_seg = normalize_text_for_search(seg.text)
        if norm_quote in norm_seg or norm_seg in norm_quote:
            return seg

    # Second attempt: token overlap if segment was slightly split
    quote_tokens = set(norm_quote.split())
    if len(quote_tokens) >= 3:
        best_seg = None
        best_overlap = 0
        for seg in segments:
            seg_tokens = set(normalize_text_for_search(seg.text).split())
            overlap = len(quote_tokens.intersection(seg_tokens))
            if overlap > best_overlap and overlap >= min(3, len(quote_tokens) // 2):
                best_overlap = overlap
                best_seg = seg
        if best_seg:
            return best_seg

    return None


def calculate_resolution_time(
    resolution: str,
    evidence: List[EvidenceItem],
    segments: List[Segment]
) -> Optional[float]:
    """Calculate resolution time in minutes from the first customer segment.

    Formula:
      (timestamp of segment containing resolution evidence - timestamp of first customer segment) / 60.0

    Rules:
    - If resolution is NOT 'resolved', returns None.
    - If evidence list is empty, returns None.
    - If no evidence quote matches any segment, returns None (not guessed).
    - If calculated minutes < 0 (edge case where agent resolves before first customer speech), returns 0.0.
    """
    if resolution != "resolved":
        return None

    if not evidence or not segments:
        return None

    first_cust_seg = find_first_customer_segment(segments)
    if first_cust_seg is None:
        return None

    # Search for the earliest matching segment among evidence items
    # (or the one matching the resolution evidence)
    matched_timestamp = None
    for ev in evidence:
        # If timestamp is directly provided and valid, try to find matching segment around timestamp
        if ev.timestamp is not None:
            # Verify against segments
            for seg in segments:
                if abs(seg.start - ev.timestamp) <= 2.0:
                    matched_timestamp = seg.start
                    break
        if matched_timestamp is None and ev.quote:
            matching_seg = find_matching_segment(ev.quote, segments)
            if matching_seg:
                matched_timestamp = matching_seg.start
                break

    if matched_timestamp is None:
        return None

    diff_seconds = matched_timestamp - first_cust_seg.start
    if diff_seconds < 0:
        return 0.0

    return round(diff_seconds / 60.0, 2)
