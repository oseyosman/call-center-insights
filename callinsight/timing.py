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

    Returns None when no segment is labeled 'customer'.  Does NOT fall back to
    segments[0] – a mono/unknown speaker must be handled explicitly by callers.
    """
    if not segments:
        return None
    for seg in segments:
        if seg.speaker == "customer":
            return seg
    return None


def find_matching_segment(quote: str, segments: List[Segment]) -> Optional[Segment]:
    """Find the transcript segment that contains the given quote.

    Matching strategy (in priority order):
    1. Normalised direct substring: norm_quote is a substring of norm_seg text.
    2. If the quote is ≥4 words: require that at least 80 % of its 4-gram
       windows (in order) appear in the concatenated normalised transcript.
    Both checks are case-insensitive and punctuation-stripped.
    """
    norm_quote = normalize_text_for_search(quote)
    if not norm_quote:
        return None

    # 1. Direct normalised substring match per segment
    for seg in segments:
        norm_seg = normalize_text_for_search(seg.text)
        if norm_quote in norm_seg or norm_seg in norm_quote:
            return seg

    # 2. 4-gram sliding window: require ≥80 % of windows to appear in the
    #    concatenated transcript text (ordered search).
    quote_words = norm_quote.split()
    if len(quote_words) >= 4:
        norm_all = " ".join(normalize_text_for_search(s.text) for s in segments)
        windows = [" ".join(quote_words[i:i + 4]) for i in range(len(quote_words) - 3)]
        matched_windows = sum(1 for w in windows if w in norm_all)
        if matched_windows / len(windows) >= 0.80:
            # Return the segment whose text contains the most 4-grams
            best_seg = None
            best_count = 0
            for seg in segments:
                norm_seg = normalize_text_for_search(seg.text)
                count = sum(1 for w in windows if w in norm_seg)
                if count > best_count:
                    best_count = count
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
      (timestamp of EARLIEST segment containing resolution evidence
       − timestamp of first customer segment) / 60.0

    Rules:
    - If resolution is NOT 'resolved', returns None.
    - If evidence list is empty or segments is empty, returns None.
    - If no customer-labeled segment exists, returns None (never guesses).
    - If no evidence quote matches any segment, returns None (not guessed).
    - If calculated minutes < 0 (edge case), returns 0.0.

    Matching order per evidence item:
    1. Quote is tried first (find_matching_segment).
    2. Timestamp proximity (within 2 s) is used only as a tiebreaker when the
       quote match fails.
    """
    if resolution != "resolved":
        return None

    if not evidence or not segments:
        return None

    first_cust_seg = find_first_customer_segment(segments)
    if first_cust_seg is None:
        # Mono/inferred audio: cannot determine customer start time
        return None

    # Collect matched segment start times for every evidence item
    matched_starts: List[float] = []

    for ev in evidence:
        seg = None

        # Primary: quote match
        if ev.quote:
            seg = find_matching_segment(ev.quote, segments)

        # Fallback: timestamp proximity (tiebreaker only)
        if seg is None and ev.timestamp is not None:
            for s in segments:
                if abs(s.start - ev.timestamp) <= 2.0:
                    seg = s
                    break

        if seg is not None:
            matched_starts.append(seg.start)

    if not matched_starts:
        return None

    # Use the EARLIEST matched timestamp
    matched_timestamp = min(matched_starts)

    diff_seconds = matched_timestamp - first_cust_seg.start
    if diff_seconds < 0:
        return 0.0

    return round(diff_seconds / 60.0, 2)
