"""Reporting module: CSV export, aggregate metrics summary, and markdown reports."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from callinsight.schema import CallAnalysisResult


def results_to_dataframe(results: List[CallAnalysisResult]) -> pd.DataFrame:
    """Convert a list of CallAnalysisResult objects into a flattened pandas DataFrame."""
    rows = []
    for res in results:
        shift = f"{res.extraction.customer_sentiment_start} -> {res.extraction.customer_sentiment_end}"
        evidence_str = "; ".join(
            f"[{e.timestamp:.1f}s] \"{e.quote}\"" if e.timestamp is not None else f"\"{e.quote}\""
            for e in res.extraction.evidence
        )

        row = {
            "call_id": res.call_id,
            "audio_file": res.audio_file or "",
            "duration_seconds": res.duration_seconds,
            "resolution_time_minutes": res.resolution_time_minutes,
            "resolution": res.extraction.resolution,
            "customer_sentiment_start": res.extraction.customer_sentiment_start,
            "customer_sentiment_end": res.extraction.customer_sentiment_end,
            "sentiment_shift": shift,
            "agent_handled_well": res.extraction.agent_handled_well,
            "grounded": res.grounded,
            "ungrounded_quotes_count": len(res.ungrounded_quotes),
            "reason": res.extraction.reason,
            "evidence": evidence_str,
            "provider": res.provider,
            "is_stereo": res.is_stereo,
            "diarization_confidence": res.speaker_diarization_confidence,
            "llm_latency_seconds": res.llm_latency_seconds,
        }

        # Prosody descriptive columns if available
        if res.prosody_stats:
            row["mean_pitch_hz"] = res.prosody_stats.get("mean_pitch_hz")
            row["mean_energy_rms"] = res.prosody_stats.get("mean_energy_rms")
            row["speaking_rate_wps"] = res.prosody_stats.get("speaking_rate_wps")

        rows.append(row)

    return pd.DataFrame(rows)


def generate_aggregate_summary(results: List[CallAnalysisResult]) -> Dict[str, Any]:
    """Compute aggregate quality metrics across all analyzed calls."""
    total = len(results)
    if total == 0:
        return {
            "notice": "SIMULATED DATA: Evaluated on simulated demo recordings.",
            "total_calls": 0,
            "resolved_count": 0,
            "percent_resolved": 0.0,
            "average_resolution_time_minutes": None,
            "percent_agent_handled_well": 0.0,
            "grounding_rate": 0.0,
            "sentiment_shift_counts": {},
        }

    resolved_calls = [r for r in results if r.extraction.resolution == "resolved"]
    unresolved_calls = [r for r in results if r.extraction.resolution == "unresolved"]
    unclear_calls = [r for r in results if r.extraction.resolution == "unclear"]

    # Resolution times for resolved calls with non-null timing
    res_times = [
        r.resolution_time_minutes for r in resolved_calls
        if r.resolution_time_minutes is not None
    ]
    avg_res_time = round(sum(res_times) / len(res_times), 2) if res_times else None

    # Handled well
    handled_well_count = sum(1 for r in results if r.extraction.agent_handled_well)

    # Grounding rate
    grounded_count = sum(1 for r in results if r.grounded)

    # Sentiment shifts
    shift_counts: Dict[str, int] = {}
    for r in results:
        shift_key = f"{r.extraction.customer_sentiment_start} -> {r.extraction.customer_sentiment_end}"
        shift_counts[shift_key] = shift_counts.get(shift_key, 0) + 1

    return {
        "notice": "SIMULATED DATA: Evaluated on simulated demo recordings.",
        "total_calls": total,
        "resolved_count": len(resolved_calls),
        "unresolved_count": len(unresolved_calls),
        "unclear_count": len(unclear_calls),
        "percent_resolved": round((len(resolved_calls) / total) * 100.0, 1),
        "average_resolution_time_minutes": avg_res_time,
        "percent_agent_handled_well": round((handled_well_count / total) * 100.0, 1),
        "grounding_rate": round((grounded_count / total) * 100.0, 1),
        "average_duration_seconds": round(sum(r.duration_seconds for r in results) / total, 1),
        "sentiment_shift_counts": shift_counts,
    }


def save_reports(
    results: List[CallAnalysisResult],
    output_dir: str
) -> Tuple[str, str, str]:
    """Save per-call CSV, summary JSON, and summary Markdown reports to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Per-call CSV
    df = results_to_dataframe(results)
    csv_path = os.path.join(output_dir, "calls_report.csv")
    df.to_csv(csv_path, index=False)

    # 2. Aggregate Summary JSON
    summary = generate_aggregate_summary(results)
    json_path = os.path.join(output_dir, "summary_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 3. Human-readable Markdown summary
    md_path = os.path.join(output_dir, "summary_report.md")
    md_lines = [
        "# Call Center Insights Summary Report",
        "",
        "> **SIMULATED DATA NOTICE**: All metrics and data reported herein are derived from simulated demo recordings.",
        "",
        "## Aggregate Metrics",
        f"- **Total Calls Analyzed**: {summary['total_calls']}",
        f"- **Percent Resolved**: {summary['percent_resolved']}% ({summary['resolved_count']}/{summary['total_calls']})",
        f"- **Average Resolution Time**: {summary['average_resolution_time_minutes']} minutes",
        f"- **Agent Handled Well**: {summary['percent_agent_handled_well']}%",
        f"- **Evidence Grounding Rate**: {summary['grounding_rate']}%",
        f"- **Average Call Duration**: {summary['average_duration_seconds']}s",
        "",
        "## Sentiment Shifts",
        "| Initial Sentiment -> Final Sentiment | Count |",
        "| ------------------------------------ | ----- |",
    ]
    for shift, count in sorted(summary["sentiment_shift_counts"].items(), key=lambda x: -x[1]):
        md_lines.append(f"| {shift} | {count} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return csv_path, json_path, md_path
