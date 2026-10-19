"""Comprehensive evaluation harness for Call Center Insights.

Evaluates:
- Sentiment accuracy and macro-F1 (start and end)
- Resolution accuracy and confusion matrix
- Grounding rate of evidence quotes
- Resolution-time MAE in minutes
- Word Error Rate (WER) on audio subset (separating STT errors from extraction errors)
- Latency and token consumption
- Saves evals/results/latest.json and latest.md with detailed Failures section
- Exits 1 if specified gating thresholds are not met.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

from callinsight.extract import analyze_transcript
from callinsight.schema import Segment
from callinsight.transcribe import transcribe_audio
from evals.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_mae,
    compute_word_error_rate,
)

SENTIMENT_CLASSES = ["satisfied", "neutral", "upset", "angry"]
RESOLUTION_CLASSES = ["resolved", "unresolved", "unclear"]


def load_dataset(data_dir: str, split: str) -> List[dict]:
    """Load transcripts for dev, test, or both."""
    records = []
    splits_to_load = []
    if split in ("dev", "both"):
        splits_to_load.append(os.path.join(data_dir, "dev_transcripts.json"))
    if split in ("test", "both"):
        splits_to_load.append(os.path.join(data_dir, "test_transcripts.json"))

    for path in splits_to_load:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transcript dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def evaluate_audio_subset(
    records: List[dict],
    audio_dir: str,
    whisper_model: str = "base"
) -> Dict[str, Any]:
    """Calculate Word Error Rate (WER) on calls that have corresponding audio files in audio_dir."""
    if not os.path.exists(audio_dir):
        return {"audio_count": 0, "wer": None, "note": "Audio directory not found."}

    references = []
    hypotheses = []
    audio_calls = []

    for rec in records:
        call_id = rec["call_id"]
        wav_path = os.path.join(audio_dir, f"{call_id}.wav")
        if os.path.exists(wav_path):
            # Ground truth text
            ref_text = " ".join(s["text"] for s in rec["segments"])

            # Transcribe with whisper
            try:
                whisper_segs, _, _ = transcribe_audio(wav_path, model_size=whisper_model)
                hyp_text = " ".join(s.text for s in whisper_segs)
                references.append(ref_text)
                hypotheses.append(hyp_text)
                audio_calls.append(call_id)
            except Exception as e:
                print(f"Warning: Failed to transcribe {wav_path}: {e}", file=sys.stderr)

    if not references:
        return {"audio_count": 0, "wer": None, "note": "No matching audio files evaluated."}

    wer_score = compute_word_error_rate(references, hypotheses)
    return {
        "audio_count": len(references),
        "wer_percent": wer_score,
        "evaluated_calls": audio_calls,
        "whisper_model": whisper_model,
    }


def run_evaluation(
    provider: str = "baseline",
    split: str = "dev",
    data_dir: str = "evals/data",
    audio_dir: str = "samples",
    whisper_model: str = "base",
    out_json: str = "evals/results/latest.json",
    out_md: str = "evals/results/latest.md",
    eval_audio: bool = True
) -> Dict[str, Any]:
    """Run full evaluation suite on the given provider and dataset split."""
    records = load_dataset(data_dir, split)
    print(f"\n=======================================================")
    print(f"  Running Evaluation: Provider='{provider}', Split='{split}' ({len(records)} calls)")
    print(f"  NOTICE: All evaluation data is SIMULATED.")
    print(f"=======================================================\n")

    y_true_sent_start, y_pred_sent_start = [], []
    y_true_sent_end, y_pred_sent_end = [], []
    y_true_res, y_pred_res = [], []
    y_true_res_time, y_pred_res_time = [], []

    grounded_flags = []
    latencies = []
    total_prompt_tokens = 0
    total_eval_tokens = 0
    failures = []

    for idx, rec in enumerate(records, 1):
        call_id = rec["call_id"]
        gt = rec["ground_truth"]
        segments = [Segment(**s) for s in rec["segments"]]

        # Run analysis pipeline
        t0 = time.time()
        result = analyze_transcript(call_id, segments, provider=provider)
        dur = time.time() - t0
        latencies.append(dur)

        ext = result.extraction

        # Tokens if present
        if result.llm_tokens:
            total_prompt_tokens += result.llm_tokens.get("prompt_eval_count") or result.llm_tokens.get("input_tokens") or 0
            total_eval_tokens += result.llm_tokens.get("eval_count") or result.llm_tokens.get("output_tokens") or 0

        # Sentiment start & end
        y_true_sent_start.append(gt["customer_sentiment_start"])
        y_pred_sent_start.append(ext.customer_sentiment_start)

        y_true_sent_end.append(gt["customer_sentiment_end"])
        y_pred_sent_end.append(ext.customer_sentiment_end)

        # Resolution
        y_true_res.append(gt["resolution"])
        y_pred_res.append(ext.resolution)

        # Resolution timing
        y_true_res_time.append(gt.get("resolution_time_minutes"))
        y_pred_res_time.append(result.resolution_time_minutes)

        # Grounding
        grounded_flags.append(result.grounded)

        # Identify Failure Cases
        call_failed = False
        fail_reasons = []

        if ext.customer_sentiment_start != gt["customer_sentiment_start"]:
            call_failed = True
            fail_reasons.append(
                f"Sentiment start mismatch: predicted '{ext.customer_sentiment_start}', true '{gt['customer_sentiment_start']}'"
            )

        if ext.customer_sentiment_end != gt["customer_sentiment_end"]:
            call_failed = True
            fail_reasons.append(
                f"Sentiment end mismatch: predicted '{ext.customer_sentiment_end}', true '{gt['customer_sentiment_end']}'"
            )

        if ext.resolution != gt["resolution"]:
            call_failed = True
            fail_reasons.append(
                f"Resolution mismatch: predicted '{ext.resolution}', true '{gt['resolution']}'"
            )

        if not result.grounded:
            call_failed = True
            fail_reasons.append(f"Ungrounded quotes: {result.ungrounded_quotes}")

        if call_failed:
            failures.append({
                "call_id": call_id,
                "hard_case_type": rec.get("hard_case_type", "standard"),
                "reasons": fail_reasons,
                "extractor_reason": ext.reason,
            })

    # Metrics computation
    acc_sent_start = compute_accuracy(y_true_sent_start, y_pred_sent_start)
    f1_sent_start = compute_macro_f1(y_true_sent_start, y_pred_sent_start, SENTIMENT_CLASSES)

    acc_sent_end = compute_accuracy(y_true_sent_end, y_pred_sent_end)
    f1_sent_end = compute_macro_f1(y_true_sent_end, y_pred_sent_end, SENTIMENT_CLASSES)

    acc_resolution = compute_accuracy(y_true_res, y_pred_res)
    conf_matrix = compute_confusion_matrix(y_true_res, y_pred_res, RESOLUTION_CLASSES)

    grounding_rate = round((sum(1 for g in grounded_flags if g) / len(grounded_flags)) * 100.0, 2)
    res_time_mae, evaluated_timing_count = compute_mae(y_true_res_time, y_pred_res_time)

    mean_latency = round(float(np.mean(latencies)), 4)
    p95_latency = round(float(np.percentile(latencies, 95)), 4)

    # Audio subset WER evaluation
    audio_metrics = {}
    if eval_audio:
        print("Evaluating Word Error Rate (WER) on rendered audio subset...")
        audio_metrics = evaluate_audio_subset(records, audio_dir, whisper_model=whisper_model)

    report_data = {
        "metadata": {
            "notice": "SIMULATED DATA: All evaluation results are strictly from synthetic/simulated call transcripts and recordings.",
            "provider": provider,
            "split": split,
            "total_calls": len(records),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "sentiment_start": {
            "accuracy_percent": acc_sent_start,
            "macro_f1": f1_sent_start,
        },
        "sentiment_end": {
            "accuracy_percent": acc_sent_end,
            "macro_f1": f1_sent_end,
        },
        "resolution": {
            "accuracy_percent": acc_resolution,
            "confusion_matrix": conf_matrix,
        },
        "grounding": {
            "grounding_rate_percent": grounding_rate,
        },
        "timing": {
            "resolution_time_mae_minutes": res_time_mae,
            "evaluated_calls_count": evaluated_timing_count,
        },
        "audio_transcription": audio_metrics,
        "performance": {
            "mean_latency_seconds": mean_latency,
            "p95_latency_seconds": p95_latency,
            "total_prompt_tokens": total_prompt_tokens if total_prompt_tokens > 0 else None,
            "total_eval_tokens": total_eval_tokens if total_eval_tokens > 0 else None,
        },
        "failures_count": len(failures),
        "failures": failures,
    }

    # Save JSON report
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    md_content = build_markdown_report(report_data)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print summary to console
    print_console_summary(report_data)
    print(f"\nSaved evaluation reports:")
    print(f"  JSON: {out_json}")
    print(f"  Markdown: {out_md}")

    return report_data


def build_markdown_report(data: Dict[str, Any]) -> str:
    """Build formatted Markdown report with metrics tables, confusion matrix, and Failures section."""
    meta = data["metadata"]
    cm = data["resolution"]["confusion_matrix"]

    lines = [
        f"# Evaluation Report: {meta['provider'].upper()} ({meta['split'].upper()} split)",
        "",
        "> **SIMULATED DATA NOTICE**: All metrics, transcripts, and audio files are synthetic and simulated for QA testing.",
        "",
        f"- **Evaluated Date**: {meta['timestamp']}",
        f"- **Total Calls**: {meta['total_calls']}",
        f"- **Provider**: `{meta['provider']}`",
        "",
        "## Summary Metrics",
        "| Metric | Score | Note |",
        "| ------ | ----- | ---- |",
        f"| Sentiment Start Accuracy | {data['sentiment_start']['accuracy_percent']}% | Macro-F1: {data['sentiment_start']['macro_f1']} |",
        f"| Sentiment End Accuracy | {data['sentiment_end']['accuracy_percent']}% | Macro-F1: {data['sentiment_end']['macro_f1']} |",
        f"| Resolution Accuracy | {data['resolution']['accuracy_percent']}% | Evaluated on 3-class problem |",
        f"| Evidence Grounding Rate | {data['grounding']['grounding_rate_percent']}% | Verbatim transcript match |",
        f"| Resolution-Time MAE | {data['timing']['resolution_time_mae_minutes']} min | Evaluated on {data['timing']['evaluated_calls_count']} calls |",
        f"| Latency (Mean / P95) | {data['performance']['mean_latency_seconds']}s / {data['performance']['p95_latency_seconds']}s | Per-call latency |",
    ]

    audio_res = data.get("audio_transcription", {})
    if audio_res.get("wer_percent") is not None:
        lines.append(
            f"| Audio Subset WER | {audio_res['wer_percent']}% | {audio_res['audio_count']} calls transcribed via faster-whisper ({audio_res.get('whisper_model')}) |"
        )

    lines.extend([
        "",
        "## Resolution Confusion Matrix",
        "| True \\ Pred | resolved | unresolved | unclear |",
        "| ----------- | -------- | ---------- | ------- |",
        f"| **resolved** | {cm['resolved'].get('resolved', 0)} | {cm['resolved'].get('unresolved', 0)} | {cm['resolved'].get('unclear', 0)} |",
        f"| **unresolved** | {cm['unresolved'].get('resolved', 0)} | {cm['unresolved'].get('unresolved', 0)} | {cm['unresolved'].get('unclear', 0)} |",
        f"| **unclear** | {cm['unclear'].get('resolved', 0)} | {cm['unclear'].get('unresolved', 0)} | {cm['unclear'].get('unclear', 0)} |",
        "",
        f"## Failures Section ({data['failures_count']} misclassified calls)",
    ])

    if not data["failures"]:
        lines.append("\n*No failures observed.*")
    else:
        for idx, f in enumerate(data["failures"], 1):
            lines.append(f"\n### {idx}. `{f['call_id']}` (Hard case: *{f['hard_case_type']}*)")
            for r in f["reasons"]:
                lines.append(f"- **Issue**: {r}")
            lines.append(f"- **Model Explanation**: *{f['extractor_reason']}*")

    return "\n".join(lines) + "\n"


def print_console_summary(data: Dict[str, Any]):
    """Print readable summary to stdout."""
    print("--- EVALUATION RESULTS ---")
    print(f"Sentiment Start Acc : {data['sentiment_start']['accuracy_percent']}% (Macro-F1: {data['sentiment_start']['macro_f1']})")
    print(f"Sentiment End Acc   : {data['sentiment_end']['accuracy_percent']}% (Macro-F1: {data['sentiment_end']['macro_f1']})")
    print(f"Resolution Acc      : {data['resolution']['accuracy_percent']}%")
    print(f"Grounding Rate      : {data['grounding']['grounding_rate_percent']}%")
    print(f"Resolution-Time MAE : {data['timing']['resolution_time_mae_minutes']} min")
    print(f"Mean Latency        : {data['performance']['mean_latency_seconds']}s")
    if data.get("audio_transcription", {}).get("wer_percent") is not None:
        print(f"Audio Subset WER    : {data['audio_transcription']['wer_percent']}%")
    print(f"Failure Cases Count : {data['failures_count']}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on Call Center Insights.")
    parser.add_argument("--provider", default="baseline", help="Extractor provider (baseline, ollama, anthropic).")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "both"], help="Split to evaluate.")
    parser.add_argument("--data-dir", default="evals/data", help="Directory of transcript JSON files.")
    parser.add_argument("--audio-dir", default="samples", help="Directory of rendered audio samples.")
    parser.add_argument("--whisper", default="base", help="Whisper model size for audio WER evaluation.")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio WER evaluation.")
    parser.add_argument("--out-json", default="evals/results/latest.json", help="Path for JSON output.")
    parser.add_argument("--out-md", default="evals/results/latest.md", help="Path for Markdown output.")

    # Quality Gate Thresholds
    parser.add_argument("--min-sentiment-start-acc", type=float, default=None, help="Threshold for sentiment start accuracy.")
    parser.add_argument("--min-sentiment-end-acc", type=float, default=None, help="Threshold for sentiment end accuracy.")
    parser.add_argument("--min-resolution-acc", type=float, default=None, help="Threshold for resolution accuracy.")
    parser.add_argument("--min-grounding-rate", type=float, default=None, help="Threshold for grounding rate.")
    parser.add_argument("--max-resolution-time-mae", type=float, default=None, help="Threshold for maximum resolution time MAE.")

    args = parser.parse_args()

    results = run_evaluation(
        provider=args.provider,
        split=args.split,
        data_dir=args.data_dir,
        audio_dir=args.audio_dir,
        whisper_model=args.whisper,
        out_json=args.out_json,
        out_md=args.out_md,
        eval_audio=not args.no_audio
    )

    # Gating checks
    gate_failed = False
    if args.min_sentiment_start_acc is not None:
        actual = results["sentiment_start"]["accuracy_percent"]
        if actual < args.min_sentiment_start_acc:
            print(f"FAILED GATING: Sentiment start accuracy {actual}% < threshold {args.min_sentiment_start_acc}%", file=sys.stderr)
            gate_failed = True

    if args.min_sentiment_end_acc is not None:
        actual = results["sentiment_end"]["accuracy_percent"]
        if actual < args.min_sentiment_end_acc:
            print(f"FAILED GATING: Sentiment end accuracy {actual}% < threshold {args.min_sentiment_end_acc}%", file=sys.stderr)
            gate_failed = True

    if args.min_resolution_acc is not None:
        actual = results["resolution"]["accuracy_percent"]
        if actual < args.min_resolution_acc:
            print(f"FAILED GATING: Resolution accuracy {actual}% < threshold {args.min_resolution_acc}%", file=sys.stderr)
            gate_failed = True

    if args.min_grounding_rate is not None:
        actual = results["grounding"]["grounding_rate_percent"]
        if actual < args.min_grounding_rate:
            print(f"FAILED GATING: Grounding rate {actual}% < threshold {args.min_grounding_rate}%", file=sys.stderr)
            gate_failed = True

    if args.max_resolution_time_mae is not None:
        actual = results["timing"]["resolution_time_mae_minutes"]
        if actual is not None and actual > args.max_resolution_time_mae:
            print(f"FAILED GATING: Resolution time MAE {actual}m > threshold {args.max_resolution_time_mae}m", file=sys.stderr)
            gate_failed = True

    if gate_failed:
        print("\nQuality gate FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll gating checks passed successfully (exit 0).")
        sys.exit(0)


if __name__ == "__main__":
    main()
