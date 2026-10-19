"""Comprehensive evaluation harness for Call Center Insights.

Evaluates:
- Sentiment accuracy and macro-F1 with ordinal metrics (start and end)
- Agent-handled-well accuracy and F1
- Resolution accuracy and confusion matrix
- Grounding rate of evidence quotes
- Resolution-time MAE in minutes
- Word Error Rate (WER) on audio subset
- Wilson 95% CIs for all accuracy metrics
- Bootstrap 95% CIs (1 000 resamples, seed=42) for MAE and macro-F1
- Latency and token consumption

Outputs:
- evals/results/<name>.json and .md with detailed Failures section
- Exits 1 if specified gating thresholds are not met
- Exits 1 if any metric regresses by more than a configured delta vs --baseline-json

⚠️  SIMULATED DATA NOTICE: All evaluation data is SYNTHETIC / SIMULATED.
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
    compute_binary_accuracy,
    compute_binary_f1,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_mae,
    compute_word_error_rate,
    compute_sentiment_within_one_accuracy,
    compute_sentiment_mae,
    compute_quadratic_weighted_kappa,
    wilson_confidence_interval,
    accuracy_wilson_ci,
    binary_accuracy_wilson_ci,
    grounding_wilson_ci,
    mae_bootstrap_ci,
    macro_f1_bootstrap_ci,
)

SENTIMENT_CLASSES = ["satisfied", "neutral", "upset", "angry"]
RESOLUTION_CLASSES = ["resolved", "unresolved", "unclear"]


def load_dataset(data_dir: str, split: str) -> List[dict]:
    """Load transcripts for dev, test, heldout, dev+test, or all."""
    records: List[dict] = []
    split_map = {
        "dev": ["dev_transcripts.json"],
        "test": ["test_transcripts.json"],
        "heldout": ["heldout_transcripts.json"],
        "dev+test": ["dev_transcripts.json", "test_transcripts.json"],
        "both": ["dev_transcripts.json", "test_transcripts.json"],
        "all": ["dev_transcripts.json", "test_transcripts.json", "heldout_transcripts.json"],
    }
    filenames = split_map.get(split)
    if filenames is None:
        raise ValueError(f"Unknown split '{split}'. Use: {list(split_map)}")

    for fname in filenames:
        path = os.path.join(data_dir, fname)
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
    """Calculate WER on calls that have corresponding audio files in audio_dir."""
    if not os.path.exists(audio_dir):
        return {"audio_count": 0, "wer": None, "note": "Audio directory not found."}

    references = []
    hypotheses = []
    audio_calls = []

    for rec in records:
        call_id = rec["call_id"]
        wav_path = os.path.join(audio_dir, f"{call_id}.wav")
        if os.path.exists(wav_path):
            ref_text = " ".join(s["text"] for s in rec["segments"])
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
    eval_audio: bool = True,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
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
    y_true_handled, y_pred_handled = [], []

    grounded_flags = []
    latencies = []
    total_prompt_tokens = 0
    total_eval_tokens = 0
    failures = []

    for idx, rec in enumerate(records, 1):
        call_id = rec["call_id"]
        gt = rec["ground_truth"]
        segments = [Segment(**s) for s in rec["segments"]]

        t0 = time.time()
        result = analyze_transcript(call_id, segments, provider=provider)
        dur = time.time() - t0
        latencies.append(dur)

        ext = result.extraction

        if result.llm_tokens:
            total_prompt_tokens += result.llm_tokens.get("prompt_eval_count") or result.llm_tokens.get("input_tokens") or 0
            total_eval_tokens += result.llm_tokens.get("eval_count") or result.llm_tokens.get("output_tokens") or 0

        y_true_sent_start.append(gt["customer_sentiment_start"])
        y_pred_sent_start.append(ext.customer_sentiment_start)

        y_true_sent_end.append(gt["customer_sentiment_end"])
        y_pred_sent_end.append(ext.customer_sentiment_end)

        y_true_res.append(gt["resolution"])
        y_pred_res.append(ext.resolution)

        y_true_res_time.append(gt.get("resolution_time_minutes"))
        y_pred_res_time.append(result.resolution_time_minutes)

        y_true_handled.append(bool(gt.get("agent_handled_well", True)))
        y_pred_handled.append(bool(ext.agent_handled_well))

        grounded_flags.append(result.grounded)

        # Identify failure cases
        call_failed = False
        fail_reasons = []

        if ext.customer_sentiment_start != gt["customer_sentiment_start"]:
            call_failed = True
            fail_reasons.append(f"Sentiment start mismatch: predicted '{ext.customer_sentiment_start}', true '{gt['customer_sentiment_start']}'")

        if ext.customer_sentiment_end != gt["customer_sentiment_end"]:
            call_failed = True
            fail_reasons.append(f"Sentiment end mismatch: predicted '{ext.customer_sentiment_end}', true '{gt['customer_sentiment_end']}'")

        if ext.resolution != gt["resolution"]:
            call_failed = True
            fail_reasons.append(f"Resolution mismatch: predicted '{ext.resolution}', true '{gt['resolution']}'")

        if ext.agent_handled_well != bool(gt.get("agent_handled_well", True)):
            call_failed = True
            fail_reasons.append(f"agent_handled_well mismatch: predicted {ext.agent_handled_well}, true {gt.get('agent_handled_well')}")

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

    # -----------------------------------------------------------------------
    # Metrics with CIs
    # -----------------------------------------------------------------------

    # Sentiment start
    acc_sent_start = compute_accuracy(y_true_sent_start, y_pred_sent_start)
    f1_sent_start = compute_macro_f1(y_true_sent_start, y_pred_sent_start, SENTIMENT_CLASSES)
    ci_acc_start = accuracy_wilson_ci(y_true_sent_start, y_pred_sent_start)
    ci_f1_start = macro_f1_bootstrap_ci(y_true_sent_start, y_pred_sent_start, SENTIMENT_CLASSES,
                                         n_resamples=n_bootstrap, seed=bootstrap_seed)
    w1_start = compute_sentiment_within_one_accuracy(y_true_sent_start, y_pred_sent_start)
    smae_start = compute_sentiment_mae(y_true_sent_start, y_pred_sent_start)
    kappa_start = compute_quadratic_weighted_kappa(y_true_sent_start, y_pred_sent_start)

    # Sentiment end
    acc_sent_end = compute_accuracy(y_true_sent_end, y_pred_sent_end)
    f1_sent_end = compute_macro_f1(y_true_sent_end, y_pred_sent_end, SENTIMENT_CLASSES)
    ci_acc_end = accuracy_wilson_ci(y_true_sent_end, y_pred_sent_end)
    ci_f1_end = macro_f1_bootstrap_ci(y_true_sent_end, y_pred_sent_end, SENTIMENT_CLASSES,
                                       n_resamples=n_bootstrap, seed=bootstrap_seed)
    w1_end = compute_sentiment_within_one_accuracy(y_true_sent_end, y_pred_sent_end)
    smae_end = compute_sentiment_mae(y_true_sent_end, y_pred_sent_end)
    kappa_end = compute_quadratic_weighted_kappa(y_true_sent_end, y_pred_sent_end)

    # Resolution
    acc_resolution = compute_accuracy(y_true_res, y_pred_res)
    conf_matrix = compute_confusion_matrix(y_true_res, y_pred_res, RESOLUTION_CLASSES)
    ci_acc_res = accuracy_wilson_ci(y_true_res, y_pred_res)

    # agent_handled_well
    acc_handled = compute_binary_accuracy(y_true_handled, y_pred_handled)
    f1_handled = compute_binary_f1(y_true_handled, y_pred_handled)
    ci_acc_handled = binary_accuracy_wilson_ci(y_true_handled, y_pred_handled)

    # Grounding
    grounding_rate = round((sum(1 for g in grounded_flags if g) / len(grounded_flags)) * 100.0, 2)
    ci_grounding = grounding_wilson_ci(grounded_flags)

    # Resolution time
    res_time_mae, evaluated_timing_count = compute_mae(y_true_res_time, y_pred_res_time)
    ci_mae = mae_bootstrap_ci(y_true_res_time, y_pred_res_time,
                               n_resamples=n_bootstrap, seed=bootstrap_seed)

    # Latency
    mean_latency = round(float(np.mean(latencies)), 4)
    p95_latency = round(float(np.percentile(latencies, 95)), 4)

    # Audio WER
    audio_metrics: Dict[str, Any] = {}
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
            "bootstrap_resamples": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
        },
        "sentiment_start": {
            "accuracy_percent": acc_sent_start,
            "accuracy_wilson_95ci": list(ci_acc_start),
            "macro_f1": f1_sent_start,
            "macro_f1_bootstrap_95ci": [ci_f1_start[0], ci_f1_start[1]] if ci_f1_start[0] is not None else None,
            "within_one_accuracy_percent": w1_start,
            "ordinal_mae": smae_start,
            "quadratic_weighted_kappa": kappa_start,
        },
        "sentiment_end": {
            "accuracy_percent": acc_sent_end,
            "accuracy_wilson_95ci": list(ci_acc_end),
            "macro_f1": f1_sent_end,
            "macro_f1_bootstrap_95ci": [ci_f1_end[0], ci_f1_end[1]] if ci_f1_end[0] is not None else None,
            "within_one_accuracy_percent": w1_end,
            "ordinal_mae": smae_end,
            "quadratic_weighted_kappa": kappa_end,
        },
        "resolution": {
            "accuracy_percent": acc_resolution,
            "accuracy_wilson_95ci": list(ci_acc_res),
            "confusion_matrix": conf_matrix,
        },
        "agent_handled_well": {
            "accuracy_percent": acc_handled,
            "accuracy_wilson_95ci": list(ci_acc_handled),
            "f1_positive_class": f1_handled,
        },
        "grounding": {
            "grounding_rate_percent": grounding_rate,
            "grounding_wilson_95ci": list(ci_grounding),
            "note": (
                "The baseline's grounding rate is trivially 100% because its evidence quotes are "
                "copied verbatim from transcript segments. This number is only meaningful for LLM "
                "providers (ollama, anthropic) that must generate quotes from memory."
            ),
        },
        "timing": {
            "resolution_time_mae_minutes": res_time_mae,
            "mae_bootstrap_95ci_minutes": [ci_mae[0], ci_mae[1]] if ci_mae[0] is not None else None,
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

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    md_content = build_markdown_report(report_data)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print_console_summary(report_data)
    print(f"\nSaved evaluation reports:")
    print(f"  JSON: {out_json}")
    print(f"  Markdown: {out_md}")

    return report_data


def _fmt_ci(ci_pair: Optional[List[float]], pct: bool = True) -> str:
    """Format a [lo, hi] CI pair for display."""
    if ci_pair is None:
        return "n/a"
    lo, hi = ci_pair
    if pct:
        return f"[{lo:.1f}%, {hi:.1f}%]"
    return f"[{lo:.4f}, {hi:.4f}]"


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
        f"- **Bootstrap**: {meta.get('bootstrap_resamples', 1000)} resamples, seed={meta.get('bootstrap_seed', 42)}",
        "",
        "## Summary Metrics",
        "| Metric | Score | 95% CI | Ordinal / Extra |",
        "| ------ | ----- | ------ | --------------- |",
        f"| Sentiment Start Accuracy | {data['sentiment_start']['accuracy_percent']}% | {_fmt_ci(data['sentiment_start']['accuracy_wilson_95ci'])} | Macro-F1: {data['sentiment_start']['macro_f1']} {_fmt_ci(data['sentiment_start'].get('macro_f1_bootstrap_95ci'), pct=False)} |",
        f"| Sentiment Start Within-One | {data['sentiment_start']['within_one_accuracy_percent']}% | — | OrdMAE: {data['sentiment_start']['ordinal_mae']}, QWK: {data['sentiment_start']['quadratic_weighted_kappa']} |",
        f"| Sentiment End Accuracy | {data['sentiment_end']['accuracy_percent']}% | {_fmt_ci(data['sentiment_end']['accuracy_wilson_95ci'])} | Macro-F1: {data['sentiment_end']['macro_f1']} {_fmt_ci(data['sentiment_end'].get('macro_f1_bootstrap_95ci'), pct=False)} |",
        f"| Sentiment End Within-One | {data['sentiment_end']['within_one_accuracy_percent']}% | — | OrdMAE: {data['sentiment_end']['ordinal_mae']}, QWK: {data['sentiment_end']['quadratic_weighted_kappa']} |",
        f"| Resolution Accuracy | {data['resolution']['accuracy_percent']}% | {_fmt_ci(data['resolution']['accuracy_wilson_95ci'])} | 3-class |",
        f"| Agent Handled Well Accuracy | {data['agent_handled_well']['accuracy_percent']}% | {_fmt_ci(data['agent_handled_well']['accuracy_wilson_95ci'])} | F1 (pos): {data['agent_handled_well']['f1_positive_class']} |",
        f"| Evidence Grounding Rate | {data['grounding']['grounding_rate_percent']}% | {_fmt_ci(data['grounding']['grounding_wilson_95ci'])} | Verbatim transcript match |",
        f"| Resolution-Time MAE | {data['timing']['resolution_time_mae_minutes']} min | {_fmt_ci(data['timing'].get('mae_bootstrap_95ci_minutes'), pct=False)} | {data['timing']['evaluated_calls_count']} resolved calls |",
        f"| Latency (Mean / P95) | {data['performance']['mean_latency_seconds']}s / {data['performance']['p95_latency_seconds']}s | — | Per-call |",
    ]

    audio_res = data.get("audio_transcription", {})
    if audio_res.get("wer_percent") is not None:
        lines.append(
            f"| Audio Subset WER | {audio_res['wer_percent']}% | — | {audio_res['audio_count']} calls via faster-whisper ({audio_res.get('whisper_model')}) |"
        )

    lines.extend([
        "",
        "> **Grounding note**: " + data["grounding"]["note"],
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
    ss = data["sentiment_start"]
    se = data["sentiment_end"]
    res = data["resolution"]
    ah = data["agent_handled_well"]
    gr = data["grounding"]
    ti = data["timing"]
    perf = data["performance"]

    print(f"Sentiment Start Acc     : {ss['accuracy_percent']}%  95%CI={_fmt_ci(ss['accuracy_wilson_95ci'])}  (Macro-F1: {ss['macro_f1']}  95%CI={_fmt_ci(ss.get('macro_f1_bootstrap_95ci'), pct=False)})")
    print(f"  Within-one Acc        : {ss['within_one_accuracy_percent']}%   OrdMAE: {ss['ordinal_mae']}   QWK: {ss['quadratic_weighted_kappa']}")
    print(f"Sentiment End Acc       : {se['accuracy_percent']}%  95%CI={_fmt_ci(se['accuracy_wilson_95ci'])}  (Macro-F1: {se['macro_f1']}  95%CI={_fmt_ci(se.get('macro_f1_bootstrap_95ci'), pct=False)})")
    print(f"  Within-one Acc        : {se['within_one_accuracy_percent']}%   OrdMAE: {se['ordinal_mae']}   QWK: {se['quadratic_weighted_kappa']}")
    print(f"Resolution Acc          : {res['accuracy_percent']}%  95%CI={_fmt_ci(res['accuracy_wilson_95ci'])}")
    print(f"Agent Handled Well Acc  : {ah['accuracy_percent']}%  95%CI={_fmt_ci(ah['accuracy_wilson_95ci'])}  F1: {ah['f1_positive_class']}")
    print(f"Grounding Rate          : {gr['grounding_rate_percent']}%  95%CI={_fmt_ci(gr['grounding_wilson_95ci'])}")
    print(f"Resolution-Time MAE     : {ti['resolution_time_mae_minutes']} min  95%CI={_fmt_ci(ti.get('mae_bootstrap_95ci_minutes'), pct=False)}  ({ti['evaluated_calls_count']} calls)")
    print(f"Mean Latency            : {perf['mean_latency_seconds']}s")
    if data.get("audio_transcription", {}).get("wer_percent") is not None:
        print(f"Audio Subset WER        : {data['audio_transcription']['wer_percent']}%")
    print(f"Failure Cases Count     : {data['failures_count']}")


def check_regression(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    max_acc_drop: float = 5.0,
    max_mae_increase: float = 0.1,
) -> List[str]:
    """Compare current results to baseline; return list of regression messages."""
    regressions = []

    def _check_acc(key_path: List[str], label: str):
        try:
            cur_val = current
            bas_val = baseline
            for k in key_path:
                cur_val = cur_val[k]
                bas_val = bas_val[k]
            drop = bas_val - cur_val
            if drop > max_acc_drop:
                regressions.append(
                    f"REGRESSION [{label}]: {cur_val:.1f}% vs baseline {bas_val:.1f}% "
                    f"(drop {drop:.1f}pp > threshold {max_acc_drop}pp)"
                )
        except (KeyError, TypeError):
            pass

    def _check_mae(key_path: List[str], label: str):
        try:
            cur_val = current
            bas_val = baseline
            for k in key_path:
                cur_val = cur_val[k]
                bas_val = bas_val[k]
            if cur_val is None or bas_val is None:
                return
            increase = cur_val - bas_val
            if increase > max_mae_increase:
                regressions.append(
                    f"REGRESSION [{label}]: {cur_val:.3f} min vs baseline {bas_val:.3f} min "
                    f"(increase {increase:.3f} min > threshold {max_mae_increase} min)"
                )
        except (KeyError, TypeError):
            pass

    _check_acc(["sentiment_start", "accuracy_percent"], "Sentiment Start Accuracy")
    _check_acc(["sentiment_end", "accuracy_percent"], "Sentiment End Accuracy")
    _check_acc(["resolution", "accuracy_percent"], "Resolution Accuracy")
    _check_acc(["agent_handled_well", "accuracy_percent"], "Agent Handled Well Accuracy")
    _check_acc(["grounding", "grounding_rate_percent"], "Grounding Rate")
    _check_mae(["timing", "resolution_time_mae_minutes"], "Resolution Time MAE")

    return regressions


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on Call Center Insights.")
    parser.add_argument("--provider", default="baseline", help="Extractor provider (baseline, ollama, anthropic).")
    parser.add_argument(
        "--split", default="dev",
        choices=["dev", "test", "heldout", "dev+test", "both", "all"],
        help="Split to evaluate. 'both' and 'dev+test' are identical (dev+test). 'all' includes heldout."
    )
    parser.add_argument("--data-dir", default="evals/data", help="Directory of transcript JSON files.")
    parser.add_argument("--audio-dir", default="samples", help="Directory of rendered audio samples.")
    parser.add_argument("--whisper", default="base", help="Whisper model size for audio WER evaluation.")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio WER evaluation.")
    parser.add_argument("--out-json", default="evals/results/latest.json", help="Path for JSON output.")
    parser.add_argument("--out-md", default="evals/results/latest.md", help="Path for Markdown output.")
    parser.add_argument("--bootstrap-n", type=int, default=1000, help="Number of bootstrap resamples.")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for bootstrap.")

    # Quality Gate Thresholds (absolute)
    parser.add_argument("--min-sentiment-start-acc", type=float, default=None)
    parser.add_argument("--min-sentiment-end-acc", type=float, default=None)
    parser.add_argument("--min-resolution-acc", type=float, default=None)
    parser.add_argument("--min-grounding-rate", type=float, default=None)
    parser.add_argument("--max-resolution-time-mae", type=float, default=None)

    # Regression gate vs committed baseline
    parser.add_argument(
        "--baseline-json", default=None,
        help=(
            "Path to a committed baseline JSON (e.g. evals/results/baseline_v0.json). "
            "If provided, the run fails if any metric regresses by more than "
            "--max-acc-drop pp or --max-mae-increase min."
        )
    )
    parser.add_argument("--max-acc-drop", type=float, default=5.0,
                        help="Maximum allowed accuracy drop vs baseline (percentage points).")
    parser.add_argument("--max-mae-increase", type=float, default=0.1,
                        help="Maximum allowed MAE increase vs baseline (minutes).")

    args = parser.parse_args()

    results = run_evaluation(
        provider=args.provider,
        split=args.split,
        data_dir=args.data_dir,
        audio_dir=args.audio_dir,
        whisper_model=args.whisper,
        out_json=args.out_json,
        out_md=args.out_md,
        eval_audio=not args.no_audio,
        n_bootstrap=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    gate_failed = False

    # Absolute threshold gates
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

    # Regression gate vs committed baseline
    if args.baseline_json:
        if not os.path.exists(args.baseline_json):
            print(f"WARNING: --baseline-json path not found: {args.baseline_json}", file=sys.stderr)
        else:
            with open(args.baseline_json, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            regressions = check_regression(
                results, baseline_data,
                max_acc_drop=args.max_acc_drop,
                max_mae_increase=args.max_mae_increase,
            )
            if regressions:
                print("\n--- REGRESSION GATE FAILURES ---", file=sys.stderr)
                for msg in regressions:
                    print(f"  {msg}", file=sys.stderr)
                gate_failed = True
            else:
                print("\nRegression gate passed: no metric degraded by more than the threshold.")

    if gate_failed:
        print("\nQuality gate FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll gating checks passed successfully (exit 0).")
        sys.exit(0)


if __name__ == "__main__":
    main()
