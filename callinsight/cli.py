"""Command-line interface for Call Center Insights.

Usage:
    python -m callinsight.cli analyze <folder> --out reports/ --whisper base [--prosody] [--provider baseline]
"""

import argparse
import glob
import os
import sys
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from callinsight.extract import analyze_transcript
from callinsight.prosody import extract_prosody_features
from callinsight.report import save_reports
from callinsight.schema import CallAnalysisResult, Segment
from callinsight.transcribe import transcribe_audio


def find_supported_files(folder: str) -> List[str]:
    """Find audio or transcript json files in target folder."""
    exts = ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.json")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    # Deduplicate and sort
    return sorted(list(set(files)))


def process_audio_file(
    file_path: str,
    whisper_model: str = "base",
    provider: str = "baseline",
    include_prosody: bool = False
) -> CallAnalysisResult:
    """Process a single audio recording through transcription, redaction, and extraction."""
    call_id = os.path.splitext(os.path.basename(file_path))[0]

    # 1. Transcribe
    segments, is_stereo, diarization_conf = transcribe_audio(
        file_path, model_size=whisper_model
    )

    # 2. Optional Prosody
    prosody_stats = None
    if include_prosody:
        word_count = sum(len(s.text.split()) for s in segments)
        prosody_stats = extract_prosody_features(file_path, transcript_word_count=word_count)

    # 3. Analyze (Redaction + Extraction + Timing)
    result = analyze_transcript(
        call_id=call_id,
        segments=segments,
        provider=provider,
        is_stereo=is_stereo,
        speaker_diarization_confidence=diarization_conf,
        audio_file=file_path,
        prosody_stats=prosody_stats
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="callinsight",
        description="Local call-center audio insights pipeline."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze call audio recordings in a directory.")
    analyze_parser.add_argument("folder", help="Directory containing audio files (.wav, .mp3, etc.)")
    analyze_parser.add_argument("--out", default="reports/", help="Output directory for reports (default: reports/)")
    analyze_parser.add_argument("--whisper", default="base", help="Whisper model size (tiny/base/small, default: base)")
    analyze_parser.add_argument("--provider", default=None, help="LLM extraction provider (baseline/ollama/anthropic)")
    analyze_parser.add_argument("--prosody", action="store_true", help="Extract descriptive librosa acoustic features.")

    args = parser.parse_args()

    if args.subcommand == "analyze":
        target_dir = args.folder
        if not os.path.exists(target_dir):
            print(f"Error: Target directory '{target_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)

        files = find_supported_files(target_dir)
        # Filter for audio files
        audio_files = [f for f in files if not f.endswith(".json")]

        if not audio_files:
            print(f"No audio recordings found in '{target_dir}'.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(audio_files)} audio file(s) in '{target_dir}'.")
        print(f"Configuration: whisper={args.whisper}, provider={args.provider or 'default (baseline)'}, prosody={args.prosody}")
        print("Note: All demo/eval data is SIMULATED.\n")

        results: List[CallAnalysisResult] = []
        for idx, af in enumerate(audio_files, 1):
            print(f"[{idx}/{len(audio_files)}] Processing: {os.path.basename(af)}...")
            try:
                res = process_audio_file(
                    af,
                    whisper_model=args.whisper,
                    provider=args.provider,
                    include_prosody=args.prosody
                )
                results.append(res)
                print(
                    f"    Done: Resolution={res.extraction.resolution}, "
                    f"Sentiment={res.extraction.customer_sentiment_start}->{res.extraction.customer_sentiment_end}, "
                    f"Grounded={res.grounded}"
                )
            except Exception as e:
                print(f"    Error processing {af}: {e}", file=sys.stderr)

        if not results:
            print("No calls were successfully processed.", file=sys.stderr)
            sys.exit(1)

        csv_p, json_p, md_p = save_reports(results, args.out)
        print("\nAnalysis Complete!")
        print(f"CSV report written to: {csv_p}")
        print(f"Summary JSON written to: {json_p}")
        print(f"Markdown summary written to: {md_p}")


if __name__ == "__main__":
    main()
