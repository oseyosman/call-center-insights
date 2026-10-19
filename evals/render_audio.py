"""Render simulated call recordings to stereo audio using local macOS TTS (`say`) and soundfile.

All generated audio is purely SYNTHETIC / SIMULATED for automated pipeline testing.
Left Channel  = Agent (Samantha)
Right Channel = Customer (Daniel / Fred)
"""

import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import soundfile as sf


def render_tts_snippet(text: str, voice: str, temp_dir: str, sample_rate: int = 16000) -> np.ndarray:
    """Generate audio for text using macOS `say` and convert to numpy array at sample_rate."""
    aiff_path = os.path.join(temp_dir, f"temp_{abs(hash(text))}.aiff")
    wav_path = os.path.join(temp_dir, f"temp_{abs(hash(text))}.wav")

    # Call macOS `say`
    cmd_say = ["say", "-v", voice, "-o", aiff_path, text]
    subprocess.run(cmd_say, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Convert AIFF to 16kHz WAV using ffmpeg
    cmd_ffmpeg = [
        "ffmpeg", "-y", "-i", aiff_path,
        "-ar", str(sample_rate), "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd_ffmpeg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    data, _ = sf.read(wav_path)
    return data


def render_call_to_stereo_audio(
    call_record: dict,
    output_path: str,
    sample_rate: int = 16000,
    agent_voice: str = "Samantha",
    customer_voice: str = "Daniel"
) -> str:
    """Render a full call transcript with timestamps into a stereo WAV file.

    Left channel: Agent
    Right channel: Customer
    """
    segments = call_record.get("segments", [])
    if not segments:
        raise ValueError("Cannot render call with empty segments.")

    total_duration = max(s["end"] for s in segments) + 1.0
    total_samples = int(total_duration * sample_rate)

    left_channel = np.zeros(total_samples, dtype=np.float32)
    right_channel = np.zeros(total_samples, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        for seg in segments:
            speaker = seg.get("speaker", "agent")
            text = seg.get("text", "").strip()
            if not text:
                continue

            voice = agent_voice if speaker == "agent" else customer_voice
            audio_snippet = render_tts_snippet(text, voice, tmpdir, sample_rate)

            start_sample = int(seg["start"] * sample_rate)
            end_sample = min(start_sample + len(audio_snippet), total_samples)
            snippet_slice = audio_snippet[: end_sample - start_sample]

            if speaker == "agent":
                left_channel[start_sample:end_sample] += snippet_slice
            else:
                right_channel[start_sample:end_sample] += snippet_slice

    # Normalize to prevent clipping
    max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_val > 0.95:
        left_channel = (left_channel / max_val) * 0.90
        right_channel = (right_channel / max_val) * 0.90

    stereo_audio = np.stack([left_channel, right_channel], axis=-1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, stereo_audio, sample_rate)
    return output_path


def render_audio_dataset(
    transcripts_json: str = "evals/data/dev_transcripts.json",
    output_dir: str = "samples/",
    limit: int = 10
):
    """Render the first `limit` calls from transcripts_json into stereo WAV files."""
    with open(transcripts_json, "r", encoding="utf-8") as f:
        calls = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    selected = calls[:limit]

    print(f"=== Rendering {len(selected)} SIMULATED calls to stereo audio ({output_dir}) ===")
    print("Stereo mapping: Left channel = Agent (Samantha), Right channel = Customer (Daniel)")
    print("NOTICE: All generated audio is synthetic/simulated for testing purposes.\n")

    rendered_files = []
    for idx, call in enumerate(selected, 1):
        call_id = call["call_id"]
        out_file = os.path.join(output_dir, f"{call_id}.wav")
        print(f"[{idx}/{len(selected)}] Rendering {call_id} -> {out_file}...")
        render_call_to_stereo_audio(call, out_file)
        rendered_files.append(out_file)

    print(f"\nSuccessfully rendered {len(rendered_files)} simulated audio files to {output_dir}")
    return rendered_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render simulated call transcripts to stereo audio.")
    parser.add_argument("--input", default="evals/data/dev_transcripts.json", help="Path to transcripts JSON.")
    parser.add_argument("--out", default="samples/", help="Directory to save rendered audio.")
    parser.add_argument("--limit", type=int, default=10, help="Number of calls to render (default: 10).")
    args = parser.parse_args()

    render_audio_dataset(args.input, args.out, limit=args.limit)
