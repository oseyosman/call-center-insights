import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import soundfile as sf


@pytest.mark.slow
def test_cli_smoke_analyze(tmp_path):
    try:
        from faster_whisper import WhisperModel
        try:
            WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
        except Exception:
            WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as exc:
        pytest.skip(f"Whisper model cannot load offline: {exc}")

    out_dir = str(tmp_path / "test_reports")
    single_sample_dir = str(tmp_path / "single_sample")
    os.makedirs(single_sample_dir, exist_ok=True)

    # Generate synthetic stereo audio
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left_channel = 0.1 * np.sin(2 * np.pi * 440 * t)
    right_channel = 0.1 * np.sin(2 * np.pi * 880 * t)
    stereo_data = np.stack([left_channel, right_channel], axis=-1).astype(np.float32)
    sample_wav = os.path.join(single_sample_dir, "sample_synthetic.wav")
    sf.write(sample_wav, stereo_data, sample_rate)

    cmd = [
        sys.executable,
        "-m", "callinsight.cli",
        "analyze", single_sample_dir,
        "--out", out_dir,
        "--whisper", "tiny",
        "--provider", "baseline"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI analyze failed: {res.stderr}\n{res.stdout}"

    # Verify generated files
    csv_path = os.path.join(out_dir, "calls_report.csv")
    json_path = os.path.join(out_dir, "summary_report.json")
    md_path = os.path.join(out_dir, "summary_report.md")

    assert os.path.exists(csv_path), "calls_report.csv must exist"
    assert os.path.exists(json_path), "summary_report.json must exist"
    assert os.path.exists(md_path), "summary_report.md must exist"

    # Verify CSV structure
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert "resolution" in df.columns
    assert "customer_sentiment_start" in df.columns
    assert "evidence" in df.columns
