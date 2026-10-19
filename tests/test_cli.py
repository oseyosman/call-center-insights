import os
import shutil
import subprocess
import pytest
import pandas as pd


def test_cli_smoke_analyze(tmp_path):
    out_dir = str(tmp_path / "test_reports")
    samples_dir = "samples"

    # Ensure samples dir exists and has at least one audio file
    assert os.path.exists(samples_dir), "samples/ dir must exist"
    wav_files = [f for f in os.listdir(samples_dir) if f.endswith(".wav")]
    assert len(wav_files) > 0, "samples/ must contain rendered wav files"

    # Run CLI analyze on a single audio sample directory or subset
    single_sample_dir = str(tmp_path / "single_sample")
    os.makedirs(single_sample_dir, exist_ok=True)
    shutil.copy(os.path.join(samples_dir, wav_files[0]), os.path.join(single_sample_dir, wav_files[0]))

    cmd = [
        sys_python := os.sys.executable,
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
