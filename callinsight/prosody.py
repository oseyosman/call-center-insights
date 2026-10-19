"""Optional acoustic prosody extraction using Librosa.

Descriptive features only:
- pitch (F0 mean and standard deviation)
- energy (RMS mean and standard deviation)
- speaking rate estimate (syllables or words per second)

CRITICAL NOTE:
These features are purely descriptive for QA auditing and telemetry.
As per project requirements, prosody features MUST NEVER be used to feed
or alter emotion/sentiment labels. Acoustic features vary widely across microphone
hardware, background noise, accents, and recording codecs, making them prone
to severe bias if used for categorical emotion classification.
"""

from typing import Dict, List, Optional
import numpy as np


def extract_prosody_features(audio_path: str, transcript_word_count: Optional[int] = None) -> Dict[str, float]:
    """Extract descriptive prosody features from an audio file.

    Returns a dictionary of acoustic metrics.
    """
    import soundfile as sf
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        return {"error": f"Failed to load audio: {str(e)}"}

    if len(y) == 0:
        return {
            "duration_seconds": 0.0,
            "mean_pitch_hz": 0.0,
            "std_pitch_hz": 0.0,
            "mean_energy_rms": 0.0,
            "speaking_rate_wps": 0.0,
        }

    duration_sec = len(y) / sr

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))
    std_energy = float(np.std(rms))

    # Fundamental frequency (F0 / Pitch) using pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048
        )
        voiced_f0 = f0[voiced_flag > 0] if voiced_flag is not None else np.array([])
        if len(voiced_f0) > 0 and not np.all(np.isnan(voiced_f0)):
            mean_f0 = float(np.nanmean(voiced_f0))
            std_f0 = float(np.nanstd(voiced_f0))
        else:
            mean_f0 = 0.0
            std_f0 = 0.0
    except Exception:
        mean_f0 = 0.0
        std_f0 = 0.0

    # Speaking rate (words per second)
    speaking_rate = 0.0
    if transcript_word_count and duration_sec > 0:
        speaking_rate = round(transcript_word_count / duration_sec, 2)

    return {
        "duration_seconds": round(duration_sec, 2),
        "mean_pitch_hz": round(mean_f0, 2),
        "std_pitch_hz": round(std_f0, 2),
        "mean_energy_rms": round(mean_energy, 4),
        "std_energy_rms": round(std_energy, 4),
        "speaking_rate_wps": speaking_rate,
    }
