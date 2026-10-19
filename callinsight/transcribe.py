"""Speech-to-text module using faster-whisper with stereo channel separation support."""

import os
import tempfile
from typing import List, Optional, Tuple
import soundfile as sf
import numpy as np

from callinsight.schema import Segment

_MODEL_CACHE = {}


def get_whisper_model(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """Load or retrieve cached faster-whisper model."""
    from faster_whisper import WhisperModel

    cache_key = (model_size, device, compute_type)
    if cache_key not in _MODEL_CACHE:
        # Use CPU with int8 quantization for fast, lightweight local execution on macOS
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    return _MODEL_CACHE[cache_key]


def is_audio_stereo(audio_path: str) -> Tuple[bool, int, Optional[np.ndarray]]:
    """Inspect audio file channels.

    Returns (is_stereo, sample_rate, audio_data).
    """
    try:
        data, sr = sf.read(audio_path)
        is_stereo = data.ndim == 2 and data.shape[1] >= 2
        return is_stereo, sr, data
    except Exception:
        return False, 16000, None


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
) -> Tuple[List[Segment], bool, float]:
    """Transcribe an audio file using faster-whisper.

    Features:
    - If stereo: left channel is treated as 'agent' and right channel as 'customer' with 1.0 confidence.
    - If mono: speaker roles are marked as 'inferred' with lower confidence (0.50).

    Returns:
      (segments, is_stereo, speaker_diarization_confidence)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model(model_size=model_size, device=device, compute_type=compute_type)

    is_stereo, sr, audio_data = is_audio_stereo(audio_path)

    if is_stereo and audio_data is not None:
        # Left channel = Agent, Right channel = Customer
        segments: List[Segment] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_wav = os.path.join(tmpdir, "agent_channel.wav")
            customer_wav = os.path.join(tmpdir, "customer_channel.wav")

            # Extract left channel (agent)
            sf.write(agent_wav, audio_data[:, 0], sr)
            # Extract right channel (customer)
            sf.write(customer_wav, audio_data[:, 1], sr)

            # Transcribe Agent
            agent_segs, _ = model.transcribe(agent_wav, language="en", beam_size=1)
            for s in agent_segs:
                txt = s.text.strip()
                if txt:
                    segments.append(Segment(
                        start=round(s.start, 2),
                        end=round(s.end, 2),
                        speaker="agent",
                        text=txt,
                        confidence=1.0
                    ))

            # Transcribe Customer
            cust_segs, _ = model.transcribe(customer_wav, language="en", beam_size=1)
            for s in cust_segs:
                txt = s.text.strip()
                if txt:
                    segments.append(Segment(
                        start=round(s.start, 2),
                        end=round(s.end, 2),
                        speaker="customer",
                        text=txt,
                        confidence=1.0
                    ))

        # Sort segments chronologically
        segments.sort(key=lambda s: s.start)
        return segments, True, 1.0

    else:
        # Mono recording: cannot separate channels definitively
        raw_segs, _ = model.transcribe(audio_path, language="en", beam_size=1)
        segments = []
        for s in raw_segs:
            txt = s.text.strip()
            if txt:
                segments.append(Segment(
                    start=round(s.start, 2),
                    end=round(s.end, 2),
                    speaker="inferred",
                    text=txt,
                    confidence=0.5
                ))

        return segments, False, 0.50
