import numpy as np
import torch
from typing import Optional

from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000

_silero_model = None


def get_silero_model():
    """
    Always get the same silero model(single module)
    """
    global _silero_model

    if _silero_model is None:
        _silero_model = load_silero_vad()

    return _silero_model


def _silero_vad(
    audio: np.ndarray,
    # sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> list[tuple[int, int]]:
    """
    Run Silero-VAD on a normalized float32 audio array.
    Returns list of (start_ms, end_ms) tuples.
    """
    model = get_silero_model()

    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    # Convert sample indices → milliseconds
    print(timestamps)
    segments = [
        (int(t["start"] / SAMPLE_RATE * 1000),
         int(t["end"] / SAMPLE_RATE * 1000))
        for t in timestamps
    ]

    return segments
