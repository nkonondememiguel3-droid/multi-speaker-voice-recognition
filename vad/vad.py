import numpy as np
import torch
import webrtcvad
import struct

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
    segments = [
        (int(t["start"] / SAMPLE_RATE * 1000),
         int(t["end"] / SAMPLE_RATE * 1000))
        for t in timestamps
    ]

    return segments


WEBRTC_SAMPLE_RATE = 16000   # webrtcvad also requires 16 kHz
FRAME_DURATION_MS = 30      # frame size: 10, 20, or 30 ms only
AGGRESSIVENESS = 2       # 0 (least) to 3 (most aggressive filtering)


def _frame_generator(audio_pcm: bytes, frame_duration_ms: int, sample_rate: int):
    """
    Yield fixed-duration frames of raw PCM bytes from audio_pcm.
    Each frame is frame_duration_ms milliseconds long.
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000) * \
        2  # 2 bytes per int16 sample
    offset = 0
    while offset + frame_size <= len(audio_pcm):
        yield audio_pcm[offset:offset + frame_size]
        offset += frame_size


def _webrtc_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> list[tuple[int, int]]:
    """
    Fallback VAD using WebRTC VAD v3.
    Returns list of (start_ms, end_ms) tuples.
    """
    import librosa

    # Resample to 16 kHz if needed
    if sample_rate != WEBRTC_SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=WEBRTC_SAMPLE_RATE)

    # WebRTC expects int16 PCM bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    vad = webrtcvad.Vad(AGGRESSIVENESS)

    # Classify each frame
    frame_labels = []   # list of (start_ms, is_speech)
    for i, frame in enumerate(_frame_generator(pcm_bytes, FRAME_DURATION_MS, WEBRTC_SAMPLE_RATE)):
        start_ms = i * FRAME_DURATION_MS
        try:
            is_speech = vad.is_speech(frame, WEBRTC_SAMPLE_RATE)
        except Exception:
            is_speech = False
        frame_labels.append((start_ms, is_speech))

    # Merge consecutive speech frames into segments
    segments = []
    in_speech = False
    seg_start = 0

    for start_ms, is_speech in frame_labels:
        if is_speech and not in_speech:
            seg_start = start_ms
            in_speech = True
        elif not is_speech and in_speech:
            seg_end = start_ms
            in_speech = False
            segments.append((seg_start, seg_end))

    # Close any open segment at end of audio
    if in_speech:
        segments.append((seg_start, frame_labels[-1][0] + FRAME_DURATION_MS))

    # Apply padding and duration filters (mirrors Silero's behavior)
    padded = []
    for start, end in segments:
        padded.append((max(0, start - speech_pad_ms), end + speech_pad_ms))

    # Filter out segments that are too short
    filtered = [(s, e) for s, e in padded if (e - s) >= min_speech_duration_ms]

    return filtered


def _merge_close_segments(
    segments: list[tuple[int, int]],
    min_silence_ms: int = 100,
    min_segment_ms: int = 500,
) -> list[tuple[int, int]]:
    """
    Merge segments separated by less than min_silence_ms.
    Then discard segments shorter than min_segment_ms.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end) < min_silence_ms:
            merged[-1] = (prev_start, end)   # extend previous segment
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if (e - s) >= min_segment_ms]


def run_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    min_segment_ms: int = 500,
) -> list[tuple[int, int]]:
    """
    Run VAD on a normalized float32 audio array.

    Tries Silero-VAD first; falls back to WebRTC VAD on any failure.

    Args:
        audio:                  1-D float32 NumPy array (output of load_normalize)
        sample_rate:            Original sampling rate of the audio
        min_speech_duration_ms: Minimum duration for a frame to be kept
        min_silence_duration_ms:Gap smaller than this causes two segments to merge
        speech_pad_ms:          Padding added on each side of a segment
        min_segment_ms:         Segments shorter than this are discarded entirely

    Returns:
        List of (start_ms, end_ms) tuples.
    """
    segments = []

    try:
        segments = _silero_vad(
            audio, sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        print(f"[VAD] Silero detected {len(segments)} raw segment(s)")

    except Exception as e:
        print(f"[VAD] Silero failed ({e}), falling back to WebRTC VAD")
        try:
            segments = _webrtc_vad(
                audio, sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )
            print(f"[VAD] WebRTC detected {len(segments)} raw segment(s)")
        except Exception as e2:
            print(
                f"[VAD] WebRTC also failed ({e2}). Returning empty segment list.")
            return []

    final = _merge_close_segments(segments, min_silence_ms=min_silence_duration_ms,
                                  min_segment_ms=min_segment_ms)
    print(f"[VAD] After merging/filtering: {len(final)} segment(s)")
    return final
