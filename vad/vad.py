import numpy as np
import torch
import webrtcvad

from typing import Optional
from scipy.signal import resample_poly
from math import gcd

from silero_vad import load_silero_vad, get_speech_timestamps

import logging
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WEBRTC_SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
AGGRESSIVENESS = 2

_silero_model = None


def get_silero_model():
    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad()
    return _silero_model


# ─────────────────────────────────────────────────────────────────────────────
# Internal resampler (no librosa dependency in VAD)
# ─────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Silero VAD
# ─────────────────────────────────────────────────────────────────────────────

def _silero_vad(
    audio: np.ndarray,
    sample_rate: int,           # ← restored, no longer commented out
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> list[tuple[int, int]]:
    """
    Run Silero-VAD on a normalized float32 audio array.
    Returns list of (start_ms, end_ms) tuples.

    Silero requires exactly 16 kHz mono float32 input.
    Resamples internally if the incoming rate differs.
    """
    model = get_silero_model()

    # Silero requires exactly 16 kHz — resample if needed
    if sample_rate != SAMPLE_RATE:
        logger.info(
            f"[VAD] Resampling {sample_rate} Hz → {SAMPLE_RATE} Hz for Silero")
        audio = _resample(audio, sample_rate, SAMPLE_RATE)

    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,      # always 16000 — we resampled above
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    segments = [
        (int(t["start"] / SAMPLE_RATE * 1000),
         int(t["end"] / SAMPLE_RATE * 1000))
        for t in timestamps
    ]
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC VAD fallback
# ─────────────────────────────────────────────────────────────────────────────

def _frame_generator(audio_pcm: bytes, frame_duration_ms: int, sample_rate: int):
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
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
    if sample_rate != WEBRTC_SAMPLE_RATE:
        audio = _resample(audio, sample_rate, WEBRTC_SAMPLE_RATE)

    audio_int16 = (audio * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    vad = webrtcvad.Vad(AGGRESSIVENESS)

    frame_labels = []
    for i, frame in enumerate(_frame_generator(pcm_bytes, FRAME_DURATION_MS, WEBRTC_SAMPLE_RATE)):
        start_ms = i * FRAME_DURATION_MS
        try:
            is_speech = vad.is_speech(frame, WEBRTC_SAMPLE_RATE)
        except Exception:
            is_speech = False
        frame_labels.append((start_ms, is_speech))

    if not frame_labels:
        return []

    segments = []
    in_speech = False
    seg_start = 0

    for start_ms, is_speech in frame_labels:
        if is_speech and not in_speech:
            seg_start = start_ms
            in_speech = True
        elif not is_speech and in_speech:
            in_speech = False
            segments.append((seg_start, start_ms))

    if in_speech:
        segments.append((seg_start, frame_labels[-1][0] + FRAME_DURATION_MS))

    padded = [(max(0, s - speech_pad_ms), e + speech_pad_ms)
              for s, e in segments]
    filtered = [(s, e) for s, e in padded if (e - s) >= min_speech_duration_ms]
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _merge_close_segments(
    segments: list[tuple[int, int]],
    min_silence_ms: int = 100,
    min_segment_ms: int = 500,
) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end) < min_silence_ms:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if (e - s) >= min_segment_ms]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_vad(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    min_segment_ms: int = 500,
) -> list[tuple[int, int]]:
    segments = []
    try:
        segments = _silero_vad(
            audio,
            sample_rate,            # ← passed correctly as keyword arg now
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        logger.info(f"[VAD] Silero detected {len(segments)} raw segment(s)")

    except Exception as e:
        logger.warning(
            f"[VAD] Silero failed ({e}), falling back to WebRTC VAD")
        try:
            segments = _webrtc_vad(
                audio, sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )
            logger.info(
                f"[VAD] WebRTC detected {len(segments)} raw segment(s)")
        except Exception as e2:
            logger.error(
                f"[VAD] WebRTC also failed ({e2}). Returning empty list.")
            return []

    final = _merge_close_segments(
        segments,
        min_silence_ms=min_silence_duration_ms,
        min_segment_ms=min_segment_ms,
    )
    logger.info(f"[VAD] After merging/filtering: {len(final)} segment(s)")
    return final
