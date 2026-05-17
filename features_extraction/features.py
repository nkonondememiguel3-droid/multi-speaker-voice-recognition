import numpy as np
from scipy.signal import resample_poly
from math import gcd
import logging

logger = logging.getLogger(__name__)

TARGET_SR     = 16000
N_FFT         = 512
HOP_LENGTH    = 160      # 10 ms at 16 kHz
WIN_LENGTH    = 400      # 25 ms at 16 kHz
N_MELS        = 80
FMIN          = 20.0
FMAX          = 7600.0
SVD_THRESHOLD = 0.10
MIN_FRAMES    = 10       # discard spectrograms shorter than this (degenerate segments)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Resample
# ─────────────────────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Polyphase rational resampling to TARGET_SR (16 kHz).

    Uses scipy.signal.resample_poly which applies an anti-aliasing FIR filter
    before downsampling — no spectral aliasing, no quality loss.

    GCD reduction keeps the filter order minimal:
        44100 → 16000 : gcd=100, up=160, down=441
        48000 → 16000 : gcd=16000, up=1,   down=3    (trivial case)
        16000 → 16000 : passthrough, no processing
    """
    if orig_sr == TARGET_SR:
        return audio
    g    = gcd(orig_sr, TARGET_SR)
    up   = TARGET_SR // g
    down = orig_sr   // g
    return resample_poly(audio, up, down).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pre-emphasis
# ─────────────────────────────────────────────────────────────────────────────

def pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    First-order high-pass filter: y[t] = x[t] - coeff * x[t-1]

    Boosts high-frequency energy (formants, fricatives) that is naturally
    attenuated in recorded speech. coeff=0.97 is the standard value in the
    speaker recognition literature.

    np.append preserves the first sample unchanged so the output length
    equals the input length.
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — STFT → power spectrum
# ─────────────────────────────────────────────────────────────────────────────

def stft_power(audio: np.ndarray) -> np.ndarray:
    """
    Compute the short-time power spectrum of the signal.

    Frame the signal with a Hann window (WIN_LENGTH=400 samples = 25 ms),
    advance by HOP_LENGTH (160 samples = 10 ms), zero-pad each frame to
    N_FFT=512, then take the squared magnitude of the one-sided FFT.

    Framing uses np.lib.stride_tricks.as_strided — this creates a VIEW
    into the original array (no copy, no Python loop over frames).

    Center-padding (reflect mode, N_FFT//2 samples each side) matches
    librosa's center=True default so spectrograms are time-aligned with
    the original audio.

    Returns:
        power spectrum, shape (N_FFT//2 + 1, T) = (257, T), float32
    """
    # Center-pad so frame 0 is centered on sample 0
    pad   = N_FFT // 2
    audio = np.pad(audio, pad, mode='reflect')

    # Strided framing — zero-copy view
    n_frames = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH
    shape    = (WIN_LENGTH, n_frames)
    strides  = (audio.strides[0], audio.strides[0] * HOP_LENGTH)
    frames   = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    # Hann window broadcast over all frames simultaneously
    window = np.hanning(WIN_LENGTH).reshape(-1, 1).astype(np.float32)
    frames = frames * window

    # rfft zero-pads each frame from WIN_LENGTH to N_FFT internally
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=0)   # (257, T) complex
    power    = (np.abs(spectrum) ** 2).astype(np.float32)

    return power    # (257, T)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Mel filterbank (built once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    """HTK mel scale: m = 2595 * log10(1 + hz/700)"""
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _build_mel_filterbank() -> np.ndarray:
    """
    Construct a (N_MELS, N_FFT//2+1) triangular Mel filterbank matrix.

    N_MELS+2 points are placed linearly in mel-space between FMIN and FMAX.
    Each filter k draws a triangle between its left, center, and right
    frequency bin — rising slope then falling slope, peak normalized to 1.

    The matrix H is a pure constant; multiplying it by a power spectrum
    P (shape F×T) gives the mel-domain representation: H @ P → (80, T).
    """
    n_fft_bins = N_FFT // 2 + 1                                # 257

    mel_min  = _hz_to_mel(FMIN)
    mel_max  = _hz_to_mel(FMAX)
    mel_pts  = np.linspace(mel_min, mel_max, N_MELS + 2)       # 82 points
    hz_pts   = np.array([_mel_to_hz(m) for m in mel_pts])
    bin_pts  = np.floor((N_FFT + 1) * hz_pts / TARGET_SR).astype(int)

    filterbank = np.zeros((N_MELS, n_fft_bins), dtype=np.float32)

    for k in range(N_MELS):
        left, center, right = bin_pts[k], bin_pts[k + 1], bin_pts[k + 2]
        if center > left:
            filterbank[k, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filterbank[k, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filterbank   # (80, 257)

# Built once, reused for every segment
_MEL_FILTERBANK = _build_mel_filterbank()


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Log-Mel spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute the 80-bin log-Mel spectrogram.

        power  = stft_power(audio)       # (257, T)
        mel    = H @ power               # (80,  T)  — single matrix multiply
        output = log(mel + 1e-6)         # log compression, floor avoids log(0)

    Returns shape (80, T), float32.
    """
    power   = stft_power(audio)                  # (257, T)
    mel     = _MEL_FILTERBANK @ power            # (80,  T)
    log_mel = np.log(mel + 1e-6)
    return log_mel.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — SVD denoising
# ─────────────────────────────────────────────────────────────────────────────

def svd_denoise(spectrogram: np.ndarray, threshold: float = SVD_THRESHOLD) -> np.ndarray:
    """
    Suppress stationary background noise via truncated SVD reconstruction.

    Decompose M = U @ diag(S) @ Vt.
    Singular values encode energy: S[0] is the dominant component (the speech),
    trailing values encode low-energy stationary noise (fan, HVAC).
    We zero out all components below (threshold * S[0]) and reconstruct.

    threshold=0.10 retains components carrying ≥10% of the peak energy,
    which in practice keeps all speech structure while removing noise floor.

    Edge case: always keep at least 1 component (K=max(K,1)) so a silent
    or near-silent segment doesn't collapse to an all-zero matrix.
    """
    U, S, Vt = np.linalg.svd(spectrogram, full_matrices=False)
    K        = max(int(np.sum(S > threshold * S[0])), 1)
    denoised = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
    return denoised.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — CMVN
# ─────────────────────────────────────────────────────────────────────────────

def cmvn(spectrogram: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Per-utterance Cepstral Mean & Variance Normalization.

    For each of the 80 mel bins, subtract its time-mean and divide by its
    time-std across all T frames. This cancels slow-varying channel effects
    (microphone frequency response, room coloration) that differ between
    speakers' recording setups.

    Applied AFTER SVD denoising so that the normalization sees clean energy
    rather than noise-inflated variance in the high-frequency bins.

    eps prevents division by zero on near-silent bins.
    """
    mean = spectrogram.mean(axis=1, keepdims=True)   # (80, 1)
    std  = spectrogram.std(axis=1,  keepdims=True)   # (80, 1)
    return ((spectrogram - mean) / (std + eps)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray | None:
    """
    Full front-end feature extraction pipeline for a single voiced segment.

    Args:
        audio:        1-D float32 array — one voiced segment from run_vad slicing
        sample_rate:  sampling rate of the audio array

    Returns:
        np.ndarray of shape (80, T), float32  — ready for the ECAPA-TDNN encoder
        None if the segment is too short to produce a usable spectrogram
    """
    audio = resample(audio, sample_rate)
    audio = pre_emphasis(audio)
    spec  = log_mel_spectrogram(audio)

    # Guard: drop degenerate segments that survived VAD but are still too short
    if spec.shape[1] < MIN_FRAMES:
        logger.warning(f"Discarding segment: only {spec.shape[1]} frames after STFT (min={MIN_FRAMES})")
        return None

    spec = svd_denoise(spec)
    spec = cmvn(spec)
    return spec     # (80, T)
