# debug_vad.py  — run this once: python3 debug_vad.py

import numpy as np
import torch
import librosa
from silero_vad import load_silero_vad, get_speech_timestamps

# Load the converted WAV directly — bypass your ingestion stack
audio, sr = librosa.load("./datasets/temp_converted/temp/miguel_audio.wav", sr=None)

print(f"Sample rate     : {sr} Hz")
print(f"Duration        : {len(audio)/sr:.2f} s")
print(f"Shape           : {audio.shape}")
print(f"dtype           : {audio.dtype}")
print(f"Peak amplitude  : {np.max(np.abs(audio)):.4f}")
print(f"RMS energy      : {np.sqrt(np.mean(audio**2)):.4f}")
print(f"Silent frames % : {np.mean(np.abs(audio) < 0.01) * 100:.1f}%")

# Try Silero with a very permissive threshold
model = load_silero_vad()
tensor = torch.from_numpy(audio.astype(np.float32))

for threshold in [0.5, 0.3, 0.1]:
    ts = get_speech_timestamps(
        tensor, model,
        threshold=threshold,
        sampling_rate=sr,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
        return_seconds=True,
    )
    print(f"threshold={threshold} → {len(ts)} segment(s): {ts[:3]}")
