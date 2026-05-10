import os
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple
import subprocess
import librosa
import numpy as np


def load_audio_files(path: str) -> list[str]:
    """
    This function is going to load all the  audio contained inside the 'path' directory recursively.
    """
    audios: list[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            audio_path = os.path.join(root, file)
            audios.append(audio_path)

    return audios


def extract_audio_track(audio_path: str):
    """
    This funciton is going to use ffmpeg to extract audio track from an audio file.
    """

    parent = os.path.join("./datasets/temp_converted/",
                          Path(audio_path).parent.name)
    output_audio_file = os.path.join(parent, Path(audio_path).stem + ".wav")

    os.makedirs(parent, exist_ok=True)

    command = [
        'ffmpeg',
        '-y',                       # overwrite the file if it exists
        '-i', audio_path,           # input audio file
        '-vn',                      # disable video
        '-acodec', 'pcm_s16le',     # set codec to PCM 16-bit
        '-ar', '16000',             # set sampling rate (16000 Hz)
        '-ac', '1',                 # set channels (1 for )
        output_audio_file           # output file
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Audio extracted successfully to {output_audio_file}")

        return output_audio_file

    except subprocess.CalledProcessError as e:
        print(f"Error {e.stderr.decode()}")

        return None


def load_normalize(path: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    This function is going to load an audio, convert it to wav file, normalize it and return it.
    """
    converted_audio = extract_audio_track(path)
    if converted_audio is not None:
        audio_time_serie, sampling_rate = librosa.load(
            converted_audio, sr=None)

        max_val = np.max(np.abs(audio_time_serie))
        if max_val == 0:
            return audio_time_serie, sampling_rate

        normalized_audio = 0.708 * audio_time_serie / max_val

        return normalized_audio, sampling_rate
    else:
        return None


def loads(path: str):
    audio_files_paths = load_audio_files(path)

    # with Pool(processes=os.cpu_count() // 2 or 1) as pool:
    with Pool(processes=2) as pool:
        results = pool.map(load_normalize, audio_files_paths)
    return results
