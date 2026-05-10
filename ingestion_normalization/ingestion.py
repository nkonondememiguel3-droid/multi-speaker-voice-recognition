# ingestion_normalization/ingestion.py

import os
import csv
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple
import subprocess
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.mp4', '.mkv', '.wav',
                        '.flac', '.mp3', '.ogg', '.aac', '.m4a'}
FAILURE_LOG_PATH = "./datasets/failed_files.csv"


def load_audio_files(path: str) -> list[str]:
    """
    Recursively load all audio files under path.
    Skips files with unsupported extensions and logs them.
    """
    audios: list[str] = []
    skipped: list[str] = []

    for root, _, files in os.walk(path):
        for file in files:
            audio_path = os.path.join(root, file)
            ext = Path(audio_path).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                audios.append(audio_path)
            else:
                skipped.append(audio_path)

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} unsupported file(s): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    logger.info(f"Found {len(audios)} supported audio file(s) under '{path}'")
    return audios


def _probe_file(audio_path: str) -> Optional[str]:
    """
    Run ffprobe to check the file is readable before attempting conversion.
    Returns the detected format string on success, None if the file is bad.
    """
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',           # first audio stream only
        '-show_entries', 'stream=codec_name,duration',
        '-of', 'default=noprint_wrappers=1',
        audio_path
    ]
    try:
        result = subprocess.run(
            probe_cmd, check=True, capture_output=True, timeout=15
        )
        output = result.stdout.decode().strip()
        if not output:
            logger.warning(f"No audio stream found in: {audio_path}")
            return None
        return output
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out on: {audio_path}")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(
            f"ffprobe rejected '{audio_path}': {e.stderr.decode().splitlines()[-1]}")
        return None


def _log_failure(audio_path: str, reason: str):
    """Append a failed file entry to the CSV failure log."""
    file_exists = os.path.exists(FAILURE_LOG_PATH)
    with open(FAILURE_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "reason"])
        writer.writerow([audio_path, reason])


def extract_audio_track(audio_path: str) -> Optional[str]:
    """
    Convert an audio/video file to a 16 kHz mono PCM WAV using FFmpeg.
    Validates the file with ffprobe first to catch corrupt files early
    and give a clear error rather than a cryptic FFmpeg message.
    """
    probe_result = _probe_file(audio_path)
    if probe_result is None:
        reason = "ffprobe validation failed (corrupt, incomplete, or no audio stream)"
        logger.error(f"Skipping '{audio_path}': {reason}")
        _log_failure(audio_path, reason)
        return None

    parent = os.path.join("./datasets/temp_converted/",
                          Path(audio_path).parent.name)
    output_audio_file = os.path.join(parent, Path(audio_path).stem + ".wav")
    os.makedirs(parent, exist_ok=True)

    command = [
        'ffmpeg',
        '-y',
        '-i', audio_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        output_audio_file
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        logger.info(f"Converted: {audio_path} → {output_audio_file}")
        return output_audio_file

    except subprocess.TimeoutExpired:
        reason = "ffmpeg conversion timed out (>120s)"
        logger.error(f"'{audio_path}': {reason}")
        _log_failure(audio_path, reason)
        return None

    except subprocess.CalledProcessError as e:
        # Extract only the last meaningful line from stderr
        stderr_lines = e.stderr.decode().splitlines()
        last_error = next(
            (l for l in reversed(stderr_lines)
             if l.strip() and not l.startswith('  ')),
            "unknown ffmpeg error"
        )
        reason = f"ffmpeg error: {last_error}"
        logger.error(f"'{audio_path}': {reason}")
        _log_failure(audio_path, reason)
        return None


def load_normalize(path: str) -> Optional[Tuple[np.ndarray, int, str]]:
    """
    Convert, load, and normalize a single audio file.

    Returns:
        (audio_array, sample_rate, source_path) on success
        None on any failure

    Note: now returns source_path as third element so callers
    can track which file each array came from.
    """
    converted_audio = extract_audio_track(path)
    if converted_audio is None:
        return None

    try:
        audio_time_serie, sampling_rate = librosa.load(
            converted_audio, sr=None)
    except Exception as e:
        reason = f"librosa load failed: {e}"
        logger.error(f"'{path}': {reason}")
        _log_failure(path, reason)
        return None

    # Duration gate: discard clips under 1.5 s (too short for the encoder)
    if len(audio_time_serie) / sampling_rate < 1.5:
        reason = f"audio too short ({len(audio_time_serie) / sampling_rate:.2f}s < 1.5s)"
        logger.warning(f"Discarding '{path}': {reason}")
        _log_failure(path, reason)
        return None

    max_val = np.max(np.abs(audio_time_serie))
    if max_val == 0:
        reason = "silent audio (all zeros)"
        logger.warning(f"Discarding '{path}': {reason}")
        _log_failure(path, reason)
        return None

    normalized_audio = 0.708 * audio_time_serie / max_val
    return normalized_audio, sampling_rate, path


def loads(path: str) -> Tuple[list, list]:
    """
    Process all audio files under path in parallel.

    Returns:
        (results, failed_paths)
        results:      list of (audio, sr, path) — successful conversions only
        failed_paths: list of paths that returned None
    """
    audio_files_paths = load_audio_files(path)

    with Pool(processes=2) as pool:
        raw_results = pool.map(load_normalize, audio_files_paths)

    results = [r for r in raw_results if r is not None]
    failed_paths = [audio_files_paths[i]
                    for i, r in enumerate(raw_results) if r is None]

    logger.info(
        f"Ingestion complete: {len(results)} succeeded, {len(failed_paths)} failed")
    if failed_paths:
        logger.warning(f"Failed files logged to: {FAILURE_LOG_PATH}")

    return results, failed_paths
