import os
import atexit
from ingestion_normalization.ingestion import loads
from vad.vad import run_vad


def restore_terminal():
    os.system("stty sane")


AUDIO_FILES_PATH = "./datasets/temp/"

if __name__ == "__main__":
    atexit.register(restore_terminal)

    results, failed = loads(AUDIO_FILES_PATH)

    normalized_audios = [(audio, sr, path) for audio, sr, path in results]

    print(f"DONE — {len(normalized_audios)} files loaded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for p in failed:
            print(f"  ✗ {p}")

    # hold (segment_audio, sr, source_path) per voiced segment
    all_segments = []

    for audio, sr, path in normalized_audios:
        segments = run_vad(audio, sr)

        if not segments:
            print(f"  [VAD] No voiced segments found in: {path}")
            continue

        for start_ms, end_ms in segments:
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            segment_audio = audio[start_sample:end_sample]
            all_segments.append((segment_audio, sr, path))

    print(
        f"DONE — {len(all_segments)} voiced segment(s) extracted across {len(normalized_audios)} file(s)")
