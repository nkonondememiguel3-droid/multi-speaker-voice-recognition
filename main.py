# At the top of main.py
import numpy as np
import os
import csv
import logging
import atexit
from pathlib import Path
from multiprocessing import Pool, cpu_count

from features_extraction.features import extract_features
from ingestion_normalization.ingestion import load_audio_files, load_normalize
from vad.vad import run_vad
from encoder.train import train

AUDIO_FILES_PATH = "./datasets/librispeech/LibriSpeech/train-clean-100/"
SPEC_DIR = "./data/spectrograms"
MANIFEST_DIR = "./data/manifests"
# N_WORKERS = max(1, cpu_count() // 2)   # use half your cores
N_WORKERS =  6


def restore_terminal():
    os.system("stty sane")


def process_one_file(audio_path: str) -> list[dict]:
    """
    Process a single audio file end-to-end.
    Returns a list of row dicts (one per voiced segment).
    Runs in a worker process — no shared state.
    """
    # Silence per-file logs in workers
    logging.getLogger("ingestion_normalization.ingestion").setLevel(
        logging.WARNING)
    logging.getLogger("vad.vad").setLevel(logging.WARNING)

    result = load_normalize(audio_path)
    if result is None:
        return []

    audio, sr, path = result
    segments = run_vad(audio, sr)

    if not segments:
        del audio
        return []

    rows = []
    speaker_name = Path(audio_path).parts[-3]

    for start_ms, end_ms in segments:
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment_audio = audio[start_sample:end_sample].copy()

        spec = extract_features(segment_audio, sr)
        del segment_audio

        if spec is None:
            continue

        rows.append({
            'speaker_name': speaker_name,
            'spec':         spec,         # carry the array back to main process
        })

    del audio
    return rows


if __name__ == "__main__":
    atexit.register(restore_terminal)

    os.makedirs(SPEC_DIR,     exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    audio_files = load_audio_files(AUDIO_FILES_PATH)
    print(
        f"Found {len(audio_files)} audio files — processing with {N_WORKERS} workers")

    train_rows = []
    dev_rows = []
    n_saved = 0
    n_failed = 0

    # Process in parallel chunks of 200 files at a time
    # Chunking avoids sending all 28k file paths to workers at once
    CHUNK = 200

    for chunk_start in range(0, len(audio_files), CHUNK):
        chunk = audio_files[chunk_start:chunk_start + CHUNK]

        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_one_file, chunk)

        # Save spectrograms from this chunk
        for file_rows in results:
            if not file_rows:
                n_failed += 1
                continue
            for r in file_rows:
                fname = f"{r['speaker_name']}_{n_saved:06d}.npy"
                spec_path = os.path.join(SPEC_DIR, fname)
                np.save(spec_path, r['spec'])

                row = {
                    'path':       spec_path,
                    'speaker_id': r['speaker_name'],
                    'gender':     0,
                    'language':   0,
                    'emotion':    0,
                }
                if (n_saved % 10) == 0:
                    dev_rows.append(row)
                else:
                    train_rows.append(row)

                n_saved += 1

        print(f"  Processed {min(chunk_start + CHUNK, len(audio_files))}/{len(audio_files)} files | "
              f"{n_saved} segments saved")

    print(f"\nIngestion complete: {n_saved} segments, {n_failed} failed")

    # ── Remap speaker names to integer IDs ───────────────────────────────────
    speaker_names = sorted({r['speaker_id'] for r in train_rows + dev_rows})
    speaker_to_id = {name: i for i, name in enumerate(speaker_names)}

    for r in train_rows:
        r['speaker_id'] = speaker_to_id[r['speaker_id']]
    for r in dev_rows:
        r['speaker_id'] = speaker_to_id[r['speaker_id']]

    # ── Write manifests ───────────────────────────────────────────────────────
    def write_csv(path, rows):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['path', 'speaker_id', 'gender', 'language', 'emotion'])
            writer.writeheader()
            writer.writerows(rows)

    train_manifest = os.path.join(MANIFEST_DIR, "train.csv")
    dev_manifest = os.path.join(MANIFEST_DIR, "dev.csv")
    write_csv(train_manifest, train_rows)
    write_csv(dev_manifest,   dev_rows)

    print(f"Manifests written:")
    print(f"  train : {len(train_rows)} segments → {train_manifest}")
    print(f"  dev   : {len(dev_rows)} segments  → {dev_manifest}")
    print(f"  speakers: {len(speaker_names)}")

    n_speakers = len(speaker_names)

    # ── Train ─────────────────────────────────────────────────────────────────
    model, history = train(
        manifest_train=train_manifest,
        manifest_dev=dev_manifest,
        checkpoint_dir="./checkpoints",
        n_speakers=n_speakers,
        n_epochs=50,
        eer_eval_every=5,
    )

    print("DONE")
