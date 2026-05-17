import os
import csv
import random
import numpy as np
from pathlib import Path

SPEC_DIR = "./data/spectrograms"
MANIFEST_DIR = "./data/manifests"
TRAIN_RATIO = 0.9


def build_manifests(spectrograms: list[tuple[np.ndarray, str]]):
    """
    Save spectrograms to disk as .npy files and write train/dev manifests.

    Args:
        spectrograms: list of (spec, source_audio_path) from main.py
                      spec shape: (80, T)

    The speaker_id is inferred from the parent folder name of the source
    audio file — assumes your dataset is organised as:
        datasets/temp/<speaker_name>/audio_file.aac

    Auxiliary labels (gender, language, emotion) default to 0 here.
    Replace with real labels once you have them, or leave as 0 during
    VoxCeleb pre-training where only speaker_id matters.
    """
    os.makedirs(SPEC_DIR,     exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    # Build speaker → integer ID mapping
    speaker_names = sorted(
        {Path(path).parent.name for _, path in spectrograms})
    speaker_to_id = {name: i for i, name in enumerate(speaker_names)}

    rows = []
    for i, (spec, source_path) in enumerate(spectrograms):
        speaker_name = Path(source_path).parts[-3]
        speaker_id = speaker_to_id[speaker_name]

        # Save spectrogram
        spec_filename = f"{speaker_name}_{i:06d}.npy"
        spec_path = os.path.join(SPEC_DIR, spec_filename)
        np.save(spec_path, spec)

        rows.append({
            'path':       spec_path,
            'speaker_id': speaker_id,
            'gender':     0,     # replace with real label
            'language':   0,     # replace with real label
            'emotion':    0,     # replace with real label
        })

    # Shuffle then split
    random.shuffle(rows)
    split = int(len(rows) * TRAIN_RATIO)
    train_rows = rows[:split]
    dev_rows = rows[split:]

    def write_csv(path, data):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['path', 'speaker_id', 'gender', 'language', 'emotion'])
            writer.writeheader()
            writer.writerows(data)

    train_path = os.path.join(MANIFEST_DIR, "train.csv")
    dev_path = os.path.join(MANIFEST_DIR, "dev.csv")

    write_csv(train_path, train_rows)
    write_csv(dev_path,   dev_rows)

    print(f"Manifests written:")
    print(f"  train: {len(train_rows)} utterances → {train_path}")
    print(f"  dev:   {len(dev_rows)} utterances  → {dev_path}")
    print(f"  speakers: {len(speaker_names)} → {speaker_to_id}")

    return train_path, dev_path, speaker_to_id
