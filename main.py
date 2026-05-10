import os
import atexit

from ingestion_normalization.ingestion import loads
from vad.vad import _silero_vad


def restore_terminal():
    os.system("stty sane")


AUDIO_FILES_PATH = "./datasets/temp/"

if __name__ == "__main__":
    atexit.register(restore_terminal)

    normalized_audio = [audio_file for audio_file in loads(
        AUDIO_FILES_PATH) if audio_file is not None]

    print(normalized_audio[0][0])
    print(_silero_vad(normalized_audio[0][0]))

    print("DONE")
