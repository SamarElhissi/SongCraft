import re
from dataclasses import dataclass
from pathlib import Path

# Krumhansl-Kessler profiles. Index 0 is C.
_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
# Scale degrees: do mi sol la sol mi re do.
_PATTERN = (0, 2, 4, 5, 4, 2, 1, 0)


@dataclass(frozen=True)
class Note:
    midi: int
    duration: float


@dataclass(frozen=True)
class KeyEstimate:
    tonic_midi: int
    scale: tuple[int, ...]


def target_line_count(duration_seconds: float) -> int:
    return max(2, min(8, round(duration_seconds / 4)))


def syllables_per_line(duration_seconds: float, line_count: int) -> int:
    if line_count <= 0 or duration_seconds <= 0:
        return 6
    return max(4, min(8, round(duration_seconds / line_count * 2)))


def words_of(line: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", line)
    return words or ["la"]


def build_line_melody(line: str, duration: float, key: KeyEstimate) -> list[Note]:
    """One note per word, inside one octave of the key, resolving to the tonic."""
    words = words_of(line)
    span = duration if duration > 0 else 0.35 * len(words)
    tonic = 60 + (key.tonic_midi % 12)
    scale = key.scale or MAJOR_SCALE
    notes: list[Note] = []
    cursor = 0.0
    for index in range(len(words)):
        if index == len(words) - 1:
            length = span - cursor
        else:
            length = span / len(words)
        cursor += length
        degree = _PATTERN[index % len(_PATTERN)]
        midi = tonic if index == len(words) - 1 else tonic + scale[degree]
        notes.append(Note(midi=midi, duration=max(length, 0.05)))
    return notes


def estimate_key(audio_path: Path) -> KeyEstimate:
    try:
        return _estimate_key(audio_path)
    except Exception:
        return KeyEstimate(60, MAJOR_SCALE)


def _estimate_key(audio_path: Path) -> KeyEstimate:
    import numpy as np
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True, duration=30)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4:
        return KeyEstimate(60, MAJOR_SCALE)
    chroma = np.mean(librosa.feature.chroma_cqt(y=audio, sr=sample_rate), axis=1)
    best_shift = 0
    best_score = -1e9
    best_scale = MAJOR_SCALE
    for shift in range(12):
        major_score = float(np.dot(chroma, np.roll(_MAJOR_PROFILE, shift)))
        minor_score = float(np.dot(chroma, np.roll(_MINOR_PROFILE, shift)))
        if major_score > best_score:
            best_score = major_score
            best_shift = shift
            best_scale = MAJOR_SCALE
        if minor_score > best_score:
            best_score = minor_score
            best_shift = shift
            best_scale = MINOR_SCALE
    return KeyEstimate(60 + best_shift, best_scale)
