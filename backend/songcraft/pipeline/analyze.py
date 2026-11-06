import math
from pathlib import Path


class LibrosaBeatAnalyzer:
    def analyze(
        self, audio_path: Path, duration_seconds: float
    ) -> tuple[float | None, list[float]]:
        import librosa
        import numpy as np

        if not audio_path.is_file():
            raise FileNotFoundError(f"Instrumental not found: {audio_path}")
        samples, sample_rate = librosa.load(str(audio_path), sr=None)
        if samples.size == 0 or sample_rate <= 0:
            return None, []
        tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        times = [
            float(value)
            for value in np.asarray(beat_times, dtype=float).reshape(-1).tolist()
            if math.isfinite(float(value))
        ]
        if duration_seconds > 0:
            times = [value for value in times if 0 <= value <= duration_seconds]
        return _tempo_bpm(tempo), times


def _tempo_bpm(tempo) -> float | None:
    import numpy as np

    values = np.asarray(tempo, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    bpm = float(values[0])
    # librosa reports 0 when it cannot find onsets.
    if not math.isfinite(bpm) or bpm <= 0:
        return None
    return bpm
