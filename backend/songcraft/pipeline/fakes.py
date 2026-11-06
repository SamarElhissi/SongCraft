import wave
from pathlib import Path

from songcraft.pipeline.beats import LinePlacement

# Placeholder audio stands in for the real models until phase 2.
_MAX_PLACEHOLDER_SECONDS = 1.0


def write_silent_wav(path: Path, duration_seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds = min(max(duration_seconds, 0.05), _MAX_PLACEHOLDER_SECONDS)
    frame_count = max(1, int(sample_rate * seconds))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


class FakeMusicGenerator:
    def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
        write_silent_wav(output_path, duration_seconds)


class FakeLyricsGenerator:
    def generate(self, prompt: str) -> str:
        return f"Walking home in the rain. Thinking about {prompt}."


class FakeVocalSynthesizer:
    def synthesize(self, line: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3" + line.encode("utf-8"))


class FakeMixer:
    def mix(
        self,
        instrumental_path: Path,
        placements: list[LinePlacement],
        vocal_paths: list[Path],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = instrumental_path.read_bytes() if instrumental_path.is_file() else b"mix"
        output_path.write_bytes(payload)


class FakeBeatAnalyzer:
    def analyze(
        self, audio_path: Path, duration_seconds: float
    ) -> tuple[float | None, list[float]]:
        tempo = 120.0
        if duration_seconds <= 0:
            return None, []
        interval = 60.0 / tempo
        times: list[float] = []
        cursor = interval
        while cursor < duration_seconds:
            times.append(round(cursor, 3))
            cursor += interval
        return tempo, times
