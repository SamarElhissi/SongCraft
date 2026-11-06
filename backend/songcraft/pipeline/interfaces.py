from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from songcraft.pipeline.beats import LinePlacement
from songcraft.pipeline.melody import Note


class MusicGenerator(Protocol):
    def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
        """Write an instrumental wav to output_path."""


class LyricsGenerator(Protocol):
    def generate(self, prompt: str, line_count: int = 2, syllables: int = 6) -> str:
        """Return lyric text. line_count is how many lines the song has room for."""


class VocalSynthesizer(Protocol):
    def synthesize(self, line: str, notes: list[Note], output_path: Path) -> None:
        """Write one line, following notes when the synthesizer can sing them."""


class Mixer(Protocol):
    def mix(
        self,
        instrumental_path: Path,
        placements: list[LinePlacement],
        vocal_paths: list[Path],
        output_path: Path,
    ) -> None:
        """Overlay the vocal clips on the instrumental and write the final mix."""


class BeatAnalyzer(Protocol):
    def analyze(
        self, audio_path: Path, duration_seconds: float
    ) -> tuple[float | None, list[float]]:
        """Return tempo in BPM and beat times in seconds."""


@dataclass
class Pipeline:
    music: MusicGenerator
    lyrics: LyricsGenerator
    vocals: VocalSynthesizer
    mixer: Mixer
    beats: BeatAnalyzer
