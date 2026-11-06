import shutil
from pathlib import Path

from songcraft.pipeline.beats import LinePlacement


class PydubMixer:
    def mix(
        self,
        instrumental_path: Path,
        placements: list[LinePlacement],
        vocal_paths: list[Path],
        output_path: Path,
    ) -> None:
        from pydub import AudioSegment

        _ensure_ffmpeg()
        if not instrumental_path.is_file():
            raise FileNotFoundError(f"Instrumental not found: {instrumental_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        beat = AudioSegment.from_file(instrumental_path)
        vocals = AudioSegment.silent(duration=len(beat))
        for placement, vocal_path in zip(placements, vocal_paths):
            if not vocal_path.is_file():
                continue
            start_ms = int(placement.start * 1000)
            duration_ms = int(placement.end * 1000) - start_ms
            if duration_ms <= 0:
                continue
            vocal = AudioSegment.from_file(vocal_path)[:duration_ms]
            if len(vocal) == 0:
                continue
            fade_ms = min(200, duration_ms // 4)
            if fade_ms > 0:
                vocal = vocal.fade_in(fade_ms).fade_out(fade_ms)
            vocal = vocal - 3
            if len(vocal) < duration_ms:
                vocal += AudioSegment.silent(duration=duration_ms - len(vocal))
            vocals = vocals.overlay(vocal, position=start_ms)
        beat.overlay(vocals).export(output_path, format="mp3")


def _ensure_ffmpeg() -> None:
    from pydub import AudioSegment

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    # This repo keeps a binary in ffmpeg/ and it is not always on PATH.
    ffmpeg = _bundled_tool("ffmpeg")
    ffprobe = _bundled_tool("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg is required to decode speech mp3 and export the mix")
    AudioSegment.converter = str(ffmpeg)
    AudioSegment.ffprobe = str(ffprobe)


def _bundled_tool(name: str) -> Path | None:
    folder = Path(__file__).resolve().parents[3] / "ffmpeg"
    for candidate in (folder / f"{name}.exe", folder / name):
        if candidate.is_file():
            return candidate
    return None
