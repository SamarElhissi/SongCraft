from pathlib import Path

from songcraft.config import Settings


def song_dir(settings: Settings, song_id: str) -> Path:
    return settings.data_dir / "songs" / song_id


def instrumental_path(settings: Settings, song_id: str) -> Path:
    return song_dir(settings, song_id) / "instrumental.wav"


def final_path(settings: Settings, song_id: str) -> Path:
    return song_dir(settings, song_id) / "final.mp3"


def vocal_path(settings: Settings, song_id: str, index: int) -> Path:
    return song_dir(settings, song_id) / "vocals" / f"line_{index}.mp3"
