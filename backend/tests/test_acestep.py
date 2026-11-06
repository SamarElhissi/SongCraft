from pathlib import Path

from songcraft.db import Database
from songcraft.models import Song, SongStatus
from songcraft.paths import final_path, instrumental_path
from songcraft.pipeline.ace_song import AceStepGenerator, acestep_duration
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.runner import run_song


def test_duration_is_at_least_ten_seconds() -> None:
    assert acestep_duration(8) == 10
    assert acestep_duration(30) == 30
    assert acestep_duration(900) == 600


def test_acestep_builds_without_starting_a_process(settings_factory) -> None:
    pipeline = build_pipeline(settings_factory(music_generator="acestep"))
    assert isinstance(pipeline.music, AceStepGenerator)


def test_acestep_saves_the_mix_and_skips_the_vocal_steps(settings_factory, monkeypatch) -> None:
    settings = settings_factory(music_generator="acestep")
    database = Database(settings)
    database.create_all(settings)
    song_id = "ace-song"
    pipeline = build_pipeline(settings)

    def generate_song(self, prompt: str, duration_seconds: float, output_path: Path) -> dict:
        assert prompt == "rain"
        assert duration_seconds == 8
        output_path.write_bytes(b"ID3-ace")
        return {"lyrics": "Rain on the glass", "bpm": 96, "duration": 10.0}

    monkeypatch.setattr(AceStepGenerator, "generate_song", generate_song)
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="rain",
                duration_seconds=8,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)
        song = session.get(Song, song_id)

    assert song is not None
    assert song.status == SongStatus.completed.value
    assert song.lyrics == "Rain on the glass"
    assert song.tempo_bpm == 96
    assert song.duration_seconds == 10
    assert song.step_timings == {"generating_music": song.step_timings["generating_music"]}
    assert final_path(settings, song_id).read_bytes() == b"ID3-ace"
    assert not instrumental_path(settings, song_id).exists()
