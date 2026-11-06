import logging
import sqlite3
import uuid
from pathlib import Path

import pytest

from songcraft.db import Database
from songcraft.models import STEP_ORDER, Song, SongStatus
from songcraft.paths import final_path, instrumental_path, song_dir
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.fakes import FakeMusicGenerator
from songcraft.pipeline.runner import run_song
from songcraft.recovery import recover_interrupted_jobs


def test_fake_pipeline_completes_and_persists(
    tmp_path: Path, settings_factory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    song_id = str(uuid.uuid4())
    observed: list[str] = []

    class ProbeMusic(FakeMusicGenerator):
        def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
            connection = sqlite3.connect(tmp_path / "test.db")
            try:
                row = connection.execute(
                    "SELECT status FROM song WHERE id = ?",
                    (song_id,),
                ).fetchone()
            finally:
                connection.close()
            assert row is not None
            observed.append(row[0])
            super().generate(prompt, duration_seconds, output_path)

    pipeline = build_pipeline(settings)
    pipeline.music = ProbeMusic()
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="rain on the window",
                duration_seconds=8,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)

    database.engine.dispose()
    reloaded = Database(settings)
    with reloaded.session() as session:
        song = session.get(Song, song_id)

    assert observed == ["generating_music"]
    assert song is not None
    assert song.status == SongStatus.completed.value
    assert song.tempo_bpm == 120
    assert song.lyrics is not None
    assert "rain on the window" in song.lyrics
    assert set(song.step_timings) == {step.value for step in STEP_ORDER}
    assert all(timing >= 0 for timing in song.step_timings.values())
    assert instrumental_path(settings, song_id).is_file()
    assert final_path(settings, song_id).is_file()
    vocals = list((song_dir(settings, song_id) / "vocals").glob("*.mp3"))
    assert len(vocals) >= 2
    finished = [record for record in caplog.records if record.message.startswith("step_finished")]
    assert len(finished) == 5
    assert all(song_id in record.message and "duration_s=" in record.message for record in finished)


def test_empty_lyrics_still_finish(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    song_id = str(uuid.uuid4())
    pipeline = build_pipeline(settings)

    class EmptyLyrics:
        def generate(self, prompt: str, line_count: int = 2, syllables: int = 6) -> str:
            return ""

    pipeline.lyrics = EmptyLyrics()
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="quiet song",
                duration_seconds=6,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)
        song = session.get(Song, song_id)

    assert song is not None
    assert song.status == SongStatus.completed.value
    assert song.lyrics == ""
    assert set(song.step_timings) == {step.value for step in STEP_ORDER}
    vocal_dir = song_dir(settings, song_id) / "vocals"
    assert not vocal_dir.exists() or list(vocal_dir.glob("*")) == []
    assert final_path(settings, song_id).is_file()


def test_failed_step_stores_a_clean_message(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    song_id = str(uuid.uuid4())
    pipeline = build_pipeline(settings)

    class ExplodingMusic:
        def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
            raise RuntimeError("cuda exploded: secret stack")

    pipeline.music = ExplodingMusic()
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="broken song",
                duration_seconds=6,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)
        song = session.get(Song, song_id)

    assert song is not None
    assert song.status == SongStatus.failed.value
    assert song.error_message == "Music generation failed"
    assert "secret stack" not in song.error_message
    assert "Traceback" not in song.error_message
    assert "generating_music" not in (song.step_timings or {})


def test_vocal_failure_keeps_earlier_timings(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    song_id = str(uuid.uuid4())
    pipeline = build_pipeline(settings)

    class ExplodingVocals:
        def synthesize(self, line: str, notes: list, output_path: Path) -> None:
            raise RuntimeError("tts down")

    pipeline.vocals = ExplodingVocals()
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="broken voice",
                duration_seconds=6,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)
        song = session.get(Song, song_id)

    assert song is not None
    assert song.status == SongStatus.failed.value
    assert song.error_message == "Vocal synthesis failed"
    assert "generating_music" in song.step_timings
    assert "analyzing_beats" in song.step_timings
    assert "writing_lyrics" in song.step_timings
    assert "synthesizing_vocals" not in song.step_timings
    assert "mixing" not in song.step_timings


def test_recovery_fails_running_jobs_and_returns_queued_ids(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    running_id = str(uuid.uuid4())
    queued_id = str(uuid.uuid4())
    done_id = str(uuid.uuid4())
    with database.session() as session:
        session.add(
            Song(
                id=running_id,
                prompt="was mixing",
                duration_seconds=5,
                status=SongStatus.mixing.value,
            )
        )
        session.add(
            Song(
                id=queued_id,
                prompt="still waiting",
                duration_seconds=5,
                status=SongStatus.queued.value,
            )
        )
        session.add(
            Song(
                id=done_id,
                prompt="already done",
                duration_seconds=5,
                status=SongStatus.completed.value,
                lyrics="kept",
            )
        )
        session.commit()
        requeued = recover_interrupted_jobs(session)
        running = session.get(Song, running_id)
        queued = session.get(Song, queued_id)
        done = session.get(Song, done_id)

    assert requeued == [queued_id]
    assert running is not None
    assert running.status == SongStatus.failed.value
    assert running.error_message == "Interrupted by server restart"
    assert queued is not None
    assert queued.status == SongStatus.queued.value
    assert done is not None
    assert done.status == SongStatus.completed.value
    assert done.lyrics == "kept"


def test_unknown_generator_is_rejected(settings_factory) -> None:
    settings = settings_factory(music_generator="no-such-model")
    with pytest.raises(ValueError, match="music generator"):
        build_pipeline(settings)
