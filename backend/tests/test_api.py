import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from songcraft.db import Database
from songcraft.main import create_app
from songcraft.models import Song, SongStatus


def test_create_returns_queued_then_get_shows_completed(client: TestClient) -> None:
    created = client.post(
        "/api/songs",
        json={"prompt": "  rain on the window  ", "duration_seconds": 12},
    )

    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["prompt"] == "rain on the window"
    assert body["duration_seconds"] == 12
    assert body["lyrics"] is None
    assert body["error_message"] is None
    assert body["step_timings"] == {}
    assert body["tracks"] == {"instrumental": False, "final": False}

    fetched = client.get(f"/api/songs/{body['id']}")
    assert fetched.status_code == 200
    finished = fetched.json()
    assert finished["status"] == "completed"
    assert finished["lyrics"]
    assert "rain on the window" in finished["lyrics"]
    assert finished["tempo_bpm"] == 120
    assert set(finished["step_timings"]) == {
        "generating_music",
        "analyzing_beats",
        "writing_lyrics",
        "synthesizing_vocals",
        "mixing",
    }
    assert finished["tracks"] == {"instrumental": True, "final": True}

    instrumental = client.get(f"/api/songs/{body['id']}/audio?track=instrumental")
    final = client.get(f"/api/songs/{body['id']}/audio?track=final")
    assert instrumental.status_code == 200
    assert instrumental.headers["content-type"].startswith("audio/wav")
    assert instrumental.content
    assert final.status_code == 200
    assert final.headers["content-type"].startswith("audio/mpeg")
    assert final.content


def test_list_is_newest_first(client: TestClient) -> None:
    assert client.get("/api/songs").json() == []
    first = client.post(
        "/api/songs", json={"prompt": "first song idea", "duration_seconds": 8}
    ).json()
    second = client.post(
        "/api/songs", json={"prompt": "second song idea", "duration_seconds": 8}
    ).json()

    ids = [song["id"] for song in client.get("/api/songs").json()]
    assert ids == [second["id"], first["id"]]


def test_duration_is_clamped(client: TestClient) -> None:
    short = client.post(
        "/api/songs", json={"prompt": "too short please", "duration_seconds": 1}
    ).json()
    long = client.post(
        "/api/songs", json={"prompt": "too long please", "duration_seconds": 90}
    ).json()

    assert short["duration_seconds"] == 5
    assert long["duration_seconds"] == 30


def test_prompt_must_be_3_to_500_characters(client: TestClient) -> None:
    short = client.post("/api/songs", json={"prompt": "  no  ", "duration_seconds": 8})
    long = client.post(
        "/api/songs", json={"prompt": "a" * 501, "duration_seconds": 8}
    )
    missing = client.post("/api/songs", json={"duration_seconds": 8})

    assert short.status_code == 422
    assert long.status_code == 422
    assert missing.status_code == 422
    assert client.post(
        "/api/songs", json={"prompt": "a" * 500, "duration_seconds": 8}
    ).status_code == 202


def test_missing_song_and_bad_track(client: TestClient) -> None:
    missing = client.get(f"/api/songs/{uuid.uuid4()}")
    bad_id = client.get("/api/songs/not-a-uuid")
    bad_track = client.get(f"/api/songs/{uuid.uuid4()}/audio?track=vocals")

    assert missing.status_code == 404
    assert missing.json()["detail"] == "Song not found"
    assert bad_id.status_code == 404
    assert bad_track.status_code == 400
    assert bad_track.json()["detail"] == "track must be final or instrumental"


def test_delete_removes_the_row_and_files(client: TestClient, settings) -> None:
    created = client.post(
        "/api/songs", json={"prompt": "delete this song", "duration_seconds": 8}
    ).json()
    directory = settings.data_dir.resolve() / "songs" / created["id"]
    assert directory.is_dir()

    deleted = client.delete(f"/api/songs/{created['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/api/songs/{created['id']}").status_code == 404
    assert client.delete(f"/api/songs/{created['id']}").status_code == 404
    assert not directory.exists()


def test_failed_job_hides_the_exception_text(client: TestClient) -> None:
    class ExplodingMusic:
        def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
            raise RuntimeError("cuda exploded: secret stack")

    client.app.state.pipeline.music = ExplodingMusic()
    created = client.post(
        "/api/songs", json={"prompt": "this will fail", "duration_seconds": 8}
    )
    assert created.status_code == 202
    assert created.json()["status"] == "queued"

    fetched = client.get(f"/api/songs/{created.json()['id']}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["status"] == "failed"
    assert body["error_message"] == "Music generation failed"
    assert "secret stack" not in fetched.text
    assert "Traceback" not in fetched.text


def test_audio_not_ready_is_404(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    song_id = str(uuid.uuid4())
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="no files yet",
                duration_seconds=5,
                status=SongStatus.completed.value,
            )
        )
        session.commit()
    database.engine.dispose()

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/songs/{song_id}/audio?track=final")

    assert response.status_code == 404
    assert response.json()["detail"] == "Audio not ready"


def test_startup_fails_interrupted_song_and_finishes_queued(settings_factory) -> None:
    settings = settings_factory()
    database = Database(settings)
    database.create_all(settings)
    running_id = str(uuid.uuid4())
    queued_id = str(uuid.uuid4())
    with database.session() as session:
        session.add(
            Song(
                id=running_id,
                prompt="caught mixing",
                duration_seconds=5,
                status=SongStatus.mixing.value,
            )
        )
        session.add(
            Song(
                id=queued_id,
                prompt="still queued",
                duration_seconds=5,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
    database.engine.dispose()

    with TestClient(create_app(settings)) as client:
        running = client.get(f"/api/songs/{running_id}").json()
        queued = client.get(f"/api/songs/{queued_id}").json()

    assert running["status"] == "failed"
    assert running["error_message"] == "Interrupted by server restart"
    assert queued["status"] == "completed"


def test_inprocess_queue_finishes_in_the_background(settings_factory) -> None:
    settings = settings_factory(job_queue="inprocess")
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/songs", json={"prompt": "background song", "duration_seconds": 8}
        )
        assert created.status_code == 202
        assert created.json()["status"] == "queued"
        song_id = created.json()["id"]
        deadline = time.time() + 5
        body = created.json()
        while time.time() < deadline:
            body = client.get(f"/api/songs/{song_id}").json()
            if body["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

    assert body["status"] == "completed"
