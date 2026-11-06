import shutil
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from sqlmodel import select

from songcraft.clock import utcnow
from songcraft.config import clamp_duration
from songcraft.models import Song, SongStatus
from songcraft.paths import final_path, instrumental_path, song_dir
from songcraft.schemas import SongCreate, SongRead, song_to_read

router = APIRouter(prefix="/songs", tags=["songs"])


@router.post("", status_code=202, response_model=SongRead)
def create_song(body: SongCreate, request: Request) -> SongRead:
    settings = request.app.state.settings
    song = Song(
        id=str(uuid.uuid4()),
        prompt=body.prompt,
        duration_seconds=clamp_duration(body.duration_seconds, settings),
        status=SongStatus.queued.value,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    # Close this session before enqueue. An inline worker would otherwise
    # deadlock on SQLite while this request still holds the connection.
    with request.app.state.db.session() as session:
        session.add(song)
        session.commit()
        payload = song_to_read(song, settings)
        song_id = song.id
    request.app.state.queue.enqueue(song_id)
    return payload


@router.get("", response_model=list[SongRead])
def list_songs(request: Request) -> list[SongRead]:
    settings = request.app.state.settings
    with request.app.state.db.session() as session:
        songs = session.exec(
            select(Song).order_by(Song.created_at.desc(), Song.id.desc())
        ).all()
        return [song_to_read(song, settings) for song in songs]


@router.get("/{song_id}", response_model=SongRead)
def read_song(song_id: str, request: Request) -> SongRead:
    settings = request.app.state.settings
    with request.app.state.db.session() as session:
        song = session.get(Song, _canonical_id(song_id))
        if song is None:
            raise HTTPException(status_code=404, detail="Song not found")
        return song_to_read(song, settings)


@router.get("/{song_id}/audio")
def read_audio(
    song_id: str,
    request: Request,
    track: str = Query(...),
) -> FileResponse:
    if track not in {"final", "instrumental"}:
        raise HTTPException(status_code=400, detail="track must be final or instrumental")
    settings = request.app.state.settings
    with request.app.state.db.session() as session:
        song = session.get(Song, _canonical_id(song_id))
        if song is None:
            raise HTTPException(status_code=404, detail="Song not found")
        song_key = song.id
    if track == "final":
        path = final_path(settings, song_key)
        media_type = "audio/mpeg"
    else:
        path = instrumental_path(settings, song_key)
        media_type = "audio/wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not ready")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.delete("/{song_id}", status_code=204, response_class=Response)
def delete_song(song_id: str, request: Request) -> Response:
    settings = request.app.state.settings
    with request.app.state.db.session() as session:
        song = session.get(Song, _canonical_id(song_id))
        if song is None:
            raise HTTPException(status_code=404, detail="Song not found")
        directory = song_dir(settings, song.id)
        if directory.exists():
            shutil.rmtree(directory)
        session.delete(song)
        session.commit()
    return Response(status_code=204)


def _canonical_id(song_id: str) -> str:
    try:
        return str(uuid.UUID(song_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Song not found") from None
