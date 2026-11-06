from sqlmodel import Session, select

from songcraft.clock import utcnow
from songcraft.models import RUNNING_STATUSES, Song, SongStatus


def recover_interrupted_jobs(session: Session) -> list[str]:
    """Fail songs caught mid-step and return ids that were still queued.

    A running step cannot be resumed, so it is marked failed. A song that never
    left the queue is returned so the caller can enqueue it again.
    """
    requeue: list[str] = []
    songs = session.exec(select(Song)).all()
    for song in songs:
        status = _status_value(song.status)
        if status in {item.value for item in RUNNING_STATUSES}:
            song.status = SongStatus.failed.value
            song.error_message = "Interrupted by server restart"
            song.updated_at = utcnow()
            session.add(song)
        elif status == SongStatus.queued:
            requeue.append(song.id)
    session.commit()
    return requeue


def _status_value(status: SongStatus | str) -> str:
    if isinstance(status, SongStatus):
        return status.value
    return status
