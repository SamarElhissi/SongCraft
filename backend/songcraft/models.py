from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from songcraft.clock import utcnow


class SongStatus(str, Enum):
    queued = "queued"
    generating_music = "generating_music"
    analyzing_beats = "analyzing_beats"
    writing_lyrics = "writing_lyrics"
    synthesizing_vocals = "synthesizing_vocals"
    mixing = "mixing"
    completed = "completed"
    failed = "failed"


RUNNING_STATUSES = {
    SongStatus.generating_music,
    SongStatus.analyzing_beats,
    SongStatus.writing_lyrics,
    SongStatus.synthesizing_vocals,
    SongStatus.mixing,
}

# Steps that record a duration. queued, completed, and failed are not work steps.
STEP_ORDER = (
    SongStatus.generating_music,
    SongStatus.analyzing_beats,
    SongStatus.writing_lyrics,
    SongStatus.synthesizing_vocals,
    SongStatus.mixing,
)


class Song(SQLModel, table=True):
    __tablename__ = "song"

    id: str = Field(primary_key=True)
    prompt: str
    duration_seconds: float
    status: str = Field(default=SongStatus.queued.value, index=True)
    error_message: str | None = None
    lyrics: str | None = None
    tempo_bpm: float | None = None
    step_timings: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)
