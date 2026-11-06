from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from songcraft.config import Settings
from songcraft.models import Song  # noqa: F401  (register the table on SQLModel.metadata)


def _sqlite_file(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:///"):
        return None
    raw = database_url.removeprefix("sqlite:///")
    if raw.startswith(":memory:"):
        return None
    return Path(raw)


class Database:
    def __init__(self, settings: Settings) -> None:
        connect_args: dict[str, object] = {}
        if settings.database_url.startswith("sqlite"):
            # The request thread and the worker thread each open a session.
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = 30
        self.engine = create_engine(settings.database_url, connect_args=connect_args)
        if settings.database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", _enable_wal)

    @contextmanager
    def session(self) -> Iterator[Session]:
        # expire_on_commit stays off so the runner can keep using the song row
        # across the commit that happens after every step.
        with Session(self.engine, expire_on_commit=False) as session:
            yield session

    def create_all(self, settings: Settings) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "songs").mkdir(parents=True, exist_ok=True)
        sqlite_file = _sqlite_file(settings.database_url)
        if sqlite_file is not None and sqlite_file.parent != Path(""):
            sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(self.engine)


def _enable_wal(connection: object, _record: object) -> None:
    cursor = connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
