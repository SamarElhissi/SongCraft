import logging
import queue
import threading
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger("songcraft.queue")


class JobQueue(Protocol):
    def start(self) -> None:
        """Start consuming jobs. Inline queues do nothing here."""

    def shutdown(self) -> None:
        """Stop the consumer. Safe to call when the queue was never started."""

    def enqueue(self, song_id: str) -> None:
        """Schedule one song. The id is already stored in the database."""


class InlineJobQueue:
    """Runs a job inside enqueue so tests can observe the result without sleeping."""

    def __init__(self, handler: Callable[[str], None]) -> None:
        self._handler = handler

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def enqueue(self, song_id: str) -> None:
        self._handler(song_id)


class InProcessJobQueue:
    """Single worker thread. Inference is heavy, so songs run one at a time."""

    def __init__(self, handler: Callable[[str], None]) -> None:
        self._handler = handler
        self._queue: queue.Queue[str | None] = queue.Queue()
        # Daemon so a process exit is not stuck on a long generation.
        # The next startup marks any song left in a running status as failed.
        self._thread = threading.Thread(
            target=self._loop, name="songcraft-worker", daemon=True
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join(timeout=5)

    def enqueue(self, song_id: str) -> None:
        self._queue.put(song_id)

    def _loop(self) -> None:
        while True:
            song_id = self._queue.get()
            if song_id is None:
                return
            try:
                self._handler(song_id)
            except Exception:
                logger.exception("worker_crashed job_id=%s", song_id)


def build_job_queue(name: str, handler: Callable[[str], None]) -> JobQueue:
    if name == "inline":
        return InlineJobQueue(handler)
    if name == "inprocess":
        return InProcessJobQueue(handler)
    raise ValueError(f"Unknown job queue '{name}'. Expected inline or inprocess.")
