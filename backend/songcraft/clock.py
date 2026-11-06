import threading
from datetime import datetime, timedelta, timezone

_lock = threading.Lock()
_last: datetime | None = None


def utcnow() -> datetime:
    """UTC timestamp that always moves forward.

    Windows often reports the same clock value for calls a few milliseconds
    apart, which would make newest-first ordering unstable.
    """
    global _last
    now = datetime.now(timezone.utc)
    with _lock:
        if _last is not None and now <= _last:
            now = _last + timedelta(microseconds=1)
        _last = now
        return now
