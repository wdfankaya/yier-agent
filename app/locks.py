"""Reference-counted per-key locks for single-process state."""
import threading
from contextlib import contextmanager

class KeyedLocks:
    def __init__(self):
        self._guard = threading.Lock()
        self._locks = {}

    @contextmanager
    def hold(self, key):
        with self._guard:
            lock, users = self._locks.get(key, (threading.RLock(), 0))
            self._locks[key] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._guard:
                _, users = self._locks[key]
                if users == 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (lock, users - 1)
