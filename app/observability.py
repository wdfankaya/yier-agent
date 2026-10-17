"""Structured diagnostics without prompts, tokens or exception messages."""
import json
import logging
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

request_id = ContextVar("request_id", default="-")
logger = logging.getLogger("yier.service")

def new_request_id():
    return uuid4().hex

def event(name, **fields):
    logger.info(json.dumps({"event": name, "request_id": request_id.get(), **fields}))

def failure(name, exc):
    # Traceback frames only: raw exception messages may contain API keys or prompts.
    frames = [{"file": f.filename, "line": f.lineno, "function": f.name}
              for f in traceback.extract_tb(exc.__traceback__)]
    logger.error(json.dumps({"event": name, "request_id": request_id.get(),
                             "exception_type": type(exc).__name__, "frames": frames}))

@contextmanager
def stage(name):
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        event("stage", stage=name, elapsed_ms=round((time.perf_counter()-started)*1000, 2), outcome=outcome)
