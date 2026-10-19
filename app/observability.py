"""Structured diagnostics without prompts, tokens or exception messages."""
import json
import logging
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

request_id = ContextVar("request_id", default="-")
span_id = ContextVar("span_id", default=None)
logger = logging.getLogger("yier.service")

def new_request_id():
    return uuid4().hex

def event(name, **fields):
    payload = {"event": name, "request_id": request_id.get(), "span_id":span_id.get(), **fields}
    logger.info(json.dumps(payload))
    try:
        from app.trace_store import append
        append(request_id.get(), payload)
    except Exception:
        logger.warning("trace_store_unavailable")

def failure(name, exc):
    # Traceback frames only: raw exception messages may contain API keys or prompts.
    frames = [{"file": f.filename, "line": f.lineno, "function": f.name}
              for f in traceback.extract_tb(exc.__traceback__)]
    logger.error(json.dumps({"event": name, "request_id": request_id.get(),
                             "exception_type": type(exc).__name__, "frames": frames}))
    event("failure", code=name, exception_type=type(exc).__name__)

@contextmanager
def stage(name):
    started = time.perf_counter()
    parent = span_id.get()
    current = uuid4().hex
    token = span_id.set(current)
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        event("stage", stage=name, span_id=current, parent_span_id=parent,
              elapsed_ms=round((time.perf_counter()-started)*1000, 2), outcome=outcome)
        span_id.reset(token)

def model_usage(response):
    from app.config.settings import settings
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    event("model_usage", input_tokens=inp, output_tokens=out,
          pricing_configured=bool(settings.llm_input_cost_per_million or settings.llm_output_cost_per_million),
          estimated_cost=(inp*settings.llm_input_cost_per_million+out*settings.llm_output_cost_per_million)/1_000_000)
