"""Local bounded trace store. Only metadata is recorded, never prompts or tool arguments."""
import json
import sqlite3
import time
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager
from app.config.settings import settings

trace_user = ContextVar("trace_user", default=None)

@contextmanager
def connect():
    path = Path(settings.trace_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute("CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS trace_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, request TEXT NOT NULL, time REAL NOT NULL, payload TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS trace_events_request ON trace_events(request, seq)")
    try:
        with db:
            yield db
    finally:
        db.close()

def append(request, payload):
    user = trace_user.get()
    if not settings.trace_enabled or user is None or request == "-":
        return
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO traces VALUES (?,?,?)", (request, user, time.time()))
        owner = db.execute("SELECT owner FROM traces WHERE id=?", (request,)).fetchone()
        if owner[0] != user:
            return
        count = db.execute("SELECT count(*) FROM trace_events WHERE request=?", (request,)).fetchone()[0]
        if count < 1000:
            db.execute("INSERT INTO trace_events(request,time,payload) VALUES (?,?,?)", (request, time.time(), json.dumps(payload)))

def read(request, user):
    with connect() as db:
        row = db.execute("SELECT created FROM traces WHERE id=? AND owner=?", (request, user)).fetchone()
        if row is None:
            return None
        events = [{"time": ts, **json.loads(payload)} for ts,payload in db.execute(
            "SELECT time,payload FROM trace_events WHERE request=? ORDER BY seq", (request,))]
    usage = [e for e in events if e.get("event") == "model_usage"]
    return {"request_id": request, "created": row[0], "events": events,
            "metrics": {"input_tokens": sum(e.get("input_tokens",0) for e in usage),
                        "output_tokens": sum(e.get("output_tokens",0) for e in usage),
                        "estimated_cost": sum(e.get("estimated_cost",0) for e in usage),
                        "pricing_configured": bool(usage) and all(e.get('pricing_configured',False) for e in usage),
                        "embedding_tokens": sum(e.get('tokens',0) for e in events if e.get('event')=='embedding_usage'),
                        "first_response_ms": next((e['elapsed_ms'] for e in events if e.get('event')=='first_response'),None),
                        "duration_ms": next((e['elapsed_ms'] for e in events if e.get('event')=='request_complete'),None),
                        "tool_calls": sum(e.get('event')=='tool_outcome' for e in events),
                        "invalid_tool_arguments": sum(e.get('event')=='tool_validation' and not e.get('valid') for e in events)}}

def prune(days=7):
    with connect() as db:
        db.execute("DELETE FROM trace_events WHERE request IN (SELECT id FROM traces WHERE created < ?)", (time.time()-days*86400,))
        db.execute("DELETE FROM traces WHERE created < ?", (time.time()-days*86400,))

@contextmanager
def user_trace(user):
    token = trace_user.set(user)
    try:
        yield
    finally:
        trace_user.reset(token)
