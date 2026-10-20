"""Versioned extractive summaries backed by immutable, scoped source records."""
import json
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from app.config.settings import settings
from app.agent.context import dumps,compact_evidence,evidence

def scope_key(user,session):
    return hashlib.sha256(dumps([user,session]).encode()).hexdigest()

@contextmanager
def archive():
    path=Path(settings.summary_archive_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=15)
    try:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS sources(scope TEXT,id TEXT,payload TEXT,seq INTEGER PRIMARY KEY AUTOINCREMENT,UNIQUE(scope,id))')
            db.execute('CREATE TABLE IF NOT EXISTS summaries(scope TEXT,revision INTEGER,payload TEXT,PRIMARY KEY(scope,revision))')
            yield db
    finally: db.close()

def rebuild(user,session,excluded=(),additions=()):
    scope=scope_key(user,session)
    with archive() as db:
        db.execute('BEGIN IMMEDIATE')
        for msg in additions:
            db.execute('INSERT OR IGNORE INTO sources(scope,id,payload) VALUES (?,?,?)',(scope,evidence(msg)['source_id'],dumps(msg)))
        previous=db.execute('SELECT revision,payload FROM summaries WHERE scope=? ORDER BY revision DESC LIMIT 1',(scope,)).fetchone()
        prior=json.loads(previous[1]) if previous else {}
        excluded=set(excluded)|set(prior.get('excluded_sources',[]))
        rows=db.execute('SELECT id,payload FROM sources WHERE scope=? ORDER BY seq',(scope,)).fetchall()
        if not excluded.issubset({r[0] for r in rows}): raise ValueError('未知来源不能被更正')
        messages=[json.loads(payload) for sid,payload in rows if sid not in excluded]
        record={'schema_version':2,'scope':scope,'revision':previous[0]+1 if previous else 1,
                'excluded_sources':sorted(excluded),'source_ids':[sid for sid,_ in rows if sid not in excluded],
                'method':'extractive_verified','summary':compact_evidence(messages,settings.summary_max_bytes)}
        raw=dumps(record)
        db.execute('INSERT INTO summaries VALUES (?,?,?)',(scope,record['revision'],raw))
        return raw

def verified(summary,user,session):
    if not summary: return None
    try: data=json.loads(summary)
    except (TypeError,ValueError): return None
    if not isinstance(data,dict) or data.get('scope')!=scope_key(user,session): return None
    with archive() as db:
        row=db.execute('SELECT payload FROM summaries WHERE scope=? ORDER BY revision DESC LIMIT 1',(data['scope'],)).fetchone()
        if not row: return None
        canonical=json.loads(row[0])
        known={sid:json.loads(payload) for sid,payload in db.execute('SELECT id,payload FROM sources WHERE scope=?',(data['scope'],))}
        if any(sid not in known or evidence(known[sid])['source_id']!=sid for sid in canonical['source_ids']): return None
        # Verify evidence itself, not merely the cached caller string.
        recomputed=compact_evidence([known[sid] for sid in canonical['source_ids']],settings.summary_max_bytes)
        canonical['summary']=recomputed
    return canonical

def prompt_summary(summary,user,session):
    data=verified(summary,user,session)
    return dumps(data['summary']) if data else ''

async def summarize(client,model,old_messages,prev_summary,user_id='default',session_key='default'):
    import asyncio
    return await asyncio.to_thread(rebuild,user_id,session_key,(),old_messages)

def delete_archive(user,session):
    scope=scope_key(user,session)
    with archive() as db:
        db.execute('DELETE FROM sources WHERE scope=?',(scope,))
        db.execute('DELETE FROM summaries WHERE scope=?',(scope,))

async def compress_agent(agent):
    from app.agent.context import interaction_units,estimate_tokens,input_budget
    units=interaction_units(agent.raw_messages)
    if len(units)<=1: return
    keep=min(settings.history_keep_turns,len(units)-1)
    while keep>1 and estimate_tokens([m for u in units[-keep:] for m in u])>input_budget()//2:
        keep-=1
    old=[m for u in units[:-keep] for m in u]
    agent.summary=await summarize(agent.client,agent.model,old,agent.summary,agent.user_id,agent.session_path)
    agent.raw_messages=[m for u in units[-keep:] for m in u]


def read_source(user, session, source_id):
    """Read original evidence only within the authenticated user's session scope."""
    with archive() as db:
        row = db.execute('SELECT payload FROM sources WHERE scope=? AND id=?',
                         (scope_key(user, session), source_id)).fetchone()
    if not row:
        return None
    message = json.loads(row[0])
    return message if evidence(message)['source_id'] == source_id else None


def search_sources(user, session, query, top_k=5):
    from app.agent.rag.lexical import tokenize
    if not query.strip():
        return []
    terms = set(tokenize(query)) - {'我','的','是','了','你','请','一下','什么','有'}
    for word, alias in {'订单':'order_id','金额':'amount','退款':'refund','状态':'status'}.items():
        if word in terms: terms.add(alias)
    with archive() as db:
        scope = scope_key(user, session)
        latest = db.execute('SELECT payload FROM summaries WHERE scope=? ORDER BY revision DESC LIMIT 1', (scope,)).fetchone()
        excluded = set(json.loads(latest[0]).get('excluded_sources', [])) if latest else set()
        rows = db.execute('SELECT id,payload,seq FROM sources WHERE scope=? ORDER BY seq DESC', (scope,)).fetchall()
    matches = []
    for sid, payload, seq in rows:
        if sid in excluded: continue
        message = json.loads(payload)
        row = evidence(message)
        if row['source_id'] != sid: continue
        score = len(terms & set(tokenize(dumps(row))))
        if score: matches.append((score, seq, row))
    return [row for _, _, row in sorted(matches, key=lambda r:(r[0],r[1]), reverse=True)[:top_k]]
