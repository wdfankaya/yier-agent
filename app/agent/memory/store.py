"""Atomic user-state transactions, independent of ContextVar bindings."""
import copy
import json
import os
import time
import uuid
from contextlib import contextmanager
from app.locks import KeyedLocks

LOCKS=KeyedLocks()

@contextmanager
def file_lock(path):
    """Interprocess lock on a stable sidecar (the JSON itself is atomically replaced)."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with LOCKS.hold(str(path.resolve())):
        with path.with_suffix('.lock').open('a+b') as handle:
            handle.seek(0,2)
            if handle.tell()==0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            if os.name=='nt':
                import msvcrt
                deadline=time.monotonic()+15
                while True:
                    try:
                        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                        break
                    except OSError:
                        if time.monotonic()>=deadline: raise TimeoutError('Memory state lock timeout') from None
                        time.sleep(.02)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            try: yield
            finally:
                handle.seek(0)
                if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

def json_transaction(path,update=None):
    with file_lock(path):
        data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if not isinstance(data,dict): raise ValueError('Memory state must be an object')
        if update is None: return data
        result=update(copy.deepcopy(data))
        temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            with temp.open('w',encoding='utf-8') as out:
                json.dump(result,out,ensure_ascii=False,indent=2)
                out.flush(); os.fsync(out.fileno())
            os.replace(temp,path)
        finally: temp.unlink(missing_ok=True)
        return result

async def pg_transaction(user,update=None):
    """Row lock serializes read/merge/write. JSON metadata survives PG round trips."""
    from sqlalchemy import text
    from app.server.repository import _get_session_factory
    async with _get_session_factory()() as db:
        async with db.begin():
            await db.execute(text("INSERT INTO memory_states_v2 VALUES (:user, '{}'::jsonb) ON CONFLICT DO NOTHING"),{'user':user})
            data=(await db.execute(text('SELECT payload FROM memory_states_v2 WHERE user_id=:user FOR UPDATE'),{'user':user})).scalar_one()
            if not data:
                # Old facts have no evidence; migrate without inventing provenance.
                rows=(await db.execute(text('SELECT category,content,created_at,kind FROM memory_facts WHERE user_id=:user ORDER BY id'),{'user':user})).mappings().all()
                data={'facts':[{'content':r['content'],'category':r['category'],'created_at':r['created_at'].isoformat()} for r in rows if r['kind']=='fact'],
                      'interaction_summaries':[{'summary':r['content'],'timestamp':r['created_at'].isoformat()} for r in rows if r['kind']=='session_summary']}
            result=update(copy.deepcopy(data)) if update else data
            if update:
                await db.execute(text('UPDATE memory_states_v2 SET payload=CAST(:payload AS jsonb) WHERE user_id=:user'),
                                 {'user':user,'payload':json.dumps(result,ensure_ascii=False)})
            return result

def transaction(user,path,update=None):
    from app.config.settings import settings
    if settings.db_enabled:
        from app.agent.async_utils import run_sync
        return run_sync(pg_transaction(user,update))
    return json_transaction(path,update)
