"""Opt-in real PostgreSQL tests. Use scripts/verify_memory_postgres.py for isolation."""
import asyncio
import json
import multiprocessing
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch
from sqlalchemy import create_engine, text
from app.config.settings import settings
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.facts import extract
from app.agent.memory.store import pg_transaction, transaction
from app.server import repository

URL = os.environ.get('MEMORY_TEST_DATABASE_URL')


def make_fact(value, observed=None):
    return extract([{'role':'user','content':value,'_observed_at':observed or time.time()}], 'pg-test')


def process_write(index):
    settings.db_enabled=True
    settings.database_url=os.environ['MEMORY_TEST_DATABASE_URL']
    m=LongTermMemory('processes',source_session=str(index))
    m.load(); m.add_facts(make_fact('我喜欢颜色'+str(index))); m.save()
    repository.dispose_engine()


@unittest.skipUnless(URL, 'Run isolated PostgreSQL runner; never use application database by default')
class PostgreSQLMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overrides=[patch.object(settings,'db_enabled',True),patch.object(settings,'memory_semantic_enabled',False),patch.object(settings,'database_url',URL)]
        for p in cls.overrides: p.start()
        repository.dispose_engine()
        cls.engine=create_engine(URL.replace('postgresql+asyncpg:', 'postgresql+psycopg:'))
        from app.server.models import Base, MemoryFact, MemoryState
        Base.metadata.create_all(cls.engine,tables=[MemoryFact.__table__,MemoryState.__table__])
        with cls.engine.begin() as db:
            db.execute(text("""CREATE OR REPLACE FUNCTION reject_test_memory() RETURNS trigger LANGUAGE plpgsql AS $$
              BEGIN IF NEW.payload->>'reject' = 'true' THEN RAISE EXCEPTION 'injected failure'; END IF; RETURN NEW; END $$"""))
            db.execute(text('CREATE TRIGGER reject_memory BEFORE UPDATE ON memory_states_v2 FOR EACH ROW EXECUTE FUNCTION reject_test_memory()'))
    @classmethod
    def tearDownClass(cls):
        repository.dispose_engine(); cls.engine.dispose()
        for p in reversed(cls.overrides): p.stop()
    def setUp(self):
        with self.engine.begin() as db:
            db.execute(text('TRUNCATE memory_states_v2, memory_facts'))
    def memory(self,user='alice'):
        m=LongTermMemory(user,source_session='s'); m.load(); return m
    def test_same_user_parallel_writers_preserve_union(self):
        writers=[self.memory() for _ in range(12)]
        start=threading.Barrier(len(writers))
        def write(i):
            writers[i].add_facts(make_fact('我喜欢颜色'+str(i))); start.wait(timeout=10); writers[i].save()
        with ThreadPoolExecutor(max_workers=12) as pool: list(pool.map(write,range(12)))
        facts=self.memory().facts
        self.assertEqual(len(facts),12)
        self.assertEqual({f.attribute for f in facts},{'preference:颜色'+str(i) for i in range(12)})
        self.assertEqual(self.memory('bob').facts,[])
    def test_same_user_processes_preserve_union(self):
        ctx=multiprocessing.get_context('spawn')
        workers=[ctx.Process(target=process_write,args=(i,)) for i in range(4)]
        for process in workers: process.start()
        for process in workers:
            process.join(25)
            if process.is_alive(): process.terminate(); process.join(); self.fail('writer timeout')
            self.assertEqual(process.exitcode,0)
        self.assertEqual(len(self.memory('processes').facts),4)
    def wait_for_db_lock(self):
        until=time.monotonic()+10
        while time.monotonic()<until:
            with self.engine.connect() as db:
                waiting=db.execute(text("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock'")).scalar_one()
            if waiting: return
            time.sleep(.02)
        self.fail('Expected real PostgreSQL lock contention')
    def compete(self, first_reset):
        old=self.memory(); old.add_facts(make_fact('我喜欢黑色',time.time()-10))
        resetter=self.memory()
        entered, release=threading.Event(),threading.Event()
        errors=[]
        def gated(user,path,update=None):
            if update is None: return transaction(user,path)
            def callback(data):
                if threading.current_thread().name=='hold-db-row':
                    entered.set()
                    if not release.wait(15): raise TimeoutError('test gate')
                return update(data)
            return transaction(user,path,callback)
        def run(fn):
            try: fn()
            except BaseException as exc: errors.append(exc)
        first=resetter.reset if first_reset else old.save
        second=old.save if first_reset else resetter.reset
        with patch('app.agent.memory.long_term.transaction',gated):
            a=threading.Thread(target=run,args=(first,),name='hold-db-row')
            b=threading.Thread(target=run,args=(second,),name='wait-db-row')
            a.start()
            try:
                self.assertTrue(entered.wait(10)); b.start(); self.wait_for_db_lock()
            finally:
                release.set(); a.join(20)
                if b.ident: b.join(20)
        self.assertFalse(errors, [type(e).__name__ for e in errors])
        self.assertFalse(a.is_alive()); self.assertFalse(b.is_alive())
        self.assertEqual(self.memory().facts,[])
        old.refresh(); old.add_facts(make_fact('我喜欢黑色',time.time()-60)); old.save()
        self.assertEqual(self.memory().facts,[])
    def test_reset_commits_before_waiting_stale_write(self): self.compete(True)
    def test_old_write_commits_before_waiting_reset(self): self.compete(False)
    def test_late_preference_cannot_replace_newer_observation(self):
        old,new=self.memory(),self.memory()
        old.add_facts(make_fact('我不喜欢黑色',time.time()-10))
        new.add_facts(make_fact('我喜欢黑色'))
        new.save(); old.save()
        self.assertEqual([f.value for f in self.memory().recall('黑色')],['like'])
    def test_update_sql_failure_rolls_back_and_releases_lock(self):
        m=self.memory(); m.add_facts(make_fact('我喜欢黑色')); m.save()
        before=asyncio.run(pg_transaction('alice'))
        from sqlalchemy.exc import DBAPIError
        with self.assertRaises(DBAPIError): asyncio.run(pg_transaction('alice',lambda data:{**data,'reject':True}))
        self.assertEqual(asyncio.run(pg_transaction('alice')),before)
        m.add_facts(make_fact('我喜欢蓝色')); m.save()
        self.assertEqual(len(self.memory().facts),2)
    def test_new_user_failure_rolls_back_insert(self):
        def fail(data): raise RuntimeError('intentional rollback')
        with self.assertRaises(RuntimeError): asyncio.run(pg_transaction('new-user',fail))
        with self.engine.connect() as db:
            count=db.execute(text("SELECT count(*) FROM memory_states_v2 WHERE user_id='new-user'")).scalar_one()
        self.assertEqual(count,0)
    def test_legacy_rows_migrate_unverified_and_reset_does_not_restore_them(self):
        with self.engine.begin() as db:
            db.execute(text("INSERT INTO memory_facts(user_id,kind,category,content) VALUES ('alice','fact','preference','旧偏好')"))
        m=self.memory(); self.assertEqual(len(m.facts),1); self.assertEqual(m.recall(),[])
        m.reset(); self.assertEqual(self.memory().facts,[])


if __name__=='__main__': unittest.main()
