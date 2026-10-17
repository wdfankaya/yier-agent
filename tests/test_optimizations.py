"""Offline tests for memory merge, bounded work, lifecycle and reply latency."""
import asyncio
import io
import json
import logging
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.config.settings import settings
from app.agent.memory.long_term import LongTermMemory, MemoryFact
from app.agent.tools.hitl import ConfirmationGate, aresolve_on_agent
from app.agent.tools.manager import ToolManager
from app.agent.maintenance import finish_turn, drain_maintenance
from app.server.sessions import SessionRegistry, SessionCapacityError
from app.observability import logger, failure, request_id, stage


def fact(text):
    return MemoryFact(text, 'preference', 'test')


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.db = patch.object(settings, 'db_enabled', False)
        self.db.start()
    def tearDown(self):
        self.db.stop(); self.temp.cleanup()
    def memory(self, user='alice'):
        m = LongTermMemory(user_id=user, memory_dir=self.temp.name)
        m.load()
        return m
    def test_stale_sessions_merge_and_user_isolation(self):
        a, b = self.memory(), self.memory()
        a.add_facts([fact('A')]); b.add_facts([fact('B')])
        a.save(); b.save()
        self.assertEqual({f.content for f in self.memory().facts}, {'A','B'})
        self.assertEqual(self.memory('bob').facts, [])
        b.save(); b.load(); b.load()
        self.assertEqual(len(b.facts), 2)
    def test_parallel_writers_merge_without_shared_temp_file(self):
        writers = [self.memory() for _ in range(8)]
        barrier = threading.Barrier(8)
        def save(i):
            writers[i].add_facts([fact(str(i))]); barrier.wait(); writers[i].save()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(save, range(8)))
        self.assertEqual({f.content for f in self.memory().facts}, set(map(str,range(8))))
        self.assertEqual(list(Path(self.temp.name).glob('*.tmp')), [])
    def test_reset_blocks_stale_resurrection(self):
        stale = self.memory(); stale.add_facts([fact('old')]); stale.save()
        self.memory().reset()
        stale.add_facts([fact('stale-update')]); stale.save()
        self.assertEqual(self.memory().facts, [])
    def test_corrupt_file_not_overwritten(self):
        m = self.memory(); m.memory_path.write_text('broken',encoding='utf-8')
        m.add_facts([fact('new')])
        with self.assertRaises(ValueError): m.save()
        self.assertEqual(m.memory_path.read_text(), 'broken')


class InitTests(unittest.TestCase):
    def test_different_sessions_build_concurrently_same_session_once(self):
        reg = SessionRegistry(); barrier = threading.Barrier(4)
        def build(*args): barrier.wait(timeout=2); return object()
        reg._build_agent = build
        with ThreadPoolExecutor(max_workers=4) as pool:
            agents = list(pool.map(lambda i: reg.get_or_create(str(i),'u'),range(4)))
        self.assertEqual(len({id(a) for a in agents}),4)
        reg = SessionRegistry(); reg._build_agent=Mock(side_effect=lambda *a:(time.sleep(.03),object())[1])
        with ThreadPoolExecutor(max_workers=8) as pool:
            agents=list(pool.map(lambda _:reg.get_or_create('s','u'),range(8)))
        self.assertEqual(len({id(a) for a in agents}),1)
        reg._build_agent.assert_called_once()
        self.assertFalse(reg._key_locks._locks)
    def test_failed_init_releases_capacity(self):
        reg=SessionRegistry(); reg._build_agent=Mock(side_effect=RuntimeError('failure'))
        with patch.object(settings,'session_max_resident',1):
            with self.assertRaises(RuntimeError): reg.get_or_create('s','u')
            self.assertFalse(reg._initializing)
            reg._build_agent=lambda *a:object()
            reg.get_or_create('s','u')
            with self.assertRaises(SessionCapacityError): reg.get_or_create('other','u')


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_timeout_is_unknown_and_not_replayed(self):
        gate=ConfirmationGate(); tm=ToolManager(hitl=gate)
        args={'order_id':'ORD-20240120-002','reason':'test'}
        complete=threading.Event(); calls=[]
        def slow(*args): calls.append(True); complete.wait(timeout=2); return '{"success":true}'
        agent=SimpleNamespace(hitl=gate,tool_manager=tm,raw_messages=[],save=Mock())
        with patch.object(settings,'hitl_enabled',True),patch.object(settings,'tool_timeout',.03),patch('app.agent.tools.manager.local_execute_tool',slow):
            token=json.loads(tm.execute_tool('apply_refund',args))['confirm_token']
            started=time.perf_counter()
            result=await aresolve_on_agent(agent,token,True)
            self.assertLess(time.perf_counter()-started,.5)
            self.assertEqual(result['execution_status'],'unknown');self.assertFalse(result['executed'])
            self.assertTrue(tm.has_pending_work)
            self.assertEqual(json.loads(tm.execute_tool('apply_refund',args))['status'],'execution_unknown')
            self.assertFalse((await aresolve_on_agent(agent,token,True))['ok'])
            complete.set(); await asyncio.gather(*list(tm._inflight))
            self.assertEqual(len(calls),1)
            self.assertTrue(json.loads(tm.execute_tool('apply_refund',args))['success'])
    async def test_business_failure_not_reported_as_success(self):
        gate=ConfirmationGate(); tm=ToolManager(hitl=gate)
        args={'order_id':'missing','reason':'test'}
        with patch.object(settings,'hitl_enabled',True):
            token=json.loads(tm.execute_tool('apply_refund',args))['confirm_token']
            result=await aresolve_on_agent(SimpleNamespace(hitl=gate,tool_manager=tm,raw_messages=[]),token,True)
        self.assertEqual(result['execution_status'],'failed'); self.assertFalse(result['executed'])
    def fake_agent(self):
        return SimpleNamespace(save=Mock(),aclose=AsyncMock())
    async def test_idle_eviction_preserves_busy_and_pending_confirmations(self):
        reg=SessionRegistry()
        agents={name:self.fake_agent() for name in ('idle','busy','confirmation','background')}
        for name,agent in agents.items():
            reg._agents[('u',name)]=agent;reg._last_used[('u',name)]=0
        reg.acquire('busy','u')
        gate=ConfirmationGate()
        with patch.object(settings,'hitl_enabled',True):gate.intercept('apply_refund',{'order_id':'a'})
        agents['confirmation'].hitl=gate
        task=asyncio.create_task(asyncio.sleep(5));agents['background']._maintenance_task=task
        with patch.object(settings,'session_idle_ttl',.01):await reg.reap_idle()
        self.assertIsNone(reg.get('idle','u'))
        agents['idle'].save.assert_called_once();agents['idle'].aclose.assert_awaited_once()
        self.assertEqual(len(reg),3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
    async def test_capacity_reclaims_lru_and_save_failure_keeps_data(self):
        reg=SessionRegistry(); old=self.fake_agent();new=self.fake_agent()
        reg._agents={('u','old'):old,('u','new'):new}
        reg._last_used={('u','old'):time.monotonic()-1,('u','new'):time.monotonic()}
        with patch.object(settings,'session_max_resident',2):await reg.reap_idle(force_room=True)
        self.assertIsNone(reg.get('old','u'));self.assertIs(reg.get('new','u'),new)
        new.save.side_effect=OSError('disk')
        with patch.object(settings,'session_idle_ttl',0):await reg.reap_idle()
        self.assertIs(reg.get('new','u'),new);new.aclose.assert_not_awaited()
    async def test_shutdown_during_initialization_closes_late_client(self):
        reg = SessionRegistry()
        entered, proceed = threading.Event(), threading.Event()
        client = SimpleNamespace(close=AsyncMock())
        def build(*args):
            entered.set()
            proceed.wait(timeout=2)
            return SimpleNamespace(client=client)
        reg._build_agent = build
        build_task = asyncio.create_task(asyncio.to_thread(reg.get_or_create, 'late', 'u'))
        await asyncio.to_thread(entered.wait, 1)
        await reg.close_all(timeout=.03)
        proceed.set()
        with self.assertRaises(SessionCapacityError):
            await build_task
        self.assertEqual(len(reg), 0)
        client.close.assert_awaited_once()

    async def test_shutdown_has_total_budget(self):
        reg=SessionRegistry();cancelled=[]
        async def close():
            try:await asyncio.sleep(10)
            finally:cancelled.append(True)
        for i in range(4):reg._agents[('u',str(i))]=SimpleNamespace(aclose=close)
        start=time.perf_counter();await reg.close_all(timeout=.03)
        self.assertLess(time.perf_counter()-start,.2)
        await asyncio.sleep(0)
        self.assertEqual(len(cancelled),4)
    async def test_reply_saved_before_final_and_maintenance_is_awaitable(self):
        events=[];unblock=asyncio.Event()
        async def memory(*args):events.append('memory-start');await unblock.wait();events.append('memory-done')
        agent=SimpleNamespace(session_path='unused',raw_messages=[],summary=None,user_id='u',history_threshold=10,
            memory_manager=SimpleNamespace(stm_to_dict=lambda:{},aupdate_short_term=memory),
            _compress_history=AsyncMock(),_maybe_consolidate=AsyncMock(),
            _emit=lambda *a,**kw:events.append('final'))
        result=SimpleNamespace(reply='answer',intent=SimpleNamespace(value='greeting'),confidence=1,requires_human=False)
        await finish_turn(agent,result,lambda *a:events.append('saved'))
        self.assertEqual(events[:2],['saved','final'])
        wait=asyncio.create_task(drain_maintenance(agent));await asyncio.sleep(0)
        self.assertFalse(wait.done())
        unblock.set();await wait
        self.assertEqual(events,['saved','final','memory-start','memory-done','saved'])
    async def test_memory_failure_does_not_remove_saved_reply(self):
        emitted=[];save=Mock()
        agent=SimpleNamespace(session_path='unused',raw_messages=[],summary=None,user_id='u',history_threshold=10,
            memory_manager=SimpleNamespace(stm_to_dict=lambda:{},aupdate_short_term=AsyncMock(side_effect=RuntimeError('secret'))),
            _compress_history=AsyncMock(),_maybe_consolidate=AsyncMock(),_emit=lambda *a,**k:emitted.append(a[0]))
        result=SimpleNamespace(reply='a',intent=SimpleNamespace(value='greeting'),confidence=1,requires_human=False)
        await finish_turn(agent,result,save);await drain_maintenance(agent)
        self.assertEqual(emitted,['final']);self.assertEqual(save.call_count,1)


class LoggingTests(unittest.TestCase):
    def test_trace_and_timings_exclude_exception_content(self):
        output=io.StringIO();handler=logging.StreamHandler(output)
        old=logger.level;logger.setLevel(logging.INFO);logger.addHandler(handler)
        token=request_id.set('request-test')
        try:
            try:
                with stage('fake-step'):raise RuntimeError('secret-token-and-chat-content')
            except RuntimeError as exc:failure('test_failed',exc)
        finally:
            request_id.reset(token);logger.removeHandler(handler);logger.setLevel(old)
        text=output.getvalue()
        self.assertNotIn('secret-token-and-chat-content',text)
        rows=[json.loads(line) for line in text.splitlines()]
        self.assertEqual(rows[0]['request_id'],'request-test')
        self.assertIn('elapsed_ms',rows[0]);self.assertTrue(rows[1]['frames'])

if __name__=='__main__':unittest.main()
