"""Offline regressions for identity, paths, session concurrency and HITL."""
import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, Mock
import httpx
from app.config.settings import settings
from app.server import main as server
from app.server.sessions import SessionRegistry, session_path_for
from app.server.schemas import ChatRequest
from app.agent.tools.hitl import ConfirmationGate, resolve_on_agent
from app.agent.tools.manager import ToolManager
from app.agent.chat import YierAgent

TOKEN_A = 'a' * 40
TOKEN_B = 'b' * 40

class FakeAgent:
    def __init__(self, user):
        self.user_id = user
        self.raw_messages = []
        self.on_event = None
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
    async def achat(self, message):
        self.started.set()
        await self.finish.wait()
        self.raw_messages.append({'role': 'assistant', 'content': message})
        self.on_event({'type': 'final', 'reply': message})

class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reg = SessionRegistry()
        self.patches = [patch.object(server, 'registry', self.reg),
            patch.object(settings, 'api_tokens', {'alice':TOKEN_A, 'bob':TOKEN_B}),
            patch.object(settings, 'rate_limit_enabled', False),
            patch.object(settings, 'db_enabled', False)]
        for p in self.patches: p.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test',headers={'Authorization':'Bearer '+TOKEN_A})
    async def asyncTearDown(self):
        await self.client.aclose()
        for p in reversed(self.patches): p.stop()
    async def test_auth_required_and_no_impersonation(self):
        r=await self.client.get('/api/sessions/test',headers={'Authorization':''})
        self.assertEqual(r.status_code,401)
        r=await self.client.post('/api/chat',json={'session_id':'s','user_id':'bob','message':'hi'})
        self.assertEqual(r.status_code,403)
        with patch.object(settings,'api_tokens',{}):
            self.assertEqual((await self.client.get('/api/sessions/s')).status_code,503)
    async def test_same_id_distinct_users(self):
        self.reg._build_agent=lambda sid,uid: SimpleNamespace(user_id=uid,raw_messages=[{'role':'user','content':uid}])
        a=self.reg.get_or_create('same','alice'); b=self.reg.get_or_create('same','bob')
        self.assertIsNot(a,b)
        self.assertEqual((await self.client.get('/api/sessions/same')).json()['messages'][0]['content'],'alice')
        self.assertEqual((await self.client.get('/api/sessions/same',headers={'Authorization':'Bearer '+TOKEN_B})).json()['messages'][0]['content'],'bob')
    async def test_foreign_session_inaccessible(self):
        self.reg._agents[('bob','private')]=SimpleNamespace(raw_messages=[])
        for method in ('get','delete'):
            self.assertEqual((await getattr(self.client,method)('/api/sessions/private')).status_code,404)
        r=await self.client.post('/api/confirm',json={'session_id':'private','token':'x','approved':True})
        self.assertEqual(r.status_code,404)
    async def test_invalid_paths_rejected(self):
        for sid in ('../bad','a/b','a\\b','C:bad','CON','', 'a'*65):
            with self.assertRaises(ValueError): session_path_for('alice',sid)
            r=await self.client.post('/api/chat',json={'session_id':sid,'message':'hi'})
            self.assertEqual(r.status_code,422)
        with self.assertRaises(ValueError): session_path_for('../bad','s')
    async def test_busy_all_endpoints_and_disconnect_finishes(self):
        agent=FakeAgent('alice'); self.reg._agents[('alice','s')]=agent
        response=await server.chat(ChatRequest(session_id='s',message='first'),'alice')
        stream=response.body_iterator
        await anext(stream)
        await agent.started.wait()
        for path,body in [('/api/chat',{'session_id':'s','message':'second'}),('/api/confirm',{'session_id':'s','token':'x','approved':True})]:
            self.assertEqual((await self.client.post(path,json=body)).status_code,409)
        self.assertEqual((await self.client.delete('/api/sessions/s')).status_code,409)
        self.assertEqual((await self.client.get('/api/sessions/s')).status_code,409)
        await stream.aclose()
        self.assertFalse(self.reg.acquire('s','alice'))
        agent.finish.set()
        await asyncio.gather(*list(server._active_tasks))
        self.assertEqual(agent.raw_messages[-1]['content'],'first')
        self.assertTrue(self.reg.acquire('s','alice')); self.reg.release('s','alice')
    async def test_two_users_stream_independently(self):
        a=FakeAgent('alice'); b=FakeAgent('bob')
        self.reg._agents.update({('alice','s'):a,('bob','s'):b})
        responses=[await server.chat(ChatRequest(session_id='s',message=user),user) for user in ('alice','bob')]
        a.finish.set(); b.finish.set()
        for response,user in zip(responses,('alice','bob')):
            events=[json.loads(s[6:]) async for s in response.body_iterator]
            self.assertEqual([e['reply'] for e in events if e['type']=='final'],[user])
    async def test_cold_db_confirm_offloaded_and_scoped(self):
        from app.server import repository
        def load(uid,sid):
            with self.assertRaises(RuntimeError): asyncio.get_running_loop()
            self.assertEqual((uid,sid),('alice','cold'))
            return SimpleNamespace()
        self.reg._build_agent=lambda *a: SimpleNamespace(resolve_hitl=lambda *a:{'ok':False,'error':'expired'})
        with patch.object(settings,'db_enabled',True),patch.object(repository,'sync_load_session_state',side_effect=load):
            r=await self.client.post('/api/confirm',json={'session_id':'cold','token':'old','approved':True})
        self.assertEqual(r.status_code,400)
        self.assertEqual(r.json()['detail'],'expired')
    async def test_delete_closes_resources_and_releases_lease(self):
        agent = SimpleNamespace(raw_messages=[], _ltm_tasks=set(), reset=Mock(),
            tool_manager=SimpleNamespace(close=Mock()), client=SimpleNamespace(close=AsyncMock()))
        self.reg._agents[('alice', 'delete-me')] = agent
        response = await self.client.delete('/api/sessions/delete-me')
        self.assertEqual(response.status_code, 200)
        agent.reset.assert_called_once()
        agent.tool_manager.close.assert_called_once()
        agent.client.close.assert_awaited_once()
        self.assertIsNone(self.reg.get('delete-me', 'alice'))
        self.assertTrue(self.reg.acquire('delete-me', 'alice'))
        self.reg.release('delete-me', 'alice')

    async def test_cancel_repairs_history_and_next_turn_can_proceed(self):
        agent=object.__new__(YierAgent)
        agent.raw_messages=[]; agent.max_react_steps=1; agent.client=None; agent.model='fake'; agent.temperature=0
        agent.hitl=ConfirmationGate(); agent.on_event=lambda e:None
        agent._build_messages=lambda: agent.raw_messages
        entered=asyncio.Event()
        async def tool(*a): entered.set(); await asyncio.sleep(30)
        agent.tool_manager=SimpleNamespace(tool_definitions=[],aexecute_tool=tool)
        call=SimpleNamespace(id='call1',function=SimpleNamespace(name='query_order',arguments='{}'))
        async def llm(*a,**kw): return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None,tool_calls=[call]))])
        with patch('app.agent.chat.llm_create',llm):
            task=asyncio.create_task(agent.achat('question'))
            await asyncio.wait_for(entered.wait(),2); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual([m['role'] for m in agent.raw_messages],['user','assistant','tool'])
        self.assertEqual(agent.raw_messages[-1]['tool_call_id'],'call1')
        from app.schemas.response import CustomerServiceResponse, IntentType
        result = CustomerServiceResponse(reply='recovered', intent=IntentType.GREETING, confidence=1, requires_human=False)
        async def next_llm(*args, **kwargs):
            history = kwargs['messages']
            self.assertEqual(history[-2]['role'], 'tool')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='recovered', tool_calls=None))])
        agent._extract_structured_response = AsyncMock(return_value=result)
        agent.memory_manager = SimpleNamespace(aupdate_short_term=AsyncMock(), stm_to_dict=lambda: {})
        agent.history_threshold=100; agent.session_path='unused'; agent.summary=None; agent.user_id='alice'
        agent._maybe_consolidate=AsyncMock()
        with patch('app.agent.chat.llm_create', next_llm), patch('app.agent.chat.save_session'):
            self.assertEqual((await agent.achat('next question')).reply, 'recovered')

class GateTests(unittest.TestCase):
    def setUp(self):
        self.p=patch.object(settings,'hitl_enabled',True);self.p.start()
        self.g=ConfirmationGate();self.args={'order_id':'ORD-20240120-002','reason':'original'}
    def tearDown(self): self.p.stop()
    def test_changed_arguments_invalidate_old_token(self):
        old=self.g.intercept('apply_refund',self.args)['confirm_token']
        args={**self.args,'reason':'changed'}
        new=self.g.intercept('apply_refund',args)['confirm_token']
        self.assertNotEqual(old,new); self.assertFalse(self.g.resolve(old,True)['ok'])
        args['reason']='mutated'
        self.assertEqual(self.g.resolve(new,True)['args']['reason'],'changed')
    def test_single_execution_and_cached_retry(self):
        tm=ToolManager(hitl=self.g)
        token=json.loads(tm.execute_tool('apply_refund',self.args))['confirm_token']
        agent=SimpleNamespace(hitl=self.g,tool_manager=tm,raw_messages=[])
        with patch('app.agent.tools.manager.local_execute_tool',return_value='{"success":true}') as execute:
            result=resolve_on_agent(agent,token,True)
            self.assertTrue(result['executed'])
            self.assertFalse(resolve_on_agent(agent,token,True)['ok'])
            self.assertTrue(json.loads(tm.execute_tool('apply_refund',self.args))['success'])
            self.assertEqual(execute.call_count,1)
            changed=json.loads(tm.execute_tool('apply_refund',{**self.args,'reason':'different'}))
            self.assertEqual(changed['status'],'confirm_required')
    def test_expiration_and_ambiguous_yes(self):
        with patch.object(settings,'hitl_token_ttl',-1):
            token=self.g.intercept('apply_refund',self.args)['confirm_token']
        self.assertFalse(self.g.resolve(token,True)['ok'])
        self.g.intercept('apply_refund',self.args)
        self.g.intercept('apply_refund',{**self.args,'order_id':'another'})
        self.assertIsNone(self.g.consume_utterance('确认'))
    def test_deny_and_single_use_grant(self):
        token=self.g.intercept('apply_refund',self.args)['confirm_token']
        self.g.resolve(token,True)
        self.assertIsNone(self.g.intercept('apply_refund',self.args))
        self.assertEqual(self.g.intercept('apply_refund',self.args)['status'],'execution_unknown')

if __name__=='__main__': unittest.main()
