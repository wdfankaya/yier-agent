"""Authenticated HTTP integration with real GraphAgent and SQLite, fake model/tool."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import httpx
from app.config.settings import settings
from app.agent.workflow import GraphAgent
from app.agent.tools.manager import ToolManager
from app.schemas.response import CustomerServiceResponse,IntentType
from app.server import main as server,sessions
from tests.test_workflow import call


class WorkflowAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Path('app/sessions').mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir='app/sessions')
        root=Path(self.tmp.name)
        self.stack=ExitStack()
        overrides={'workflow_enabled':True,'workflow_backend':'sqlite',
            'workflow_sqlite_path':str(root/'graph.db'),'workflow_ledger_path':str(root/'ledger.db'),
            'trace_db_path':str(root/'trace.db'),'trace_enabled':True,'db_enabled':False,
            'memory_enabled':False,'mcp_enabled':False,'skills_enabled':False,'redis_enabled':False,
            'rate_limit_enabled':False,'openai_api_key':'fake-local-only',
            'api_tokens':{'alice':'a'*40,'bob':'b'*40}}
        for key,value in overrides.items(): self.stack.enter_context(patch.object(settings,key,value))
        self.reg=sessions.SessionRegistry()
        self.stack.enter_context(patch.object(sessions,'SESSION_DIR',root/'server'))
        self.stack.enter_context(patch.object(server,'registry',self.reg))
        self.stack.enter_context(patch.object(server,'_stopping',False))
        self.effects=0
        async def model(agent,messages):
            return {'role':'assistant','tool_calls':[call()]} if messages[-1]['role']=='user' else {'role':'assistant','content':'模拟退款完成'}
        async def structured(agent,text):
            return CustomerServiceResponse(intent=IntentType.RETURN_REQUEST,confidence=1,reply=text)
        async def execute(manager,name,args):
            self.effects+=1
            return json.dumps({'success':True})
        self.stack.enter_context(patch.object(GraphAgent,'_model',model))
        self.stack.enter_context(patch.object(GraphAgent,'_extract_structured_response',structured))
        self.stack.enter_context(patch.object(ToolManager,'aexecute_tool',execute))
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test',headers={'Authorization':'Bearer '+'a'*40})

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.reg.close_all(timeout=5)
        self.stack.close()
        self.tmp.cleanup()

    async def test_restart_restore_confirm_trace_and_user_isolation(self):
        chat=await self.client.post('/api/chat',json={'session_id':'s1','message':'退款'})
        self.assertEqual(chat.status_code,200)
        self.assertIn('confirm_required',chat.text)
        rid=chat.headers['X-Request-ID']
        status=await self.client.get('/api/workflows/s1')
        token=status.json()['pending']['confirm_token']
        self.assertEqual(self.effects,0)
        # Emulate registry loss AND the JSON save being lost after a crash.
        await self.reg.close_all(timeout=5)
        Path(sessions.session_path_for('alice','s1')).unlink(missing_ok=True)
        self.reg=sessions.SessionRegistry()
        server.registry=self.reg
        restored=await self.client.get('/api/workflows/s1')
        self.assertEqual(restored.status_code,200)
        self.assertEqual(restored.json()['pending']['confirm_token'],token)
        foreign={'Authorization':'Bearer '+'b'*40}
        self.assertEqual((await self.client.get('/api/workflows/s1',headers=foreign)).status_code,404)
        self.assertEqual((await self.client.get('/api/traces/'+rid,headers=foreign)).status_code,404)
        wrong=await self.client.post('/api/confirm',json={'session_id':'s1','token':'wrong','approved':True})
        self.assertEqual(wrong.status_code,400)
        body={'session_id':'s1','token':token,'approved':True}
        approved=await self.client.post('/api/confirm',json=body)
        self.assertEqual(approved.status_code,200,approved.text)
        self.assertTrue(approved.json()['executed'])
        self.assertEqual(approved.json()['reply'],'模拟退款完成')
        self.assertEqual((await self.client.post('/api/confirm',json=body)).status_code,400)
        self.assertEqual(self.effects,1)
        trace=await self.client.get('/api/traces/'+rid)
        self.assertEqual(trace.status_code,200)
        self.assertIsNotNone(trace.json()['metrics']['duration_ms'])
        self.assertIsNotNone(trace.json()['metrics']['first_response_ms'])


if __name__=='__main__': unittest.main()
