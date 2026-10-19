"""Real SQLite checkpoints; injected model/tools, no remote requests."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from app.agent.workflow import compile_workflow, EffectLedger, thread_key


def call(order='ORD-20240120-002', reason='不要了'):
    return {'id':'c1','type':'function','function':{'name':'apply_refund',
            'arguments':json.dumps({'order_id':order,'reason':reason})}}


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Path('app/sessions').mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir='app/sessions')
        self.db=str(Path(self.tmp.name)/'checkpoint.db')
        self.ledger=EffectLedger(Path(self.tmp.name)/'effects.db')
        self.effects=0
        self.config={'configurable':{'thread_id':thread_key('alice','s1')}}

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def model(self,messages):
        if messages[-1]['role']=='user':
            return {'role':'assistant','content':None,'tool_calls':[call()]}
        return {'role':'assistant','content':'完成'}

    async def execute(self,name,args):
        self.effects+=1
        return {'success':True,'order_id':'ORD-001'}

    async def invoke(self,value,model=None,execute=None,ttl=300,max_steps=5,config=None):
        async with AsyncSqliteSaver.from_conn_string(self.db) as saver:
            graph=compile_workflow(model or self.model,execute or self.execute,
                lambda n:n=='apply_refund',saver,self.ledger,'alice/s1',max_steps,ttl,'alice')
            return await graph.ainvoke(value,config or self.config)

    async def start(self,**kw):
        return await self.invoke({'messages':[{'role':'user','content':'退款'}],'steps':0},**kw)

    def resume(self,state,approved=True,token=None):
        return Command(resume={'token':token or state['pending']['confirm_token'],'approved':approved})

    async def test_restart_approve_once(self):
        state=await self.start()
        self.assertIn('__interrupt__',state)
        self.assertEqual(self.effects,0)
        # invoke opens a completely new SQLite saver and compiled graph each time.
        done=await self.invoke(self.resume(state))
        self.assertTrue(done['approval_result']['success'])
        self.assertEqual(self.effects,1)
        await self.invoke(self.resume(state))
        self.assertEqual(self.effects,1)

    async def test_denied(self):
        state=await self.start()
        done=await self.invoke(self.resume(state,False))
        self.assertEqual(done['approval_result']['status'],'denied')
        self.assertEqual(self.effects,0)

    async def test_wrong_token(self):
        state=await self.start()
        done=await self.invoke(self.resume(state,token='wrong'))
        self.assertEqual(done['approval_result']['status'],'denied')
        self.assertEqual(self.effects,0)

    async def test_expired(self):
        state=await self.start(ttl=-1)
        done=await self.invoke(self.resume(state),ttl=-1)
        self.assertEqual(done['approval_result']['status'],'expired')
        self.assertEqual(self.effects,0)

    async def test_crash_after_effect_recovers_without_duplicate(self):
        state=await self.start()
        async def broken(messages):
            raise RuntimeError('injected model crash')
        with self.assertRaises(RuntimeError):
            await self.invoke(self.resume(state),model=broken)
        self.assertEqual(self.effects,1)
        done=await self.invoke(None)
        self.assertEqual(done['final'],'完成')
        self.assertEqual(self.effects,1)

    async def test_unknown_is_not_reexecuted_in_new_session(self):
        async def unknown(name,args):
            self.effects+=1
            return {'status':'execution_unknown'}
        state=await self.start()
        await self.invoke(self.resume(state),execute=unknown)
        state=await self.start(config={'configurable':{'thread_id':'another-session'}})
        done=await self.invoke(self.resume(state),config={'configurable':{'thread_id':'another-session'}})
        self.assertEqual(done['approval_result']['status'],'execution_unknown')
        self.assertEqual(self.effects,1)

    async def test_changed_parameters_need_new_token(self):
        state=await self.start()
        await self.invoke(self.resume(state,False))
        async def changed(messages):
            if messages[-1]['role']=='user':
                return {'role':'assistant','tool_calls':[call('ORD-002')]}
            return {'role':'assistant','content':'完成'}
        new=await self.start(model=changed)
        self.assertNotEqual(state['pending']['confirm_token'],new['pending']['confirm_token'])
        done=await self.invoke(self.resume(new,token=state['pending']['confirm_token']))
        self.assertEqual(done['approval_result']['status'],'denied')
        self.assertEqual(self.effects,0)

    async def test_step_budget(self):
        async def looping(messages):
            return {'role':'assistant','tool_calls':[{'id':'r','type':'function','function':{'name':'read','arguments':'{}'}}]}
        done=await self.start(model=looping,max_steps=2)
        self.assertIn('上限',done['final'])
        self.assertEqual(self.effects,2)

    async def test_user_checkpoint_isolation(self):
        await self.start()
        async with AsyncSqliteSaver.from_conn_string(self.db) as saver:
            self.assertIsNone(await saver.aget_tuple({'configurable':{'thread_id':thread_key('bob','s1')}}))

    async def test_atomic_claim(self):
        results=await asyncio.gather(*(asyncio.to_thread(self.ledger.claim,'same') for _ in range(12)))
        self.assertEqual(sum(r is None for r in results),1)
        self.assertTrue(all(r is None or r['status']=='execution_unknown' for r in results))


if __name__=='__main__':
    unittest.main()
