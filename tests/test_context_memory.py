"""Offline regression for budgets, evidence, expiry, retrieval and shared state."""
import asyncio
import copy
import json
import multiprocessing
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from app.config.settings import settings
from app.agent.context import prepare_call, ContextBudgetExceeded, interaction_units, tool_fields
from app.agent.context_summary import rebuild, verified, prompt_summary, delete_archive
from app.agent.memory.facts import extract, merge, select, MemoryFact
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.manager import MemoryManager
from app.agent.memory.short_term import ShortTermMemory


def message(text, observed=None):
    return {'role':'user','content':text,'_observed_at':observed or time.time()}


def process_write(directory, index):
    settings.db_enabled = False
    memory = LongTermMemory('alice', directory, source_session=str(index))
    memory.load()
    memory.add_facts(extract([message('我喜欢颜色'+str(index))], str(index)))
    memory.save()


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(settings, 'db_enabled', False))
        self.stack.enter_context(patch.object(settings, 'memory_semantic_enabled', False))
        self.stack.enter_context(patch.object(settings, 'summary_archive_path', str(Path(self.temp.name)/'archive.db')))
    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()
    def memory(self, session='s', **kw):
        m = LongTermMemory('alice', self.temp.name, source_session=session, **kw)
        m.load()
        return m


class BudgetTests(Base):
    def small(self):
        for k,v in {'context_window_tokens':4096,'context_output_tokens':512,'context_safety_tokens':256}.items():
            self.stack.enter_context(patch.object(settings,k,v))
    def test_single_huge_user_fails_before_network(self):
        self.small()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
        from app.agent.resilience import llm_create
        with self.assertRaises(ContextBudgetExceeded):
            asyncio.run(llm_create(client, model='fake', messages=[message('长'*10000)]))
        client.chat.completions.create.assert_not_called()
    def test_output_reserved_and_schema_counts(self):
        self.small()
        call = prepare_call({'messages':[message('你好')], 'max_tokens':10000})
        self.assertEqual(call['max_tokens'],512)
        with self.assertRaises(ContextBudgetExceeded):
            prepare_call({'messages':[message('hi')], 'tools':[{'description':'x'*5000}]})
    def test_trim_complete_units_and_no_mutation(self):
        self.small()
        source = [message('旧消息'*800), {'role':'assistant','tool_calls':[{'id':'a','type':'function','function':{'name':'lookup','arguments':'{}'}}]},
                  {'role':'tool','tool_call_id':'a','content':'{}'}, {'role':'assistant','content':'完成'}, message('新的问题')]
        original = copy.deepcopy(source)
        result = prepare_call({'messages':source})['messages']
        self.assertEqual(source,original)
        self.assertEqual(result[-1]['content'],'新的问题')
        self.assertFalse(any(m['role']=='tool' for m in result))
        self.assertEqual([len(x) for x in interaction_units(source)],[4,1])
    def test_tool_tail_fields_survive(self):
        self.small()
        payload = {'irrelevant':'x'*10000,'order':{'order_id':'ORD-99','amount':129.5,'status':'paid'},'success':True}
        messages = [message('查订单'),{'role':'assistant','tool_calls':[{'id':'a','type':'function','function':{'name':'lookup','arguments':'{}'}}]},
                    {'role':'tool','tool_call_id':'a','content':json.dumps(payload)}]
        result=prepare_call({'messages':messages})['messages']
        self.assertEqual(result[-1]['tool_call_id'],'a')
        facts=json.loads(result[-1]['content'])['fields']
        self.assertEqual(facts['order.amount'],129.5)
        self.assertEqual(facts['order.order_id'],'ORD-99')
        self.assertTrue(facts['success'])
    def test_hidden_metadata_never_sent(self):
        self.assertNotIn('_observed_at', prepare_call({'messages':[message('hi')]})['messages'][0])


class SummaryTests(Base):
    def test_verified_rebuild_corrects_cache_and_new_revision(self):
        source=message('订单 ORD-99 金额 129.5元')
        raw=rebuild('alice','s',additions=[source])
        data=json.loads(raw)
        data['summary']={'evidence':[{'quote':'invented'}]}
        fixed=verified(json.dumps(data),'alice','s')
        self.assertNotIn('invented',json.dumps(fixed))
        sid=fixed['source_ids'][0]
        newer=rebuild('alice','s',excluded=[sid])
        self.assertEqual(json.loads(newer)['revision'],2)
        self.assertEqual(verified(raw,'alice','s')['summary']['evidence'],[])
    def test_cross_user_and_legacy_summary_rejected(self):
        raw=rebuild('alice','s',additions=[message('我喜欢黑色')])
        self.assertIsNone(verified(raw,'bob','s'))
        self.assertEqual(prompt_summary('模型编造的旧摘要','alice','s'),'')
        with self.assertRaises(ValueError): rebuild('alice','other',excluded=[json.loads(raw)['source_ids'][0]])
    def test_original_tool_source_and_amount_retained(self):
        raw=rebuild('alice','s',additions=[{'role':'tool','tool_call_id':'x','content':json.dumps({'padding':'x'*1000,'order_id':'ORD-8','amount':200,'success':False})}])
        item=verified(raw,'alice','s')['summary']['evidence'][0]
        self.assertEqual(item['fields']['amount'],200)
        self.assertFalse(item['fields']['success'])
        delete_archive('alice','s')
        self.assertIsNone(verified(raw,'alice','s'))
    def test_tampered_source_rejected(self):
        from app.agent.context_summary import archive
        raw=rebuild('alice','s',additions=[message('我喜欢黑色')])
        with archive() as db: db.execute("UPDATE sources SET payload='{}'")
        self.assertIsNone(verified(raw,'alice','s'))


class FactTests(Base):
    def test_preference_update_not_completion_order(self):
        now=time.time()
        old=extract([message('我不喜欢黑色',now-20)],'old')
        new=extract([message('最近喜欢黑色',now)],'new')
        a,b=self.memory('old'),self.memory('new')
        b.add_facts(new); b.save()
        a.add_facts(old); a.save()
        facts=self.memory().recall('黑色')
        self.assertEqual(len(facts),1)
        self.assertEqual(facts[0].value,'like')
    def test_two_updates_in_same_message(self):
        facts=merge([],extract([message('以前不喜欢黑色现在喜欢黑色')],'s'),50)
        self.assertEqual([f.value for f in facts if f.active()],['like'])
    def test_source_time_stays_stable_on_reextraction(self):
        source={'role':'user','content':'我喜欢黑色'}
        a=extract([source],'s',now=100)
        b=extract([source],'s',now=200)
        self.assertEqual(a[0].fact_id,b[0].fact_id)
        self.assertEqual(a[0].observed_at,b[0].observed_at)
    def test_expiry_does_not_reactivate_old_preference(self):
        now=time.time()
        facts=extract([message('我不喜欢黑色',now-10),message('最近喜欢黑色',now)],'s')
        facts[0].expires_at=now+10000
        facts[1].expires_at=now+1
        result=merge([],facts,50,now+2)
        self.assertEqual(select(result,'黑色',now=now+2),[])
        result=merge(result,[facts[0]],50,now+3)
        self.assertEqual(select(result,'黑色',now=now+3),[])
    def test_dynamic_state_is_never_current_memory(self):
        facts=extract([message('我是黄金会员'),{'role':'tool','content':'{"status":"paid","amount":10}'}],'s')
        self.assertEqual(len(facts),3)
        self.assertTrue(all(f.requires_refresh for f in facts))
        self.assertEqual(select(facts),[])
    def test_unverified_inferred_third_party_and_question_excluded(self):
        self.assertEqual(extract([message('如果我喜欢黑色怎么办？她说喜欢红色。')],'s'),[])
        legacy=MemoryFact('喜欢黑色','preference','2020-01-01')
        self.assertEqual(select([legacy]),[])
    def test_query_category_topk_and_session_scope(self):
        facts=extract([message('我喜欢黑色。我的鞋码是42。'),message('这次喜欢蓝色')],'s')
        self.assertEqual([f.attribute for f in select(facts,'黑色',top_k=1)],['preference:黑色'])
        self.assertEqual(select(facts,'黑色',category='identity'),[])
        self.assertEqual(select(facts,'蓝色',scope='session:other'),[])
        self.assertEqual(len(select(facts,'蓝色',scope='session:s')),1)
    def test_eviction_uses_importance_and_access_not_append_order(self):
        now=time.time()
        facts=extract([message('我喜欢黑色',now-100),message('我喜欢蓝色',now)],'s')
        facts[0].importance=1; facts[0].stability=1
        facts[1].importance=0; facts[1].stability=0
        kept=merge([],facts,1)
        self.assertEqual([f.attribute for f in kept if f.active()],['preference:黑色'])
        m=self.memory(); m.add_facts([facts[0]]); m.save()
        m.recall('黑色')
        self.assertGreater(self.memory().facts[0].last_accessed,now-1)
    def test_reset_discards_pending_old_generation(self):
        stale=self.memory(); stale.add_facts(extract([message('我喜欢黑色')],'s'))
        self.memory().reset(); stale.save()
        self.assertEqual(self.memory().facts,[])
    def test_parallel_process_writes_preserve_union(self):
        ctx=multiprocessing.get_context('spawn')
        workers=[ctx.Process(target=process_write,args=(self.temp.name,i)) for i in range(4)]
        for p in workers: p.start()
        for p in workers:
            p.join(20)
            if p.is_alive(): p.terminate(); p.join(); self.fail('writer timeout')
            self.assertEqual(p.exitcode,0)
        self.assertEqual(len(self.memory().facts),4)
    def test_prompt_reads_snapshot_without_sync_io(self):
        m=self.memory(); m.add_facts(extract([message('我喜欢黑色')],'s')); m.save()
        with patch('app.agent.memory.long_term.transaction',side_effect=AssertionError('sync I/O')):
            self.assertIn('黑色',m.build_prompt_section('黑色'))
    def test_tool_query_and_context_reset(self):
        from app.agent.tools.memory_tool import recall_user_memory, bind_memory_manager, get_current_memory_manager
        manager=MemoryManager(None,'fake',user_id='alice',memory_dir=self.temp.name,source_session='s')
        asyncio.run(manager.aupdate_short_term([message('我喜欢黑色。我的鞋码是42。')]))
        with bind_memory_manager(manager):
            result=recall_user_memory('黑色')
            self.assertEqual(len(result['long_term_facts']),1)
            with self.assertRaises(RuntimeError):
                with bind_memory_manager(None): raise RuntimeError()
            self.assertIs(get_current_memory_manager(),manager)
        self.assertIsNone(get_current_memory_manager())


class StorageTests(Base):
    def test_json_metadata_roundtrip(self):
        from app.agent.storage import save_session, load_session
        messages=[message('我喜欢黑色')]
        extract(messages,'s')
        path=str(Path(self.temp.name)/'session.json')
        save_session(path,messages,None,{},'alice')
        self.assertEqual(load_session(path,'alice')['messages'],messages)
    def test_pg_metadata_roundtrip_without_schema_change(self):
        from app.agent.storage import save_session, load_session
        messages=[message('我喜欢黑色')]; extract(messages,'s')
        captured={}
        def save(sid,msgs,summary,stm):
            captured.update(messages=[{k:v for k,v in m.items() if not k.startswith('_')} for m in msgs],summary=summary,stm=stm)
        cache=Mock(); cache.get_session.return_value=None
        with patch.object(settings,'db_enabled',True), patch('app.agent.storage._cache',return_value=cache), patch('app.server.repository.sync_get_or_create_session',return_value=1), patch('app.server.repository.sync_replace_session_state',side_effect=save), patch('app.server.repository.sync_load_session_state',side_effect=lambda *a:SimpleNamespace(**captured)):
            save_session('s.json',messages,None,{},'alice')
            self.assertEqual(load_session('s.json','alice')['messages'],messages)


class AdditionalTests(Base):
    def test_session_stm_reconciles_newer_shared_fact_and_reset(self):
        a=MemoryManager(None,'fake','alice',self.temp.name,source_session='a')
        b=MemoryManager(None,'fake','alice',self.temp.name,source_session='b')
        asyncio.run(a.aupdate_short_term([message('我不喜欢黑色',time.time()-10)]))
        asyncio.run(b.aupdate_short_term([message('我喜欢黑色')]))
        asyncio.run(a.aprepare())
        self.assertEqual(a.stm.recall('黑色'),[])
        self.assertEqual(a.ltm.recall('黑色')[0].value,'like')
        b.reset_all(); asyncio.run(a.aprepare())
        self.assertEqual(a.stm.recall(),[])
        self.assertEqual(a.ltm.recall(),[])
    def test_archive_search_and_source_read_respect_scope_and_correction(self):
        from app.agent.context_summary import search_sources, read_source
        raw=rebuild('alice','s',additions=[{'role':'tool','tool_call_id':'t',
            'content':json.dumps({'order_id':'ORD-77','amount':250,'status':'paid'})}])
        sid=json.loads(raw)['source_ids'][0]
        self.assertEqual(search_sources('alice','s','ORD-77')[0]['fields']['amount'],250)
        self.assertIsNone(read_source('bob','s',sid))
        self.assertEqual(read_source('alice','s',sid)['tool_call_id'],'t')
        rebuild('alice','s',excluded=[sid])
        self.assertEqual(search_sources('alice','s','ORD-77'),[])
    def test_console_encoding_cannot_abort_tool_event(self):
        import io
        from app.agent.chat import YierAgent
        output=io.BytesIO()
        stream=io.TextIOWrapper(output,encoding='ascii')
        agent=YierAgent.__new__(YierAgent); agent.on_event=None
        with patch('sys.stdout',stream): agent._emit_tool_call('lookup',{'value':'订单'})
        stream.flush()
        self.assertIn(b'lookup',output.getvalue())
        stream.detach()
    def test_reset_watermark_survives_refresh_and_reextraction(self):
        old=[message('我喜欢黑色',time.time()-10)]
        m=self.memory(); m.add_facts(extract(old,'s')); m.save()
        m.reset(); m.refresh()
        m.add_facts(extract(old,'s')); m.save()
        self.assertEqual(self.memory().facts,[])
        m.add_facts(extract([message('我喜欢蓝色')],'s')); m.save()
        self.assertEqual(self.memory().facts[0].attribute,'preference:蓝色')
    def test_optional_memory_cannot_crowd_out_current_turn(self):
        with patch.object(settings,'context_window_tokens',4096), patch.object(settings,'context_output_tokens',512):
            result=prepare_call({'messages':[{'role':'system','content':'Core instructions'},
                {'role':'system','content':'x'*10000,'_context_optional':True},message('hi')]})
        self.assertEqual(len(result['messages']),2)
        self.assertEqual(result['messages'][0]['content'],'Core instructions')
    def test_negation_questions_and_stm_roundtrip(self):
        self.assertEqual(extract([message('我喜欢黑色？')],'s'),[])
        for text in ('我不太喜欢黑色','我不偏好黑色'):
            self.assertEqual(extract([message(text)],'s')[0].value,'avoid')
        stm=ShortTermMemory('s')
        asyncio.run(stm.aupdate(None,'fake',[message('我不喜欢黑色',time.time()-10),message('我喜欢黑色')]))
        restored=ShortTermMemory.from_dict(stm.to_dict(),'s')
        self.assertEqual([f['value'] for f in restored.recall('黑色')],['like'])
        restored.reset(); self.assertEqual(restored.recall(),[])
    def test_evicted_expired_head_survives_archive_limit(self):
        now=time.time()
        old=extract([message('我不喜欢黑色',now-20)],'s')[0]
        head=extract([message('我喜欢黑色',now-10)],'s')[0]
        head.expires_at=now-1
        with patch.object(settings,'memory_archive_limit',1):
            rows=merge([], [old,head]+extract([message('我喜欢蓝色')],'s'),1)
            rows=merge(rows,[old],1)
        self.assertEqual(select(rows,'黑色'),[])


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path.cwd())
        self.stack=ExitStack()
        root=Path(self.temp.name)
        for key,value in {'db_enabled':False,'memory_semantic_enabled':False,'memory_enabled':True,'memory_dir':str(root/'memory'),
            'summary_archive_path':str(root/'summary.db'),'mcp_enabled':False,'skills_enabled':False,
            'openai_api_key':'fake','history_threshold':3,'ltm_consolidate_every':0,
            'workflow_backend':'sqlite','workflow_sqlite_path':str(root/'graph.db'),
            'workflow_ledger_path':str(root/'ledger.db'),'trace_enabled':False}.items():
            self.stack.enter_context(patch.object(settings,key,value))
    async def asyncTearDown(self):
        self.stack.close(); self.temp.cleanup()
    async def test_all_three_agent_paths_with_real_memory_and_injected_model(self):
        from app.agent.chat import YierAgent
        from app.agent.workflow import GraphAgent
        from app.multi_agent.orchestrator import MultiAgentOrchestrator
        from app.agent.maintenance import drain_maintenance
        from app.schemas.response import CustomerServiceResponse, IntentType
        result=CustomerServiceResponse(intent=IntentType.OTHER,confidence=1,reply='已记录')
        for cls in (YierAgent,GraphAgent,MultiAgentOrchestrator):
            with self.subTest(agent=cls.__name__):
                observed=[]
                async def create(**kwargs):
                    observed.append(kwargs)
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='已记录',tool_calls=[]))])
                fake=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                    beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))]))))),close=AsyncMock())
                target='app.multi_agent.orchestrator.AsyncOpenAI' if cls is MultiAgentOrchestrator else 'app.agent.chat.AsyncOpenAI'
                with patch(target,return_value=fake):
                    agent=cls(str(Path(self.temp.name)/(cls.__name__+'.json')),user_id=cls.__name__)
                agent.on_event=lambda *a,**kw:None
                if cls is MultiAgentOrchestrator: agent.router.aroute=AsyncMock(return_value='presale')
                try:
                    await agent.achat('我不喜欢黑色'); await drain_maintenance(agent)
                    await agent.achat('最近喜欢黑色'); await drain_maintenance(agent)
                    await agent.achat('黑色'); await drain_maintenance(agent)
                    recalled=await asyncio.to_thread(agent.memory_manager.ltm.recall,'黑色')
                    self.assertEqual([f.value for f in recalled],['like'])
                    self.assertTrue(agent.summary)
                    self.assertIsNotNone(verified(agent.summary,agent.user_id,agent.session_path))
                    self.assertTrue(observed)
                    self.assertTrue(all('max_tokens' in call for call in observed))
                    self.assertTrue(all(not any(k.startswith('_') for k in m) for call in observed for m in call['messages']))
                finally: await agent.aclose()
    async def test_authenticated_summary_correction_and_busy_protection(self):
        import httpx
        from app.server import main as server
        from app.server.sessions import SessionRegistry
        reg=SessionRegistry()
        raw=rebuild('alice','s',additions=[message('订单 ORD-7 金额 200元')])
        agent=SimpleNamespace(summary=raw,session_path='s',save=Mock())
        reg._agents[('alice','s')]=agent
        with patch.object(server,'registry',reg),patch.object(server,'_stopping',False),patch.object(settings,'api_tokens',{'alice':'a'*40,'bob':'b'*40}):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test',headers={'Authorization':'Bearer '+'a'*40}) as client:
                path='/api/sessions/s/summary'
                reply=await client.get(path)
                self.assertEqual(reply.status_code,200)
                sid=reply.json()['summary']['source_ids'][0]
                self.assertEqual((await client.get(path,headers={'Authorization':''})).status_code,401)
                self.assertEqual((await client.get(path,headers={'Authorization':'Bearer '+'b'*40})).status_code,404)
                self.assertEqual((await client.post(path+'/rebuild',json={'excluded_source_ids':['unknown']})).status_code,400)
                self.assertTrue(reg.acquire('s','alice'))
                self.assertEqual((await client.post(path+'/rebuild',json={})).status_code,409)
                reg.release('s','alice')
                fixed=await client.post(path+'/rebuild',json={'excluded_source_ids':[sid]})
                self.assertEqual(fixed.status_code,200)
                self.assertEqual(fixed.json()['summary']['summary']['evidence'],[])
                agent.save.assert_called_once()


class PreferenceBoundaryTests(Base):
    def test_product_scopes_do_not_overwrite_each_other(self):
        facts=merge([],extract([message('我买鞋喜欢黑色，衣服不喜欢黑色')],'s'),50)
        shoes=select(facts,'黑色鞋子')
        clothes=select(facts,'黑色衣服')
        self.assertEqual([f.value for f in shoes],['like'])
        self.assertEqual([f.value for f in clothes],['avoid'])
        self.assertIn('鞋',shoes[0].source_quote)
        self.assertEqual(select(facts,'黑色裤子'),[])
    def test_today_expires_at_shanghai_midnight(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        observed=datetime(2026,10,20,18,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        facts=extract([message('今天喜欢黑色。喜欢蓝色',observed)],'s')
        expected=datetime(2026,10,21,0,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        self.assertEqual(facts[0].expires_at,expected)
        self.assertGreater(facts[1].expires_at,expected+86400)
        self.assertEqual(select(facts,'黑色',now=expected),[])
    def test_uncertain_ironic_and_unsupported_time_are_not_confirmed(self):
        for value in ('我可能喜欢黑色','我喜欢黑色才怪','下周我喜欢黑色','我以前喜欢黑色'):
            with self.subTest(value=value): self.assertEqual(extract([message(value)],'s'),[])
    def test_negation_and_correction_keep_explicit_sources(self):
        facts=merge([],extract([message('我不喜欢黑色，而是喜欢白色')],'s'),50)
        self.assertEqual(select(facts,'黑色')[0].value,'avoid')
        self.assertEqual(select(facts,'白色')[0].value,'like')
    def test_current_session_preference_does_not_replace_global(self):
        facts=merge([],extract([message('我喜欢黑色。今天这次不喜欢黑色')],'s'),50)
        self.assertEqual([f.value for f in select(facts,'黑色',scope='session:other')],['like'])
        self.assertEqual([f.value for f in select(facts,'黑色',scope='session:s')],['avoid'])


class SemanticTests(Base):
    def prepare_memory(self):
        m=self.memory(); m.add_facts(extract([message('我喜欢黑色')],'s')); m.save()
        return m
    def test_semantic_recall_and_failure_fallback(self):
        m=self.prepare_memory(); fact=m.facts[0]
        with patch.object(settings,'memory_semantic_enabled',True),patch.object(m._semantic_index,'scores',return_value={fact.fact_id:.8}):
            self.assertEqual(m.recall('深色')[0].value,'like')
        with patch.object(settings,'memory_semantic_enabled',True),patch.object(m._semantic_index,'scores',side_effect=TimeoutError):
            self.assertEqual(m.recall('深色'),[])
            self.assertEqual(len(m.recall('黑色')),1)
    def test_semantic_scores_cannot_bypass_expiry_scope_or_trust(self):
        facts=extract([message('这次喜欢黑色')],'s')
        scores={f.fact_id:.99 for f in facts}
        self.assertEqual(select(facts,'深色',scope='session:other',semantic_scores=scores),[])
        facts[0].expires_at=time.time()-1
        self.assertEqual(select(facts,'深色',scope='session:s',semantic_scores=scores),[])
        facts[0].expires_at=None; facts[0].source_type='model_inference'
        self.assertEqual(select(facts,'深色',scope='session:s',semantic_scores=scores),[])
    def test_embedding_cache_avoids_repeated_calls(self):
        from app.agent.memory.semantic import SemanticMemoryIndex
        fake=Mock(); fake.__enter__=Mock(return_value=fake); fake.__exit__=Mock(return_value=False)
        fake.embeddings.create.return_value=SimpleNamespace(data=[SimpleNamespace(index=0,embedding=[1.,0.]),SimpleNamespace(index=1,embedding=[.9,.1])])
        facts=extract([message('我喜欢黑色')],'s')
        with patch('app.agent.memory.semantic.OpenAI',return_value=fake):
            index=SemanticMemoryIndex(); index.scores('深色',facts); index.scores('深色',facts)
        fake.embeddings.create.assert_called_once()
    def test_reset_during_embedding_cannot_return_forgotten_fact(self):
        m=self.prepare_memory(); f=m.facts[0]
        def delayed(query,facts):
            self.memory().reset()
            return {f.fact_id:.99}
        with patch.object(settings,'memory_semantic_enabled',True),patch.object(m._semantic_index,'scores',side_effect=delayed):
            self.assertEqual(m.recall('深色'),[])
    def test_prompt_semantic_scores_are_prepared_off_loop(self):
        m=self.prepare_memory(); f=m.facts[0]
        with patch.object(settings,'memory_semantic_enabled',True),patch.object(m._semantic_index,'scores',return_value={f.fact_id:.8}):
            m.prepare_semantic('深色')
        with patch.object(m._semantic_index,'scores',side_effect=AssertionError('network on prompt path')):
            self.assertIn('黑色',m.build_prompt_section('深色'))


class ProviderContextTests(Base):
    def test_tool_context_survives_storage_and_preflight(self):
        from app.agent.history import retain_tool_context
        from app.agent.storage import save_session, load_session
        original=retain_tool_context({'role':'assistant','content':None,'tool_calls':[]},SimpleNamespace(reasoning_content='opaque test token'))
        path=str(Path(self.temp.name)/'provider.json')
        save_session(path,[original],None,{},'alice')
        loaded=load_session(path,'alice')['messages']
        prepared=prepare_call({'messages':loaded})['messages'][0]
        self.assertEqual(prepared['reasoning_content'],'opaque test token')
        self.assertNotIn('_reasoning_content',prepared)


if __name__=='__main__': unittest.main()
