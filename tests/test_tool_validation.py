import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
from app.config.settings import settings
from app.agent.tools import decorator as definitions
from app.agent.tools.decorator import ToolInputError, validate_arguments, parse_arguments, tool
from app.agent.tools.registry import execute_tool, TOOL_DEFINITIONS, _TOOL_MAP
from app.agent.tools.manager import ToolManager
from app.agent.tools.hitl import ConfirmationGate, aresolve_on_agent
from app.agent.chat import YierAgent
from app.multi_agent.agents import SubAgent

class ValidationTests(unittest.TestCase):
    def test_schema_and_runtime_share_constraints(self):
        schema=next(d['function']['parameters'] for d in TOOL_DEFINITIONS if d['function']['name']=='search_knowledge')
        self.assertEqual(schema,definitions._TOOL_MODELS['search_knowledge'].model_json_schema())
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['top_k']['maximum'],5)
        self.assertEqual(schema['required'],['query'])
        self.assertEqual(validate_arguments('search_knowledge',{'query':' policy '})['top_k'],3)
    def test_invalid_values_do_not_execute(self):
        cases=[('query_order',{}),('query_order',{'order_id':42}),('query_order',{'order_id':None}),
               ('query_order',{'order_id':' '*3}),('query_order',{'order_id':'x'*65}),
               ('query_order',{'order_id':'a','extra':1}),('list_user_orders',{'extra':1}),
               ('query_product',{'keyword':''}),('apply_refund',{'order_id':'a','reason':'x'*501}),
               ('load_skill',{'skill_name':'../secret'})]
        cases += [('search_knowledge',{'query':'q','top_k':v}) for v in [True,False,'3',3.0,0,6,None,[],{}]]
        for name,args in cases:
            with self.subTest(name=name,args=args):
                business=Mock()
                with patch.dict(_TOOL_MAP,{name:business}):
                    result=json.loads(execute_tool(name,args))
                self.assertEqual(result['code'],'invalid_tool_arguments')
                self.assertTrue(result['issues']);business.assert_not_called()
                self.assertNotIn('input',result['issues'][0])
    def test_bad_json_and_nonobjects_return_correction(self):
        for value in ['{','[]','null','12','{"order_id":"a","order_id":"b"}','{"x":NaN}',[],None,12]:
            with self.subTest(value=value):
                result=json.loads(ToolManager().execute_tool('query_order',value))
                self.assertEqual(result['code'],'invalid_tool_arguments')
    def test_invalid_refund_does_not_create_token(self):
        gate=ConfirmationGate();tm=ToolManager(hitl=gate)
        with patch.object(settings,'hitl_enabled',True),patch('app.agent.tools.manager.local_execute_tool') as business:
            for args in [{}, {'order_id':'a','reason':2}, {'order_id':'a','reason':' '}, {'order_id':'a','reason':'r','extra':True}]:
                self.assertEqual(json.loads(tm.execute_tool('apply_refund',args))['code'],'invalid_tool_arguments')
            self.assertFalse(gate.pending);business.assert_not_called()
    def test_disallowed_tool_checked_before_confirmation(self):
        gate=ConfirmationGate();tm=ToolManager(hitl=gate,allowed_tools={'query_order'})
        result=json.loads(tm.execute_tool('apply_refund',{'order_id':'a','reason':'r'}))
        self.assertEqual(result['code'],'unknown_tool');self.assertFalse(gate.pending)
    def test_defaults_and_direct_calls(self):
        self.assertEqual(validate_arguments('recall_user_memory',{}),{'query':'','category':None,'top_k':5})
        self.assertEqual(validate_arguments('list_user_orders',{}),{})
        from app.agent.tools.refund import apply_refund
        with self.assertRaises(ToolInputError):apply_refund('a',123)
        self.assertTrue(json.loads(execute_tool('query_order',{'order_id':'ORD-20240120-002'}))['success'])
    def test_evaluation_trace_accepts_invalid_model_arguments(self):
        from app.evaluation.sandbox import Sandbox
        trace = SimpleNamespace(tool_observations=[])
        wrapped = Sandbox._wrap_execute_tool(None, ToolManager().execute_tool, trace)
        result = json.loads(wrapped('query_order', '{'))
        self.assertEqual(result['code'], 'invalid_tool_arguments')
        self.assertEqual(trace.tool_observations[0].arguments, {'invalid_arguments': True})

    def test_types_and_optional_annotations_generate_real_schema(self):
        before=list(TOOL_DEFINITIONS)
        try:
            @tool(desc='test')
            def validation_probe(items:list[int], enabled:bool, optional:int|None=None):
                return {'items':items,'enabled':enabled,'optional':optional}
            schema=definitions._TOOL_MODELS['validation_probe'].model_json_schema()
            self.assertEqual(schema['properties']['items']['type'],'array')
            self.assertIn({'type':'null'},schema['properties']['optional']['anyOf'])
            self.assertEqual(validation_probe([1,2],True)['optional'],None)
            with self.assertRaises(ToolInputError):validation_probe(['1'],True)
        finally:
            TOOL_DEFINITIONS[:]=before
            definitions._TOOL_MODELS.pop('validation_probe',None);_TOOL_MAP.pop('validation_probe',None)

class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalized_refund_arguments_match_confirmed_execution(self):
        gate=ConfirmationGate();tm=ToolManager(hitl=gate)
        agent=SimpleNamespace(hitl=gate,tool_manager=tm,raw_messages=[])
        with patch.object(settings,'hitl_enabled',True),patch('app.agent.tools.manager.local_execute_tool',return_value='{"success":true}') as business:
            pending=json.loads(await tm.aexecute_tool('apply_refund',{'order_id':' ORD-20240120-002 ','reason':' test '}))
            self.assertEqual(pending['args']['reason'],'test')
            result=await aresolve_on_agent(agent,pending['confirm_token'],True)
            self.assertEqual(result['execution_status'],'succeeded')
            business.assert_called_once_with('apply_refund',{'order_id':'ORD-20240120-002','reason':'test'})
    async def test_single_and_multi_agent_can_correct_malformed_json(self):
        for multi in (False,True):
            count=0
            async def llm(*args,**kwargs):
                nonlocal count
                count+=1
                if count==1:
                    call=SimpleNamespace(id='bad',function=SimpleNamespace(name='query_order',arguments='{'))
                    message=SimpleNamespace(content=None,tool_calls=[call],reasoning_content='opaque replay fixture')
                elif count==2:
                    feedback=json.loads(kwargs['messages'][-1]['content'])
                    self.assertEqual(feedback['code'],'invalid_tool_arguments')
                    call=SimpleNamespace(id='good',function=SimpleNamespace(name='query_order',arguments='{"order_id":"ORD-20240120-002"}'))
                    message=SimpleNamespace(content=None,tool_calls=[call],reasoning_content='opaque replay fixture')
                else:
                    self.assertTrue(json.loads(kwargs['messages'][-1]['content'])['success'])
                    message=SimpleNamespace(content='corrected',tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])
            tm=ToolManager()
            if multi:
                agent=SubAgent('test','test',tm,None,'fake',0)
                agent.on_event=lambda e:None
                with patch('app.multi_agent.agents.llm_create',llm):
                    reply,history=await agent.ahandle([])
            else:
                agent=object.__new__(YierAgent);agent.raw_messages=[];agent.max_react_steps=5
                agent.model='fake';agent.client=None;agent.temperature=0;agent.tool_manager=tm;agent.on_event=lambda e:None
                agent._build_messages=lambda:agent.raw_messages
                with patch('app.agent.chat.llm_create',llm):reply=await agent._react_loop()
                history=agent.raw_messages
            self.assertEqual(reply,'corrected');self.assertEqual(count,3)
            self.assertEqual(len([m for m in history if m['role']=='tool']),2)
            from app.agent.context import prepare_call
            wire=prepare_call({'messages':history})['messages']
            self.assertTrue(all(m.get('reasoning_content')=='opaque replay fixture' for m in wire if m.get('tool_calls')))

if __name__=='__main__':unittest.main()
