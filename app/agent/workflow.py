"""Checkpointed ReAct graph with durable confirmation and conservative effect replay.

Checkpoints can use PostgreSQL or SQLite. The effect ledger is local SQLite:
this service remains single-host/single-worker, not a distributed transaction.
"""
import asyncio
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from app.config.settings import settings
from app.observability import event, stage, request_id, failure


class State(TypedDict, total=False):
    messages: list[dict]
    calls: list[dict]
    index: int
    steps: int
    pending: dict | None
    final: str
    last_result: dict
    approval_result: dict
    denied: list[str]


class EffectLedger:
    """Claim before executing; crash/timeout is UNKNOWN, never a blind retry."""
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, key):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,result FROM effects WHERE key=?", (key,)).fetchone()
            if row:
                return json.loads(row[1]) if row[1] else {"status":"execution_unknown", "message":"此前执行结果未知，需核实，未重复执行"}
            db.execute("INSERT INTO effects VALUES (?, 'running', NULL)", (key,))
        return None

    def finish(self, key, result):
        status = "unknown" if result.get("status") == "execution_unknown" else "done"
        with self.connect() as db:
            db.execute("UPDATE effects SET status=?,result=? WHERE key=?", (status,json.dumps(result,ensure_ascii=False),key))


def thread_key(user, session):
    return hashlib.sha256(json.dumps([user,str(session)],ensure_ascii=False).encode()).hexdigest()


@asynccontextmanager
async def checkpoint_store():
    if settings.workflow_backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        uri = settings.workflow_postgres_url or settings.database_url.replace('postgresql+asyncpg:', 'postgresql:')
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            await saver.setup()
            yield saver
    elif settings.workflow_backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        Path(settings.workflow_sqlite_path).parent.mkdir(parents=True,exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(settings.workflow_sqlite_path) as saver:
            yield saver
    else:
        raise ValueError("workflow_backend must be sqlite or postgres")


def compile_workflow(model, execute, sensitive, saver, ledger, thread_id, max_steps=5, ttl=300, business_scope=None,validate=None):
    """Model returns an OpenAI-style assistant message; callbacks are injectable."""
    async def reason(state):
        if state.get('steps',0) >= max_steps:
            text='本轮已达到处理步骤上限，请补充信息或联系人工客服。'
            return {'final':text,'calls':[], 'messages':state['messages']+[{'role':'assistant','content':text}]}
        with stage('workflow.model'):
            msg = await model(state['messages'])
        calls=msg.get('tool_calls') or []
        event('workflow_route', route='tools' if calls else 'answer', tool_count=len(calls))
        return {'messages':state['messages']+[msg], 'calls':calls, 'index':0,
                'steps':state.get('steps',0)+1, 'final':msg.get('content') or ''}

    def prepare(state):
        call=state['calls'][state['index']]
        from app.agent.tools.decorator import parse_arguments
        try:
            args=validate(call['function']['name'],call['function']['arguments']) if validate else parse_arguments(call['function']['arguments'])
        except Exception:
            args=None
        name=call['function']['name']
        pending=None
        if sensitive(name) and args is not None:
            pending={'confirm_token':uuid4().hex, 'tool':name, 'args':args,
                     'order_id':args.get('order_id'), 'expires_at':time.time()+ttl}
        return {'pending':pending}

    async def tool(state):
        call=state['calls'][state['index']]
        name=call['function']['name']
        pending=state.get('pending')
        denied=list(state.get('denied',[]))
        result=None
        confirmed=False
        if pending:
            # Full-refund demo: same user/order is one business effect even if the
            # model changes the reason or the user opens a new chat.
            identity = pending['args'].get('order_id', pending['args'])
            key=hashlib.sha256(json.dumps([business_scope or thread_id,name,identity],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if str(pending.get('order_id')) in denied:
                result={'status':'denied'}
            else:
                decision=interrupt(pending)
                event('confirmation_resumed', wait_ms=round(max(0,time.time()-(pending['expires_at']-ttl))*1000,2))
                if not isinstance(decision,dict) or not secrets.compare_digest(str(decision.get('token','')),pending['confirm_token']):
                    result={'status':'denied','error':'确认令牌不匹配'}
                elif time.time()>pending['expires_at']:
                    result={'status':'expired','error':'确认已过期，请重新发起'}
                elif decision.get('approved') is not True:
                    denied.append(str(pending.get('order_id')))
                    result={'status':'denied'}
                else:
                    confirmed=True
                    result=await asyncio.to_thread(ledger.claim,key)
                    if result is None:
                        try:
                            raw=await execute(name,pending['args'])
                            result=json.loads(raw) if isinstance(raw,str) else raw
                            if not isinstance(result,dict):
                                result={'status':'execution_unknown'}
                        except Exception:
                            result={'status':'execution_unknown'}
                        await asyncio.to_thread(ledger.finish,key,result)
        else:
            raw=await execute(name,call['function']['arguments'])
            try:
                result=json.loads(raw) if isinstance(raw,str) else raw
            except (ValueError,TypeError):
                result={'status':'execution_unknown'}
        event('workflow_tool', tool=name, status=result.get('status','completed') if isinstance(result,dict) else 'completed')
        observation={**result,'workflow_confirmation':'approved' if confirmed else 'not_approved'} if pending and isinstance(result,dict) else result
        update={'messages':state['messages']+[{'role':'tool','tool_call_id':call['id'],
                    'content':json.dumps(observation,ensure_ascii=False)}],
                'index':state['index']+1,'last_result':result,'pending':None,'denied':denied}
        if pending:
            update['approval_result']=result
        return update

    graph=StateGraph(State)
    graph.add_node('reason',reason)
    graph.add_node('prepare',prepare)
    graph.add_node('tool',tool)
    graph.add_edge(START,'reason')
    graph.add_conditional_edges('reason',lambda s:'prepare' if s.get('calls') else END)
    graph.add_edge('prepare','tool')
    graph.add_conditional_edges('tool',lambda s:'prepare' if s['index']<len(s['calls']) else 'reason')
    return graph.compile(checkpointer=saver)


from app.agent.chat import YierAgent
from app.schemas.response import CustomerServiceResponse, IntentType


class GraphAgent(YierAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.thread_id=thread_key(self.user_id,self.session_path)
        self.ledger=EffectLedger(settings.workflow_ledger_path)
        # Confirmation is enforced by graph nodes, not the old in-memory gate.
        self.tool_manager.hitl=None

    async def _execute(self,name,args):
        self._emit_tool_call(name,args if isinstance(args,dict) else {})
        result=await self.tool_manager.aexecute_tool(name,args)
        if name=='apply_refund' and self.tool_manager._tool_source.get(name)=='local':
            data=json.loads(result)
            data['simulation']=True
            result=json.dumps(data,ensure_ascii=False)
        self._emit('tool_result',name=name,content=result)
        return result

    @property
    def graph_config(self):
        return {'configurable':{'thread_id':self.thread_id},'recursion_limit':max(32,self.max_react_steps*32)}

    async def _model(self,messages):
        from app.agent.resilience import llm_create
        self.raw_messages=list(messages)
        prompt=self._build_messages()
        prompt[0]['content'] += ('\n\n当前启用持久化人工审批工作流。用户明确申请退款且订单号、原因齐全时，'
            '先查询订单，再调用 apply_refund 来创建待确认操作；框架会在真正执行前暂停并显示确认按钮。'
            '不要仅用自然语言询问确认而不创建操作，也不要要求用户回复“确认”替代按钮。'
            '工具调用本身不代表已退款，只有工具返回 success=true 才能说明模拟退款成功。'
            'workflow_confirmation=approved 表示用户已经通过按钮或API明确授权，不要说未经过确认。'
            'simulation=true 表示演示工具没有改变真实订单或转移资金，必须明确说“模拟退款”，不得声称真实款项已进入退款流程。'
            '遇到 execution_unknown 必须说明结果未知、需要人工核实，不能重复执行或声称成功。')
        result=await llm_create(self.client,model=self.model,messages=prompt,
                               temperature=self.temperature,tools=self.tool_manager.tool_definitions)
        msg=result.choices[0].message
        output={'role':'assistant','content':msg.content}
        if msg.tool_calls:
            output['tool_calls']=[{'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in msg.tool_calls]
        return output

    @asynccontextmanager
    async def _graph(self):
        async with checkpoint_store() as saver:
            yield compile_workflow(self._model,self._execute,self.tool_manager.is_sensitive,
                                   saver,self.ledger,self.thread_id,self.max_react_steps,settings.hitl_token_ttl,self.user_id,
                                   self.tool_manager.validated_arguments)

    async def workflow_status(self):
        async with self._graph() as graph:
            snapshot=await graph.aget_state(self.graph_config)
            return {'pending':snapshot.values.get('pending') if snapshot.next else None,
                    'resumable':bool(snapshot.next),'has_state':bool(snapshot.values),
                    'last_result':snapshot.values.get('last_result')}

    async def _finish(self,state):
        self.raw_messages=list(state.get('messages',[]))
        pauses=state.get('__interrupt__')
        if pauses:
            pending=pauses[0].value
            self._emit('confirm_required',**pending)
            result=CustomerServiceResponse(intent=IntentType.RETURN_REQUEST,confidence=1,
                reply='操作已暂停，请确认或拒绝；确认状态已保存，可在重启后继续。')
        else:
            result=await self._extract_structured_response(state.get('final') or '处理结束。')
        # Graph owns message history; do not compress incomplete tool-call groups.
        await asyncio.to_thread(self.save)
        self._emit('final',**result.model_dump(mode='json'))
        if not pauses and self.memory_manager.memory_enabled:
            async def maintain():
                try:
                    await asyncio.wait_for(self.memory_manager.aupdate_short_term(self.raw_messages[-6:]),settings.memory_maintenance_timeout)
                    await asyncio.to_thread(self.save)
                    await self._maybe_consolidate()
                except Exception as exc:
                    failure('workflow_memory_failed',exc)
            if self._ephemeral_loop:
                await maintain()
            else:
                self._maintenance_task=asyncio.create_task(maintain())
        return result

    async def achat(self,user_input):
        from app.agent.maintenance import drain_maintenance
        await drain_maintenance(self)
        async with self._graph() as graph:
            snap=await graph.aget_state(self.graph_config)
            if snap.next:
                if snap.values.get('pending'):
                    self._emit('confirm_required',**snap.values['pending'])
                result=CustomerServiceResponse(intent=IntentType.OTHER,confidence=1,
                    reply='上一轮尚未结束，请先确认、拒绝或恢复该流程。')
                self._emit('final',**result.model_dump(mode='json'))
                return result
            messages=list(snap.values.get('messages',self.raw_messages))
            from app.agent.history import repair_tool_groups
            repair_tool_groups(messages)
            # Compact only complete turns, persist the summary before graph entry.
            self.raw_messages=messages
            if len(messages)>self.history_threshold:
                await self._compress_history()
                messages=list(self.raw_messages)
                await asyncio.to_thread(self.save)
            state=await graph.ainvoke({'messages':messages+[{'role':'user','content':user_input}],
                'steps':0,'pending':None,'calls':[],'final':'','approval_result':{},'denied':snap.values.get('denied',[])}, self.graph_config)
        return await self._finish(state)

    async def aresolve_hitl(self,token,approved):
        async with self._graph() as graph:
            snap=await graph.aget_state(self.graph_config)
            pending=snap.values.get('pending')
            if not snap.next or not pending or not secrets.compare_digest(token,pending['confirm_token']):
                return {'ok':False,'error':'确认不存在、已使用或不属于该会话'}
            state=await graph.ainvoke(Command(resume={'token':token,'approved':approved}),self.graph_config)
        result=state.get('approval_result',{})
        reply=await self._finish(state)
        status=result.get('status')
        executed=result.get('success') is True
        return {'ok':True,'approved':approved,'executed':executed,
                'execution_status':'succeeded' if executed else ('unknown' if status=='execution_unknown' else 'denied' if status=='denied' else 'failed'),
                'tool':pending['tool'],'order_id':pending.get('order_id'),'result':result,
                'reply':reply.reply,'pending':state.get('pending'),'request_id':request_id.get()}

    async def recover(self):
        async with self._graph() as graph:
            snap=await graph.aget_state(self.graph_config)
            interrupted=any(task.interrupts for task in snap.tasks)
            if not snap.next or interrupted:
                return {'recovered':False,'pending':snap.values.get('pending')}
            state=await graph.ainvoke(None,self.graph_config)
        result=await self._finish(state)
        return {'recovered':True,'reply':result.reply,'pending':state.get('pending')}

    def reset(self):
        from app.agent.async_utils import run_sync
        async def remove():
            async with checkpoint_store() as saver:
                await saver.adelete_thread(self.thread_id)
        run_sync(remove())
        # Keep the side-effect ledger: deleting chat cannot authorize a duplicate write.
        super().reset()
