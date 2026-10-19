"""Two real model turns around a reconstructed Agent; refund tool is the repository's mock."""
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config.settings import settings
from app.agent.workflow import GraphAgent
from app.agent.maintenance import close_resources
from app.agent.tools.knowledge import reset_retriever
from app.observability import request_id,new_request_id
from app.trace_store import user_trace,read

async def run(root):
    settings.workflow_enabled=True
    settings.workflow_backend='sqlite'
    settings.workflow_sqlite_path=str(root/'graph.db')
    settings.workflow_ledger_path=str(root/'ledger.db')
    settings.trace_db_path=str(root/'trace.db')
    settings.db_enabled=False
    settings.redis_enabled=False
    settings.memory_enabled=False
    settings.mcp_enabled=False
    settings.rag_retrieval_mode='hybrid'
    settings.rag_backend='numpy'
    settings.kb_index_path='app/sessions/eval-index.json'
    reset_retriever()
    path=str(root/'chat.json')
    report={'kind':'real_model_graph_smoke','model':settings.model_name,'business_tool':'static_mock_no_money_movement','phases':[]}
    agent=None
    try:
        for phase in ['prepare','resume']:
            agent=GraphAgent(session_path=path,user_id='workflow-smoke')
            agent.on_event=lambda ev:None
            rid=new_request_id()
            context=request_id.set(rid)
            start=time.perf_counter()
            with user_trace('workflow-smoke'):
                if phase=='prepare':
                    reply=await agent.achat('请查询订单 ORD-20240120-002 并为它发起退款，原因是不想要了。提交前等我点击确认。')
                    status=await agent.workflow_status()
                    report['phases'].append({'phase':phase,'pending':bool(status['pending']),'reply':reply.reply})
                    if not status['pending']: raise RuntimeError('model_did_not_request_confirmation')
                else:
                    status=await agent.workflow_status()
                    if not status['pending']: raise RuntimeError('pending_not_restored')
                    result=await agent.aresolve_hitl(status['pending']['confirm_token'],True)
                    report['phases'].append({'phase':phase,'executed':result.get('executed'),
                        'execution_status':result.get('execution_status'),'reply':result.get('reply')})
                    if not result.get('executed'): raise RuntimeError('mock_refund_not_succeeded')
                report['phases'][-1]['elapsed_ms']=round((time.perf_counter()-start)*1000,2)
                report['phases'][-1]['trace']=read(rid,'workflow-smoke')
            request_id.reset(context)
            await close_resources(agent)
            agent=None
        report['passed']=True
    except Exception as exc:
        report.update(passed=False,error=type(exc).__name__)
    finally:
        if agent is not None: await close_resources(agent)
    Path('reports/workflow-live.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='phases'},ensure_ascii=False))
    return report['passed']

if __name__=='__main__':
    Path('app/sessions').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='app/sessions') as tmp:
        success=asyncio.run(run(Path(tmp)))
    raise SystemExit(0 if success else 1)
