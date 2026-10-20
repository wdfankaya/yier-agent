"""Small real-model context tests with synthetic data; never store credentials."""
import asyncio
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openai import AsyncOpenAI
from app.config.settings import settings
from app.agent.context import prepare_call, estimate_tokens, input_budget, ContextBudgetExceeded
from app.agent.resilience import llm_create


async def main():
    settings.llm_max_retries=0
    settings.trace_enabled=False
    report={'created':datetime.now(timezone.utc).isoformat(),'kind':'real_model_synthetic_context',
        'model':settings.model_name,'configured_window':settings.context_window_tokens,
        'output_reserve':settings.context_output_tokens,'safety_reserve':settings.context_safety_tokens,'cases':[]}
    async with AsyncOpenAI(api_key=settings.openai_api_key,base_url=settings.openai_base_url,max_retries=0) as api:
        calls=0
        async def create(**kwargs):
            nonlocal calls
            calls+=1
            return await api.chat.completions.create(**kwargs)
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        cases=[
            ('long_chinese',[{'role':'system','content':'阅读数据，只输出末尾核验码，不要解释。'},
                {'role':'user','content':'以下是无关的中文说明。'*650+'\n最终核验码：CN-9081'}],None,'CN-9081'),
            ('long_tool_tail',[{'role':'system','content':'根据工具结果，仅输出订单号和金额，不要调用其他工具。'},
                {'role':'user','content':'核验订单号和金额'},
                {'role':'assistant','content':None,'tool_calls':[{'id':'call_test','type':'function','function':{'name':'lookup_order','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'call_test','content':json.dumps({'padding':'无关资料'*15000,'order_id':'ORD-9081','amount':129.5,'status':'paid'},ensure_ascii=False)}],
                [{'type':'function','function':{'name':'lookup_order','description':'读取测试订单','parameters':{'type':'object','properties':{}}}}], 'ORD-9081'),
        ]
        for name,messages,tools,expected in cases:
            fixture_calls=0
            if name=='long_tool_tail':
                response=await llm_create(client,model=settings.model_name,max_tokens=1024,
                    messages=[{'role':'user','content':'请务必先调用 lookup_order 查询测试订单。'}],tools=tools)
                msg=response.choices[0].message
                if not msg.tool_calls: raise RuntimeError('fixture model did not call tool')
                from app.agent.history import retain_tool_context
                messages[2]=retain_tool_context({'role':'assistant','content':msg.content,
                    'tool_calls':[{'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in msg.tool_calls]},msg)
                messages[3]['tool_call_id']=msg.tool_calls[0].id
                fixture_calls=1
            kwargs={'model':settings.model_name,'messages':messages,'max_tokens':256,'temperature':0}
            if tools: kwargs.update(tools=tools,tool_choice='none')
            row={'name':name,'before_bytes':estimate_tokens(messages),'fixture_api_calls':fixture_calls,'requested_output_tokens':256}
            before=calls
            try:
                prepared=prepare_call(kwargs)
                row['after_bytes']=estimate_tokens(prepared['messages'])
                response=await llm_create(client,**kwargs)
                content=response.choices[0].message.content or ''
                usage=response.usage
                row.update(api_calls=calls-before,prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    passed=expected in content and (name!='long_tool_tail' or '129.5' in content))
                if usage: row['passed']=row['passed'] and usage.prompt_tokens<=input_budget(prepared['max_tokens'])
            except Exception as exc:
                row.update(passed=False,error_type=type(exc).__name__,api_calls=calls-before)
                if type(exc).__name__=='BadRequestError': row['provider_error']=str(getattr(exc,'body',{}))[:800]
            report['cases'].append(row)
        before=calls
        try:
            await llm_create(client,model=settings.model_name,messages=[{'role':'user','content':'超长输入'*settings.context_window_tokens}])
            rejected=False
        except ContextBudgetExceeded: rejected=True
        report['cases'].append({'name':'oversize_rejected_locally','passed':rejected and calls==before,'api_calls':calls-before})
    report['success']=all(case['passed'] for case in report['cases'])
    Path('reports/context-live.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return report['success']

if __name__=='__main__': raise SystemExit(0 if asyncio.run(main()) else 1)
