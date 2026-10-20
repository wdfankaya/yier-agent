"""Small synthetic contrast set: lexical recall versus real embedding recall."""
import asyncio
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agent.memory.facts import extract,select
from app.agent.rag.embedder import Embedder
from app.config.settings import settings
from app.agent.resilience import llm_create
from openai import AsyncOpenAI

CASES=[('深色','我喜欢黑色'),('预算有限','我喜欢便宜一点的'),
       ('衣服不要紧绷','我喜欢宽松的衣服'),('黑色','我喜欢黑色'),
       ('便宜一点','我喜欢便宜一点的'),('鞋码','我的鞋码是42'),('发票抬头',None),('物流单号',None)]
TEXTS=['我喜欢黑色','我喜欢便宜一点的','我喜欢宽松的衣服','我的鞋码是42']

async def main():
    facts=extract([{'role':'user','content':t} for t in TEXTS],'recall-eval')
    embedder=Embedder.from_settings(settings)
    try: vectors=await asyncio.to_thread(embedder.encode,TEXTS+[q for q,_ in CASES])
    finally: embedder._client.close()
    def cosine(a,b): return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    rows=[]
    for i,(query,gold) in enumerate(CASES):
        lexical=select(facts,query,top_k=1)
        scores=sorted([(cosine(vectors[len(TEXTS)+i],v),t) for t,v in zip(TEXTS,vectors)],reverse=True)
        rows.append({'query':query,'expected_source':gold,'lexical_source':lexical[0].source_quote if lexical else None,
                     'lexical_hit':(not lexical) if gold is None else bool(lexical and gold in lexical[0].source_quote),
                     'vector_source':scores[0][1] if scores[0][0]>=settings.memory_semantic_min_score else None,'vector_hit':(scores[0][1] if scores[0][0]>=settings.memory_semantic_min_score else None)==gold,'cosine':round(scores[0][0],4)})
    import tempfile
    from app.agent.memory.long_term import LongTermMemory
    settings.db_enabled=False
    settings.memory_semantic_enabled=True
    with tempfile.TemporaryDirectory(dir='app/sessions') as temp:
        memory=LongTermMemory('semantic-eval',temp,source_session='recall-eval')
        memory.load(); memory.add_facts(facts); memory.save()
        for row in rows:
            recalled=await asyncio.to_thread(memory.recall,row['query'],None,1)
            row['runtime_source']=recalled[0].source_quote if recalled else None
            row['runtime_hit']=(not recalled) if row['expected_source'] is None else bool(recalled and row['expected_source'] in recalled[0].source_quote)
    answers=[]
    async with AsyncOpenAI(api_key=settings.openai_api_key,base_url=settings.openai_base_url,max_retries=0) as client:
        for label,evidence in [('lexical',[]),('vector',[TEXTS[0]])]:
            response=await llm_create(client,model=settings.model_name,max_tokens=256,temperature=0,
                messages=[{'role':'system','content':'根据提供的历史证据回答。没有证据只能说不知道，不得猜测。只回答用户明确喜欢的具体颜色。'},
                          {'role':'user','content':json.dumps({'question':'想买深色的，之前我明确说过喜欢哪种颜色？','historical_evidence':evidence},ensure_ascii=False)}])
            content=response.choices[0].message.content or ''
            answers.append({'retrieval':label,'mentions_black':'黑色' in content,'answers_unknown':'不知道' in content})
    report={'kind':'synthetic_small_contrast_set_not_production_benchmark','created':datetime.now(timezone.utc).isoformat(),
            'embedding_model':settings.embedding_model,'chat_model':settings.model_name,'cases':rows,'answer_ablation':answers,
            'lexical_hits':sum(r['lexical_hit'] for r in rows),'vector_hits':sum(r['vector_hit'] for r in rows),'runtime_hits':sum(r['runtime_hit'] for r in rows),'total':len(rows)}
    Path('reports/memory-recall.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': asyncio.run(main())
