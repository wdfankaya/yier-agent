"""Fixed-corpus ablation: BM25 offline, or real vector/RRF/BGE APIs with --live.

Query embeddings are batched/shared, reported separately from ranking latency.
No-answer cases are diagnostics, never counted as retrieval successes.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agent.rag.chunker import Chunk,chunk_markdown_dir
from app.agent.rag.lexical import BM25Index,rrf
from app.agent.rag.backends.numpy_backend import NumpyBackend
from app.agent.rag.embedder import Embedder
from app.agent.rag.reranker import HttpReranker
from app.config.settings import settings


def percentile(values,p):
    return sorted(values)[max(0,math.ceil(len(values)*p)-1)] if values else None


def summarize(rows):
    positives=[r for r in rows if r['answerable']]
    return {'answerable_cases':len(positives),'no_answer_cases':len(rows)-len(positives),
        'hit_at_1':sum(r['hit_at_1'] for r in positives)/len(positives) if positives else None,
        'hit_at_3':sum(r['hit_at_3'] for r in positives)/len(positives) if positives else None,
        'recall_at_3':sum(r['recall_at_3'] for r in positives)/len(positives) if positives else None,
        'mrr_at_3':sum(r['mrr_at_3'] for r in positives)/len(positives) if positives else None,
        'ranking_p50_ms':percentile([r['ranking_ms'] for r in rows],.5),
        'ranking_p95_ms':percentile([r['ranking_ms'] for r in rows],.95),
        'rerank_fallbacks':sum(bool(r.get('rerank_error')) for r in rows),
        'abstention_tested':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--split',choices=['dev','test','all'],default='all')
    parser.add_argument('--output',default='reports/retrieval-offline.json')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    raw=Path('app/evaluation/retrieval_cases.json').read_bytes()
    data=json.loads(raw)
    current=chunk_markdown_dir(Path(settings.kb_dir))
    expected=[{k:v for k,v in c.items() if k!='version'} for c in data['corpus']]
    if [c.to_dict() for c in current]!=expected:
        raise ValueError('Knowledge corpus differs from frozen gold data. Review labels before rebuilding.')
    cases=[r for r in data['cases'] if args.split=='all' or r['split']==args.split]
    lexical=BM25Index(current)
    lexical.search('预热')
    modes=['bm25']
    embedding_ms=0
    cached=False
    if args.live:
        modes=['vector','hybrid','hybrid_rerank']
        embedder=Embedder.from_settings(settings)
        fingerprint=hashlib.sha256((data['corpus_sha256']+hashlib.sha256(raw).hexdigest()+embedder.model).encode()).hexdigest()
        cache=Path('app/sessions/eval-'+fingerprint+'.json')
        if cache.exists():
            vectors=json.loads(cache.read_text(encoding='utf-8'))
            cached=True
        else:
            start=time.perf_counter()
            allvectors=embedder.encode([c.text for c in current]+[r['query'] for r in data['cases']])
            embedding_ms=(time.perf_counter()-start)*1000
            vectors={'documents':allvectors[:len(current)],'queries':dict(zip([r['id'] for r in data['cases']],allvectors[len(current):]))}
            cache.parent.mkdir(parents=True,exist_ok=True)
            cache.write_text(json.dumps(vectors),encoding='utf-8')
        backend=NumpyBackend(Path('app/sessions/eval-index.json'))
        backend.upsert(current,vectors['documents'],embedder.model)
        reranker=HttpReranker.from_settings()
        if reranker is None:
            raise ValueError('Reranker is not configured')

    def run(row,mode):
        start=time.perf_counter()
        error=None
        if mode=='bm25':
            hits=lexical.search(row['query'],3)
        else:
            hits=backend.search(vectors['queries'][row['id']],settings.rag_recall_k)
            if mode!='vector':
                hits=rrf([hits,lexical.search(row['query'],settings.rag_recall_k)],settings.rag_recall_k,settings.rag_rrf_k)
            if mode=='hybrid_rerank':
                try:
                    ranked=reranker.rerank(row['query'],[h.chunk.text for h in hits],3)
                    if not ranked: raise ValueError('empty rerank')
                    hits=[hits[i] for i,_ in ranked]
                except Exception as exc:
                    error=type(exc).__name__
            hits=hits[:3]
        ids=[h.chunk.chunk_id for h in hits]
        gold=set(row['gold_chunk_ids'])
        ranks=[i+1 for i,k in enumerate(ids) if k in gold]
        return {**row,'retrieved':ids,'ranking_ms':round((time.perf_counter()-start)*1000,3),
            'hit_at_1':bool(ranks and ranks[0]==1),'hit_at_3':bool(ranks),
            'recall_at_3':len(set(ids)&gold)/len(gold) if gold else 0,
            'mrr_at_3':1/ranks[0] if ranks else 0,'rerank_error':error}

    report={'created':datetime.now(timezone.utc).isoformat(),'kind':'real_api_ablation' if args.live else 'offline_lexical_only',
        'python':platform.python_version(),'platform':platform.platform(),'dataset_sha256':hashlib.sha256(raw).hexdigest(),
        'corpus_sha256':data['corpus_sha256'],'chunks':len(current),'human_reviewed':False,
        'embedding_model':settings.embedding_model if args.live else None,'rerank_model':settings.rerank_model if args.live else None,
        'query_embedding_batch_ms':embedding_ms,'embedding_cache_used':cached,'workers':args.workers,
        'latency_note':'Ranking only, excludes separately measured/shared batch embeddings. Not end-to-end chat latency.',
        'rrf_k':settings.rag_rrf_k,'recall_k':settings.rag_recall_k,'modes':{}}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for mode in modes:
        with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
            rows=list(pool.map(lambda row:run(row,mode),cases))
        report['modes'][mode]={'summary':summarize(rows),
            'splits':{split:summarize([r for r in rows if r['split']==split]) for split in ['dev','test']},'cases':rows}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(mode,json.dumps(report['modes'][mode]['summary']),flush=True)
    print(output)

if __name__=='__main__':
    main()
