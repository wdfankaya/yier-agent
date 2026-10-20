"""Bounded per-manager embedding cache; semantic retrieval never rewrites facts."""
import math
import threading
from collections import OrderedDict
from openai import OpenAI
from app.config.settings import settings


class SemanticMemoryIndex:
    def __init__(self):
        self._vectors=OrderedDict()
        self._lock=threading.RLock()

    def scores(self,query,facts):
        if not query.strip() or not facts: return {}
        texts=[query]+[f.content for f in facts]
        model=settings.embedding_model
        endpoint=settings.embedding_base_url or settings.openai_base_url
        keys=[(endpoint,model,text) for text in texts]
        with self._lock:
            missing=list(dict.fromkeys(k for k in keys if k not in self._vectors))
            if missing:
                with OpenAI(api_key=settings.embedding_api_key or settings.openai_api_key,
                    base_url=endpoint,timeout=settings.memory_semantic_timeout,max_retries=0) as client:
                    response=client.embeddings.create(model=model,input=[k[2] for k in missing])
                rows=sorted(response.data,key=lambda row:row.index)
                if len(rows)!=len(missing): raise ValueError('Embedding response count mismatch')
                for key,row in zip(missing,rows):
                    vector=row.embedding
                    if not vector or not all(math.isfinite(v) for v in vector): raise ValueError('Invalid embedding')
                    self._vectors[key]=vector
            vectors=[self._vectors[k] for k in keys]
            for key in keys: self._vectors.move_to_end(key)
            while len(self._vectors)>256: self._vectors.popitem(last=False)
        def cosine(a,b):
            if len(a)!=len(b): raise ValueError('Embedding dimensions changed')
            denominator=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
            return sum(x*y for x,y in zip(a,b))/denominator if denominator else 0
        return {fact.fact_id:cosine(vectors[0],vector) for fact,vector in zip(facts,vectors[1:])}
