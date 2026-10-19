"""Chinese BM25 and rank-only reciprocal rank fusion; no score calibration claims."""
import math
import re
from collections import Counter
import jieba
from app.agent.rag.backends.base import RetrievedChunk

_TOKENIZER = jieba.Tokenizer()

def tokenize(text):
    # Keep product/order identifiers intact; segment Chinese with the same tokenizer
    # for documents and queries. No single-character stopword removal.
    result = []
    for part in re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*|[\u3400-\u9fff]+", text.lower()):
        result.extend(_TOKENIZER.cut(part, HMM=False) if '\u3400' <= part[0] <= '\u9fff' else [part])
    return result

class BM25Index:
    def __init__(self, chunks, k1=1.5, b=.75):
        self.chunks = list(chunks)
        self.counts = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(t for row in self.counts for t in row)
        self.k1, self.b = k1, b

    def search(self, query, top_k=20):
        terms = set(tokenize(query))
        scores = []
        n = len(self.chunks)
        for i, row in enumerate(self.counts):
            score = 0.0
            for term in terms:
                tf = row.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1-self.b+self.b*self.lengths[i]/self.avg))
            if score > 0:
                scores.append(RetrievedChunk(self.chunks[i], score))
        return sorted(scores, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k]

def rrf(lanes, top_k=20, k=60):
    scores, chunks = {}, {}
    for lane in lanes:
        seen = set()
        for rank, hit in enumerate(lane, 1):
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = hit.chunk
            scores[key] = scores.get(key, 0) + 1 / (k+rank)
    keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [RetrievedChunk(chunks[key], scores[key]) for key in keys]
