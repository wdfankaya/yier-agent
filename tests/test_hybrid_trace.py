import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.config.settings import settings
from app.agent.rag.chunker import Chunk,chunk_markdown_dir,_split_long,MAX_CHUNK_CHARS
from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.lexical import BM25Index, rrf, tokenize
from app.observability import request_id, stage, model_usage, event
from app.trace_store import user_trace, read, prune


class HybridTests(unittest.TestCase):
    def test_chinese_and_identifier(self):
        self.assertIn('ord-20240115-001',tokenize('订单 ORD-20240115-001 退款'))
        chunks=[Chunk('a','退货','质量','质量问题签收十五天运费免费'),Chunk('b','会员','等级','会员消费满五千元升级黄金')]
        self.assertEqual(BM25Index(chunks).search('质量问题退货运费')[0].chunk.chunk_id,'a')
        self.assertEqual(BM25Index(chunks).search('xyzunknown'),[])

    def test_rrf_deduplicates_and_ignores_raw_score(self):
        a=RetrievedChunk(Chunk('a','d','s','a'),.001)
        b=RetrievedChunk(Chunk('b','d','s','b'),99999)
        result=rrf([[a,b,a],[a,b]])
        self.assertEqual([h.chunk.chunk_id for h in result],['a','b'])
        self.assertAlmostEqual(result[0].score,2/61)

    def test_empty(self):
        self.assertEqual(BM25Index([]).search('退款'),[])
        self.assertEqual(rrf([[],[]]),[])

    def test_content_versions_archives_and_long_paragraph(self):
        with tempfile.TemporaryDirectory(dir='app/sessions') as tmp:
            root=Path(tmp)
            (root/'archive').mkdir()
            (root/'archive'/'policy-v1.md').write_text('# 旧规则\n## 运费\n旧规则10元',encoding='utf-8')
            active=root/'policy.md'
            active.write_text('# 现行规则\n## 运费\n现行规则8元',encoding='utf-8')
            first=chunk_markdown_dir(root)
            self.assertEqual(len(first),1)
            self.assertNotIn('10元',first[0].text)
            active.write_text('# 现行规则\n## 运费\n新版规则6元',encoding='utf-8')
            second=chunk_markdown_dir(root)
            self.assertEqual(first[0].chunk_id,second[0].chunk_id)
            self.assertNotEqual(first[0].version,second[0].version)
        pieces=_split_long('长'*5000)
        self.assertTrue(all(len(p)<=MAX_CHUNK_CHARS for p in pieces))
        self.assertEqual(''.join(pieces),'长'*5000)

    def test_functional_and_budget_are_separate(self):
        from app.evaluation.evaluator import Evaluator,EvalResult
        evaluator=Evaluator(None,None,'fake',use_judge=False)
        result=EvalResult('x','over budget',tool_accuracy=1,intent_match=1,token_pass=False)
        self.assertFalse(evaluator._decide_pass(result))
        self.assertTrue(evaluator._decide_pass(result,include_budget=False))

    def test_trace_ownership_spans_cost_and_pruning(self):
        Path('app/sessions').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir='app/sessions') as tmp,patch.object(settings,'trace_db_path',str(Path(tmp)/'traces.db')),patch.object(settings,'trace_enabled',True),patch.object(settings,'llm_input_cost_per_million',1),patch.object(settings,'llm_output_cost_per_million',2):
            token=request_id.set('r1')
            try:
                with user_trace('alice'),stage('parent'):
                    with stage('child'):
                        model_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=50)))
                    event('first_response',elapsed_ms=10)
                with user_trace('bob'):
                    event('foreign')
                self.assertIsNone(read('r1','bob'))
                trace=read('r1','alice')
                self.assertAlmostEqual(trace['metrics']['estimated_cost'],.0002)
                self.assertEqual(trace['metrics']['input_tokens'],100)
                stages={e['stage']:e for e in trace['events'] if e['event']=='stage'}
                self.assertEqual(stages['child']['parent_span_id'],stages['parent']['span_id'])
                self.assertNotIn('foreign',json.dumps(trace))
                prune(days=-1)
                self.assertIsNone(read('r1','alice'))
            finally:
                request_id.reset(token)


if __name__=='__main__':
    unittest.main()
