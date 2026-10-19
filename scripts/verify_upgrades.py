"""Run only deterministic offline regressions and write a reproducible test manifest."""
import json
import platform
import sys
import time
import unittest
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

MODULES=['tests.test_workflow','tests.test_workflow_api','tests.test_hybrid_trace',
         'tests.test_security_regressions','tests.test_tool_validation','tests.test_optimizations']

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'created':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'kind':'offline_injected_model_real_sqlite_and_http','modules':MODULES,
        'tests_run':result.testsRun,'failures':[str(t) for t,_ in result.failures],
        'errors':[str(t) for t,_ in result.errors],'skipped':[str(t) for t,_ in result.skipped],
        'elapsed_seconds':round(time.perf_counter()-started,3),'success':result.wasSuccessful()}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/upgrade-tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
