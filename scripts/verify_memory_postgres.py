"""Create an isolated real PostgreSQL container, test, report, and remove only that container."""
import json
import os
import secrets
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def docker(*args,**kwargs):
    return subprocess.run(['docker',*args],check=True,text=True,capture_output=True,timeout=90,**kwargs).stdout.strip()


def main():
    name='yier-memory-test-'+secrets.token_hex(5)
    password=secrets.token_urlsafe(24)
    environment={**os.environ,'POSTGRES_PASSWORD':password}
    started=False
    report={'kind':'real_postgresql_isolated_container','created':datetime.now(timezone.utc).isoformat()}
    try:
        docker('run','--rm','-d','--name',name,'-e','POSTGRES_PASSWORD','-e','POSTGRES_DB=memory_test',
               '-p','127.0.0.1::5432','--tmpfs','/var/lib/postgresql/data','postgres:16-alpine',env=environment)
        started=True
        port=docker('port',name,'5432/tcp').rsplit(':',1)[1]
        os.environ['MEMORY_TEST_DATABASE_URL']=f'postgresql+asyncpg://postgres:{password}@127.0.0.1:{port}/memory_test'
        import psycopg
        until=time.monotonic()+60
        while True:
            try:
                with psycopg.connect(host='127.0.0.1',port=port,dbname='memory_test',user='postgres',password=password,connect_timeout=2) as db:
                    report['server_version']=db.execute('SELECT version()').fetchone()[0]
                break
            except psycopg.OperationalError:
                if time.monotonic()>until: raise TimeoutError('PostgreSQL startup') from None
                time.sleep(.25)
        report['image_id']=docker('inspect','--format','{{.Image}}',name)
        suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_memory_postgres')
        before=time.perf_counter()
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(tests_run=result.testsRun,success=result.wasSuccessful() and not result.skipped,
            failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors],
            skipped=[str(t) for t,_ in result.skipped],elapsed_seconds=round(time.perf_counter()-before,3))
    except Exception as exc:
        report.update(success=False,error_type=type(exc).__name__)
    finally:
        if started:
            docker('stop','--time','5',name)
        Path('reports').mkdir(exist_ok=True)
        Path('reports/memory-postgres.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return report.get('success',False)

if __name__=='__main__': raise SystemExit(0 if main() else 1)
