"""Single-worker launcher; Windows psycopg needs a selector event loop."""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('WORKFLOW_ENABLED','true')
os.environ.setdefault('RAG_RETRIEVAL_MODE','hybrid')
import uvicorn
from app.config.settings import settings

if __name__=='__main__':
    server=uvicorn.Server(uvicorn.Config('app.server.main:app',host='127.0.0.1',port=8000,loop='none'))
    factory=asyncio.SelectorEventLoop if sys.platform=='win32' and settings.workflow_backend=='postgres' else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(server.serve())
