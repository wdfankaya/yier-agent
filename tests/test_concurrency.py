"""10 并发用户不串会话 + 记忆/技能隔离。

1. 10 个 user 同时各打 3 轮 HTTP SSE（LLM 打桩），session_id / user_id 不串
2. 10 路并发 recall_user_memory：各自只能看到本 user 的 LTM；load_skill 走本 Agent 的 manager

用法：python tests/test_concurrency.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config.settings import settings  # noqa: E402
from app.server.ratelimit import reset_limiter  # noqa: E402

N = 10
TURNS = 3


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _fake_response(content: str, parsed=None):
    msg = SimpleNamespace(content=content, tool_calls=None, parsed=parsed)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-d6")


async def _fake_create(*args, **kwargs):
    await asyncio.sleep(0.05)
    messages = kwargs.get("messages") or []
    blob = " ".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in messages
    )
    if "提取结构化信息" in blob:
        return _fake_response(
            '{"intent":"greeting","confidence":0.9,'
            '"reply":"你好，我是一二","requires_human":false,'
            '"follow_up_question":null}'
        )
    return _fake_response("你好，我是一二")


async def _fake_parse(*args, **kwargs):
    from app.schemas.response import CustomerServiceResponse, IntentType
    parsed = CustomerServiceResponse(
        intent=IntentType.GREETING,
        confidence=0.9,
        reply="你好，我是一二",
        requires_human=False,
    )
    return _fake_response("你好，我是一二", parsed=parsed)


def _patch_registry():
    from app.server.sessions import SessionRegistry
    orig = SessionRegistry._build_agent

    def wrapped(self, session_id: str, user_id: str):
        agent = orig(self, session_id, user_id)
        agent.client.chat.completions.create = _fake_create
        agent.client.beta.chat.completions.parse = _fake_parse
        return agent

    SessionRegistry._build_agent = wrapped  # type: ignore[method-assign]
    return orig


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        line = next((ln for ln in block.split("\n") if ln.startswith("data: ")), None)
        if not line:
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            continue
    return events


async def test_http_10():
    print("\n[1/2] 10 用户 × 3 轮 HTTP SSE，会话不串")
    old = (
        settings.db_enabled,
        settings.redis_enabled,
        settings.memory_enabled,
        settings.mcp_enabled,
        settings.hitl_enabled,
        settings.rate_limit_enabled,
        settings.ltm_consolidate_every,
        settings.multi_agent_enabled,
    )
    settings.db_enabled = False
    settings.redis_enabled = False
    settings.memory_enabled = False
    settings.mcp_enabled = False
    settings.hitl_enabled = False
    settings.rate_limit_enabled = False
    settings.ltm_consolidate_every = 0
    settings.multi_agent_enabled = False
    reset_limiter()
    orig = _patch_registry()
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            "app.server.main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        async with httpx.AsyncClient(timeout=30.0) as probe:
            while time.time() < deadline:
                try:
                    r = await probe.get(f"http://127.0.0.1:{port}/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)
            else:
                _fail("服务未在 15s 内起来")

        async def one_user(i: int) -> None:
            sid = f"c10-{i}"
            uid = f"u10-{i}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                for t in range(TURNS):
                    resp = await client.post(
                        f"http://127.0.0.1:{port}/api/chat",
                        json={
                            "session_id": sid,
                            "user_id": uid,
                            "message": f"第{t}轮-用户{i}",
                        },
                    )
                    if resp.status_code != 200:
                        _fail(f"user={i} turn={t} HTTP {resp.status_code} {resp.text[:200]}")
                    events = _parse_sse(resp.text)
                    sess = next((e for e in events if e.get("type") == "session"), None)
                    if not sess:
                        _fail(f"user={i} 缺少 session 事件: {events[:3]}")
                    if sess.get("session_id") != sid or sess.get("user_id") != uid:
                        _fail(f"串会话: 期望 {sid}/{uid} 实际 {sess}")
                    finals = [e for e in events if e.get("type") == "final"]
                    if not finals:
                        _fail(f"user={i} 没有 final: {[e.get('type') for e in events]}")
                hist = await client.get(f"http://127.0.0.1:{port}/api/sessions/{sid}")
                if hist.status_code != 200:
                    _fail(f"读历史失败 {hist.status_code}")
                body = hist.json()
                if body.get("user_id") != uid:
                    _fail(f"历史 user_id 串了: {body.get('user_id')} != {uid}")
                texts = " ".join(
                    (m.get("content") or "") for m in body.get("messages") or []
                    if m.get("role") == "user"
                )
                if f"用户{i}" not in texts:
                    _fail(f"session {sid} 没自己的用户消息")
                for j in range(N):
                    if j != i and f"用户{j}" in texts:
                        _fail(f"session {sid} 混入了用户{j} 的消息")

        t0 = time.perf_counter()
        await asyncio.gather(*[one_user(i) for i in range(N)])
        elapsed = time.perf_counter() - t0
        if elapsed > 25:
            _fail(f"10×3 并发过慢 ({elapsed:.1f}s)，可能串行卡住")
        _ok(f"10 用户各 3 轮完成 {elapsed:.2f}s，session/user 均未串")
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        from app.server.sessions import SessionRegistry
        SessionRegistry._build_agent = orig  # type: ignore[method-assign]
        (
            settings.db_enabled,
            settings.redis_enabled,
            settings.memory_enabled,
            settings.mcp_enabled,
            settings.hitl_enabled,
            settings.rate_limit_enabled,
            settings.ltm_consolidate_every,
            settings.multi_agent_enabled,
        ) = old
        reset_limiter()


async def test_pc_isolation():
    print("\n[2/2] 10 路并发 recall_user_memory / load_skill 不串")
    from app.agent.memory.long_term import MemoryFact
    from app.agent.memory.manager import MemoryManager
    from app.agent.skills import SkillManager
    from app.agent.tools.manager import ToolManager

    old_db = settings.db_enabled
    settings.db_enabled = False
    tms = []
    try:
        for i in range(N):
            mm = MemoryManager(
                client=None,
                model="x",
                user_id=f"pc-{i}",
                memory_dir=str(ROOT / "app" / "sessions" / "memory-pc-test"),
                memory_enabled=True,
            )
            mm.ltm.facts = []
            mm.ltm.add_facts([
                MemoryFact(
                    content=f"用户{i}的专属偏好是颜色{i}",
                    category="preference",
                    created_at=datetime.now().isoformat(timespec="seconds"),
                )
            ])
            sm = SkillManager(skills_dir=settings.skills_dir, enabled=True)
            tm = ToolManager(
                use_mcp=False,
                memory_manager=mm,
                skill_manager=sm,
            )
            tms.append(tm)

        async def call(i: int) -> None:
            mem_raw = await asyncio.to_thread(
                tms[i].execute_tool, "recall_user_memory", {"query": ""},
            )
            mem = json.loads(mem_raw)
            if not mem.get("success"):
                _fail(f"user {i} recall 失败: {mem}")
            blob = json.dumps(mem, ensure_ascii=False)
            if f"用户{i}的专属偏好是颜色{i}" not in blob:
                _fail(f"user {i} 没拿到自己的记忆: {blob[:200]}")
            for j in range(N):
                if j != i and f"用户{j}的专属偏好" in blob:
                    _fail(f"user {i} 串到了 user {j} 的记忆")
            skill_raw = await asyncio.to_thread(
                tms[i].execute_tool, "load_skill", {"skill_name": "track-order"},
            )
            skill = json.loads(skill_raw)
            if not skill.get("success") or skill.get("skill_name") != "track-order":
                _fail(f"user {i} load_skill 失败: {skill}")
            if "物流" not in (skill.get("instructions") or "") and "订单" not in (skill.get("instructions") or ""):
                _fail(f"user {i} 技能正文不对")

        await asyncio.gather(*[call(i) for i in range(N)])
        _ok("10 路并发各自读到本 user 记忆；load_skill 均命中 track-order")
    finally:
        settings.db_enabled = old_db


def main():
    print("=" * 60)
    print("10 用户并发与上下文隔离")
    print("=" * 60)
    asyncio.run(test_pc_isolation())
    asyncio.run(test_http_10())
    print("\n全部通过。")


if __name__ == "__main__":
    main()
