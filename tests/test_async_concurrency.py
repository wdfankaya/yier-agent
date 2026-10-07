"""异步并发 + 沙箱 async 插桩。

1. 5 个会话并发 achat：LLM 用 sleep 模拟 0.3s I/O，总时长应明显小于 5×串行
2. 给 AsyncOpenAI.create 打补丁后仍能记 latency/usage
3. 同步 chat() 门面（asyncio.run）在无 loop 时仍可用

用法：python tests/test_async_concurrency.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config.settings import settings  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402
from app.evaluation.trace import RunTrace  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402

SLEEP = 0.30
N = 5


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _fake_response(content: str, parsed=None):
    msg = SimpleNamespace(content=content, tool_calls=None, parsed=parsed)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-async")


async def _fake_create(*args, **kwargs):
    await asyncio.sleep(SLEEP)
    messages = kwargs.get("messages") or []
    blob = " ".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in messages
    )
    if "提取结构化信息" in blob:
        raw = (
            '{"intent":"greeting","confidence":0.9,'
            '"reply":"你好，我是一二","requires_human":false,'
            '"follow_up_question":null}'
        )
        return _fake_response(raw)
    return _fake_response("你好，我是一二")


async def _fake_parse(*args, **kwargs):
    parsed = CustomerServiceResponse(
        intent=IntentType.GREETING,
        confidence=0.9,
        reply="你好，我是一二",
        requires_human=False,
    )
    return _fake_response("你好，我是一二", parsed=parsed)


def _make_agent(i: int):
    from app.agent.chat import YierAgent

    settings.memory_enabled = False
    settings.mcp_enabled = False
    settings.ltm_consolidate_every = 0
    settings.db_enabled = False

    path = str(ROOT / "app" / "sessions" / f"test_async_{i}.json")
    Path(path).unlink(missing_ok=True)
    agent = YierAgent(session_path=path, user_id=f"async-{i}")
    agent.history_threshold = 100
    agent.client.chat.completions.create = _fake_create
    agent.client.beta.chat.completions.parse = _fake_parse
    return agent, path


async def test_five_concurrent():
    print("\n[1/3] 5 会话并发 achat（模拟 LLM 0.3s I/O）")
    agents = []
    paths = []
    try:
        for i in range(N):
            a, p = _make_agent(i)
            agents.append(a)
            paths.append(p)

        t0 = time.perf_counter()
        await asyncio.gather(*[a.achat("你好") for a in agents])
        parallel = time.perf_counter() - t0

        t0 = time.perf_counter()
        await agents[0].achat("再问一句")
        single = time.perf_counter() - t0

        # 5 路并行应接近 1×单次，绝不能接近 5×
        if parallel >= single * 4:
            _fail(
                f"并发未重叠: parallel={parallel:.2f}s single={single:.2f}s "
                f"(5×single={single*5:.2f}s)"
            )
        _ok(
            f"5 并发 {parallel:.2f}s，单次 {single:.2f}s "
            f"（5×串行约 {single*5:.2f}s）"
        )
        for a in agents:
            if not a.raw_messages:
                _fail("会话消息为空")
        _ok("5 个会话都有独立历史，未串台")
    finally:
        for a in agents:
            try:
                await a.aclose()
            except Exception:
                pass
        for p in paths:
            Path(p).unlink(missing_ok=True)


def test_sync_chat_facade():
    print("\n[2/3] 同步 chat() 门面（CLI / 测试路径）")
    agent, path = _make_agent(99)
    try:
        r = agent.chat("你好")
        if r.reply != "你好，我是一二":
            _fail(f"回复不对: {r.reply}")
        _ok("chat() 经 asyncio.run(achat) 拿到结构化回复")
    finally:
        agent.close()
        Path(path).unlink(missing_ok=True)


def test_sandbox_async_wrap():
    print("\n[3/3] 沙箱 async create 插桩")
    trace = RunTrace(case_id="async-wrap", turns=["hi"])
    sb = Sandbox()
    wrapped = sb._wrap_create(_fake_create, trace)

    async def _run():
        resp = await wrapped(messages=[{"role": "user", "content": "hi"}])
        return resp

    resp = asyncio.run(_run())
    if not trace.llm_calls:
        _fail("async wrapper 没有记到 llm_calls")
    rec = trace.llm_calls[0]
    if rec.total_tokens != 8:
        _fail(f"usage 未记录: {rec.total_tokens}")
    if rec.latency_ms < SLEEP * 1000 * 0.5:
        _fail(f"latency 过小，像没 await: {rec.latency_ms}")
    if resp.choices[0].message.content != "你好，我是一二":
        _fail("wrapper 没把 response 传回去")
    _ok(
        f"async def create 已记录 latency={rec.latency_ms:.0f}ms "
        f"tokens={rec.total_tokens}"
    )

    # openai.AsyncOpenAI.create 实际不是 coroutinefunction，只是返回 awaitable
    trace2 = RunTrace(case_id="awaitable-wrap", turns=["hi"])

    def _returns_awaitable(*args, **kwargs):
        return _fake_create(*args, **kwargs)

    wrapped2 = sb._wrap_create(_returns_awaitable, trace2)

    async def _run2():
        return await wrapped2(messages=[{"role": "user", "content": "hi"}])

    asyncio.run(_run2())
    if not trace2.llm_calls:
        _fail("awaitable 风格 create 没有记到 llm_calls")
    _ok(
        f"返回 awaitable 的 create 也记到了 latency={trace2.llm_calls[0].latency_ms:.0f}ms"
    )


def main():
    print("=" * 60)
    print("异步并发")
    print("=" * 60)
    asyncio.run(test_five_concurrent())
    test_sync_chat_facade()
    test_sandbox_async_wrap()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
