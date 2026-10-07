"""限流 / 超时 / 重试。

1. 令牌桶：打满后拒绝，过一会儿补令牌
2. Redis Lua 原子扣令牌（本机 Redis 可用时）
3. HTTP 超限 429
4. LLM 429 → 指数退避后成功；超时耗尽则抛
5. 工具 wait_for 超时返回 error JSON，不把事件循环卡死

用法：python tests/test_stability.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent.resilience import with_timeout_retry  # noqa: E402
from app.agent.tools.manager import ToolManager  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.server.ratelimit import TokenBucket, reset_limiter  # noqa: E402


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


class _LocalBucket(TokenBucket):
    def _try_redis(self, key: str, tokens: float):
        return None


class _Fake429(Exception):
    status_code = 429


def test_bucket_exhaust_and_refill():
    print("\n[1/6] 令牌桶打满后拒绝，补令牌后再放行")

    async def _run():
        b = _LocalBucket(rate=20.0, capacity=2, enabled=True)
        a1 = await b.acquire("u")
        a2 = await b.acquire("u")
        a3 = await b.acquire("u")
        if not (a1 and a2 and not a3):
            _fail(f"突发 2 次应放行、第 3 次拒绝，实际 {a1, a2, a3}")
        await asyncio.sleep(0.12)  # 20 token/s × 0.12s ≈ 2.4
        a4 = await b.acquire("u")
        if not a4:
            _fail("补令牌后仍拒绝")
        off = _LocalBucket(rate=0, capacity=0, enabled=False)
        if not await off.acquire("u"):
            _fail("enabled=False 应永远放行")

    asyncio.run(_run())
    _ok("capacity=2 打满拒绝；短等后续上；关闭限流则放行")


def test_redis_lua():
    print("\n[2/6] Redis Lua 令牌桶（本机 Redis 可选）")
    from app.server.cache import SessionCache

    cache = SessionCache(enabled=True, url="redis://127.0.0.1:6379/15", ttl=60)
    if cache.status() != "ok":
        print("  ⏭️  Redis 不可用，跳过 Lua 桶")
        cache.close()
        return

    from app.server.cache import reset_cache
    reset_cache(cache)

    async def _run():
        b = TokenBucket(rate=0.0, capacity=1, enabled=True)
        key = f"rate-limit-test-{time.time_ns()}"
        a1 = await b.acquire(key)
        a2 = await b.acquire(key)
        if not (a1 and not a2):
            _fail(f"Lua 桶 capacity=1 应变 1 次 True 1 次 False，实际 {a1, a2}")

    try:
        asyncio.run(_run())
        _ok("Lua EVAL 原子扣令牌，第二下拒绝")
    finally:
        reset_cache(None)
        cache.close()


def test_http_429():
    print("\n[3/6] HTTP 超限返回 429")
    reset_limiter(_LocalBucket(rate=0, capacity=0, enabled=True))
    try:
        from fastapi.testclient import TestClient
        from app.server.main import app

        with TestClient(app) as client:
            r = client.post(
                "/api/chat",
                json={"session_id": "rl", "user_id": "u-rl", "message": "hi"},
            )
        if r.status_code != 429:
            _fail(f"期望 429，实际 {r.status_code} body={r.text[:200]}")
        if "Retry-After" not in r.headers:
            _fail("429 缺少 Retry-After")
        _ok("POST /api/chat 超限 → 429 + Retry-After")
    finally:
        reset_limiter(None)


def test_llm_retry():
    print("\n[4/6] LLM 429 指数退避后成功")
    n = {"c": 0}

    async def factory():
        n["c"] += 1
        if n["c"] < 3:
            raise _Fake429("slow down")
        return "ok"

    async def _run():
        return await with_timeout_retry(factory, timeout=5, retries=3, base=0.01)

    out = asyncio.run(_run())
    if out != "ok" or n["c"] != 3:
        _fail(f"应第 3 次成功，实际 out={out} calls={n['c']}")
    _ok("前两次 429，退避后第 3 次成功")


def test_llm_timeout():
    print("\n[5/6] LLM 超时耗尽重试后抛 TimeoutError")
    n = {"c": 0}

    async def factory():
        n["c"] += 1
        await asyncio.sleep(1)
        return "late"

    async def _run():
        await with_timeout_retry(factory, timeout=0.05, retries=2, base=0.01)

    try:
        asyncio.run(_run())
        _fail("超时后应抛 TimeoutError")
    except asyncio.TimeoutError:
        if n["c"] != 3:
            _fail(f"应尝试 1+2 次，实际 {n['c']}")
        _ok(f"timeout=0.05s retries=2，共 {n['c']} 次后放弃")


def test_tool_timeout():
    print("\n[6/6] 工具执行超时返回 error JSON")
    old = settings.tool_timeout
    settings.tool_timeout = 0.05
    tm = ToolManager(use_mcp=False)

    def slow(name, arguments):
        time.sleep(1.0)
        return "never"

    tm.execute_tool = slow  # type: ignore[method-assign]
    try:
        raw = asyncio.run(tm.aexecute_tool("query_order", {"id": "x"}))
        data = json.loads(raw)
        if "超时" not in data.get("error", ""):
            _fail(f"期望超时错误，实际 {raw}")
        _ok("工具 >10s（测试里 0.05s）→ JSON error，不抛到事件循环")
    finally:
        settings.tool_timeout = old


def main():
    print("=" * 60)
    print("限流 / 超时 / 重试")
    print("=" * 60)
    test_bucket_exhaust_and_refill()
    test_redis_lua()
    test_http_429()
    test_llm_retry()
    test_llm_timeout()
    test_tool_timeout()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
