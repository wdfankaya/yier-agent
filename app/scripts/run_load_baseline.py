"""服务层压测摸底（LLM 打桩，测 SSE / 会话注册表，不含真实模型）。

起一个临时 uvicorn，打桩 AsyncOpenAI，10 个虚拟用户打 15 秒。
Locust 场景见仓库根目录 locustfile.py（需先起真服务，且限流会 429）。

用法：python -m app.scripts.run_load_baseline
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.server.ratelimit import reset_limiter  # noqa: E402
from app.server.sessions import SessionRegistry  # noqa: E402


def _fake_response(content: str, parsed=None):
    msg = SimpleNamespace(content=content, tool_calls=None, parsed=parsed)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-d7")


async def _fake_create(*args, **kwargs):
    await asyncio.sleep(0.02)
    messages = kwargs.get("messages") or []
    blob = " ".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in messages
    )
    if "提取结构化信息" in blob:
        return _fake_response(
            '{"intent":"greeting","confidence":0.9,'
            '"reply":"你好","requires_human":false,"follow_up_question":null}'
        )
    return _fake_response("你好")


async def _fake_parse(*args, **kwargs):
    from app.schemas.response import CustomerServiceResponse, IntentType
    parsed = CustomerServiceResponse(
        intent=IntentType.GREETING,
        confidence=0.9,
        reply="你好",
        requires_human=False,
    )
    return _fake_response("你好", parsed=parsed)


def _patch_registry():
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


async def _httpx_baseline(host: str, users: int, duration: float) -> dict:
    latencies: list[float] = []
    errors = 0
    ok = 0
    stop_at = time.perf_counter() + duration
    lock = asyncio.Lock()

    async def worker(i: int) -> None:
        nonlocal errors, ok
        sid = f"bl-{i}"
        uid = f"bu-{i}"
        n = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.perf_counter() < stop_at:
                t0 = time.perf_counter()
                try:
                    if n % 4 == 3:
                        r = await client.get(f"{host}/api/sessions/{sid}")
                        good = r.status_code in (200, 404)
                    else:
                        r = await client.post(
                            f"{host}/api/chat",
                            json={"session_id": sid, "user_id": uid, "message": "你好"},
                        )
                        good = r.status_code == 200 and any(
                            e.get("type") == "final" for e in _parse_sse(r.text)
                        )
                    elapsed = time.perf_counter() - t0
                    async with lock:
                        if good:
                            ok += 1
                            latencies.append(elapsed)
                        else:
                            errors += 1
                except Exception:
                    async with lock:
                        errors += 1
                n += 1

    await asyncio.gather(*[worker(i) for i in range(users)])
    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int((p / 100) * (len(latencies) - 1)))
        return round(latencies[idx] * 1000, 1)

    total = ok + errors
    return {
        "engine": "httpx",
        "users": users,
        "duration_s": duration,
        "ok": ok,
        "errors": errors,
        "rps": round(total / duration, 2) if duration else 0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mix": "POST /api/chat SSE ×3 + GET /api/sessions ×1",
    }


def main() -> int:
    users = 10
    duration = 15.0
    old = (
        settings.db_enabled,
        settings.redis_enabled,
        settings.memory_enabled,
        settings.mcp_enabled,
        settings.hitl_enabled,
        settings.rate_limit_enabled,
        settings.ltm_consolidate_every,
        settings.multi_agent_enabled,
        settings.history_threshold,
    )
    settings.db_enabled = False
    settings.redis_enabled = False
    settings.memory_enabled = False
    settings.mcp_enabled = False
    settings.hitl_enabled = False
    settings.rate_limit_enabled = False
    settings.ltm_consolidate_every = 0
    settings.multi_agent_enabled = False
    settings.history_threshold = 1000
    reset_limiter()
    orig = _patch_registry()
    port = _free_port()
    host = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            "app.server.main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                r = httpx.get(f"{host}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            print("服务未起来")
            return 1

        report = asyncio.run(_httpx_baseline(host, users, duration))
        report["note"] = (
            "服务层摸底：LLM 已打桩（~20ms），不含真实模型 token 时间；限流关闭。"
            "含 LLM 的基线：先启动 python -m app.server.main，再 "
            "locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 10 -r 10 -t 30s"
            "（生产令牌桶 10 次/分钟，真模型压测会先打到 429）。"
        )
        out = ROOT / "eval_records" / "locust_baseline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"已写入 {out}")
        total = int(report.get("ok") or 0) + int(report.get("errors") or 0)
        err = int(report.get("errors") or 0)
        return 0 if total > 0 and err / total < 0.01 else 1
    finally:
        server.should_exit = True
        thread.join(timeout=5)
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
            settings.history_threshold,
        ) = old
        reset_limiter()


if __name__ == "__main__":
    raise SystemExit(main())
