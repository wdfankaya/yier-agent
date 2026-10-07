"""FastAPI 服务入口（SSE 版：流式输出 + 事件回调架构）。

运行：.venv/Scripts/python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8000
（单 worker —— 进程内 Agent dict 不支持多 worker）

POST /api/chat 现在返回 SSE 事件流（text/event-stream）：agent 的每一步
（思考/工具调用/工具结果/待确认/最终答复/错误）都变成一个 {type: ...} JSON 事件逐帧推送。
敏感操作另有 POST /api/confirm（不受聊天限流）。调试页 GET /demo。

线程模型：
- agent.achat() 跑在 uvicorn 事件循环上：LLM 调用 await AsyncOpenAI，互不占用线程
- 构造 Agent / PG 读写 / 同步工具 下沉 asyncio.to_thread
- 事件仍通过 call_soon_threadsafe 入队（同一 loop 里也安全，工具线程回调也不乱序）
- CLI / 测试继续调同步 chat()（内部 asyncio.run）

限制：同一 session 同时只允许一个 in-flight 请求（并发会互相覆盖 agent.on_event）。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.config.settings import settings
from app.server.schemas import (
    ChatRequest,
    ConfirmRequest,
    ConfirmResponse,
    DeleteSessionResponse,
    HealthResponse,
    SessionMessagesResponse,
)
from app.server.sessions import SESSION_DIR, SessionRegistry

registry = SessionRegistry()

# SSE 流结束哨兵：用对象而非 dict，和事件本身区分；走同一队列保证 final/error 先到
_END = object()


def _sse_line(event: dict) -> str:
    """把事件 dict 序列化成一条 SSE 帧。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    mode = "Multi-Agent" if settings.multi_agent_enabled else "ReAct"
    from app.server.cache import get_cache
    redis_status = get_cache().status()
    print(
        f"🟢 服务启动 | mode={mode} | model={settings.model_name} | redis={redis_status}"
    )
    yield
    # 优雅关闭：两个阶段都做完，进程才算干净退出。
    # 1) 每个常驻 Agent await aclose()（后台 LTM 任务收尾 + 再巩固一次，5s 预算）
    # 2) 全部落库后再关 PG 连接池（顺序不能反：close 还要写库）。
    # 3) 最后关 Redis（缓存不是真相源）。
    print("🛑 收到关闭信号：开始优雅关闭")
    await registry.close_all()
    from app.server.repository import aclose_engine
    await aclose_engine()
    from app.server.cache import aclose_cache
    aclose_cache()
    print("🟢 服务已优雅关闭")


app = FastAPI(
    title="一二商城 · 智能客服「一二」服务化",
    version="0.9.0",
    lifespan=lifespan,
)

_DEMO_HTML = Path(__file__).resolve().parents[2] / "web" / "demo.html"


@app.get("/demo")
def demo_page():
    """调试页：SSE 事件流 + 退款确认按钮（非产品 UI）。"""
    if not _DEMO_HTML.is_file():
        raise HTTPException(status_code=404, detail="web/demo.html 不存在")
    return FileResponse(_DEMO_HTML, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health():
    from app.server.cache import get_cache
    return HealthResponse(
        status="ok",
        resident_sessions=len(registry),
        multi_agent_enabled=settings.multi_agent_enabled,
        redis=get_cache().status(),
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式对话：事件逐个推；session 不存在则自动创建（含历史恢复）。"""
    from app.server.ratelimit import get_limiter
    if not await get_limiter().acquire(req.user_id):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": "60"},
        )
    return StreamingResponse(
        _chat_event_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉网关/代理缓冲，让事件即时到达而不是攒一批
            "X-Accel-Buffering": "no",
        },
    )


async def _chat_event_stream(req: ChatRequest):
    """SSE 生成器：achat 与出队并行，事件顺序 thought → tool_call → tool_result → final。

    LLM 在事件循环上 await（多会话并发不再各占一条阻塞线程）。
    构造/恢复 Agent 仍可能 asyncio.run(PG)，丢 to_thread 避免和本 loop 打架。
    on_event 一律 call_soon_threadsafe 入队，同线程/工具线程都保序。
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _emit(event) -> None:
        loop.call_soon_threadsafe(q.put_nowait, event)

    async def _run() -> None:
        try:
            agent = await asyncio.to_thread(
                registry.get_or_create, req.session_id, req.user_id,
            )
        except Exception as e:  # noqa: BLE001 —— 转成 error 事件推给客户端
            _emit({"type": "error", "message": f"会话初始化失败: {type(e).__name__}: {e}"})
            _emit(_END)
            return

        _emit({"type": "session", "session_id": req.session_id, "user_id": req.user_id})
        agent.on_event = _emit
        try:
            await agent.achat(req.message)
        except Exception as e:  # noqa: BLE001
            _emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            agent.on_event = None
            _emit(_END)

    task = asyncio.create_task(_run())

    try:
        while True:
            event = await q.get()
            if event is _END:
                break
            yield _sse_line(event)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@app.post("/api/confirm", response_model=ConfirmResponse)
async def confirm(req: ConfirmRequest):
    """HITL：批准则当场执行敏感工具；拒绝则本会话拉黑该订单。不受聊天限流。"""
    entry = registry.get_or_restore(req.session_id)
    if entry is None:
        # 热态未命中且无法回源时，带 user_id 新建没有意义——没有 pending token
        raise HTTPException(status_code=404, detail=f"会话不存在: {req.session_id}")
    agent, _ = entry
    if not hasattr(agent, "resolve_hitl"):
        raise HTTPException(status_code=400, detail="该会话不支持确认流")
    info = await asyncio.to_thread(agent.resolve_hitl, req.token, req.approved)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=info.get("error") or "确认失败")
    return ConfirmResponse(
        ok=True,
        approved=info.get("approved"),
        executed=bool(info.get("executed")),
        order_id=info.get("order_id"),
        tool=info.get("tool"),
        amount=info.get("amount"),
        result=info.get("result"),
    )


@app.get("/api/sessions/{session_id}", response_model=SessionMessagesResponse)
def get_session(session_id: str):
    """取会话消息列表；进程重启后可凭历史文件恢复（数据库模式下从 PostgreSQL 恢复）。"""
    entry = registry.get_or_restore(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    agent, user_id = entry
    return SessionMessagesResponse(
        session_id=session_id,
        user_id=user_id,
        history_size=len(agent.raw_messages),
        messages=agent.raw_messages,
    )


@app.delete("/api/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    """清会话：重置 + 摘除常驻 agent + 删历史文件。"""
    entry = registry.get_or_restore(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    agent, _ = entry
    agent.reset()
    registry.remove(session_id)
    return DeleteSessionResponse(deleted=True, session_id=session_id)


if __name__ == "__main__":
    uvicorn.run("app.server.main:app", host="127.0.0.1", port=8000, reload=False)
